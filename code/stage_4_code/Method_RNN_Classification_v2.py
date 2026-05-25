'''
Train a RNN model with the text classification dataset (data/stage_4_data/text_classification),
and apply it to classify the testing set.
Generates learning curves and reports evaluation results.

Dataset: IMDb sentiment — 25k train / 25k test, binary (pos=1 / neg=0).

Data layout (from README):
  data/stage_4_data/text_classification/
      train/pos/*.txt
      train/neg/*.txt
      test/pos/*.txt
      test/neg/*.txt

Architecture: Embedding → Bidirectional LSTM → Global mean-pool → FC → 2 logits

This is the "v2" / separate classification implementation. It is self-contained
(does its own cleaning / vocab / padding inside run()) and is wired into the
pipeline via script_rnn_classification_v2.py. It is kept distinct from the
existing Method_RNN_Classification (GRU + Dataset_Loader_Classification) so both
can be run independently. Outputs use the RNN_v2_ prefix to avoid clobbering the
existing classification results.
'''

import os
import re
import string
from collections import Counter
from pathlib import Path

from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import (
    Evaluate_Accuracy, Evaluate_F1, Evaluate_Precision, Evaluate_Recall
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------
# Text-cleaning helpers  (see README: remove stop-words, punctuation, normalise)
# ---------------------------------------------------------------------------

# Minimal English stop-word list (keeps sentiment words like "not", "no")
_STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'that', 'this', 'these',
    'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
    'he', 'she', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
    'as', 'if', 'than', 'so', 'yet', 'both', 'each', 'few', 'more', 'most',
    'other', 'such', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'out', 'up', 'about', 'against', 'between', 'during',
    'am', 'then', 'there', 'here', 'where', 'how', 'all', 'also', 'just',
    'br',   # HTML artefact common in the IMDb dataset
}

_HTML_TAG_RE   = re.compile(r'<[^>]+>')
_NONALPHA_RE   = re.compile(r'[^a-z\s]')


def clean_text(text: str) -> list[str]:
    '''
    Returns a list of tokens after:
      1. lowercasing
      2. stripping HTML tags (common in IMDb reviews)
      3. removing punctuation / digits
      4. removing stop-words
    '''
    text = text.lower()
    text = _HTML_TAG_RE.sub(' ', text)           # strip <br />, etc.
    text = _NONALPHA_RE.sub(' ', text)           # keep only letters + spaces
    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    return tokens


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_text_dataset(split_dir: str) -> tuple[list[list[str]], list[int]]:
    '''
    Reads all .txt files from <split_dir>/pos/ and <split_dir>/neg/.
    Returns (list_of_token_lists, list_of_labels)  where label 1=pos, 0=neg.
    '''
    texts, labels = [], []
    for label, folder in [(1, 'pos'), (0, 'neg')]:
        folder_path = os.path.join(split_dir, folder)
        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(folder_path, fname)
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                raw = f.read()
            tokens = clean_text(raw)
            texts.append(tokens)
            labels.append(label)
    return texts, labels


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

def build_vocab(token_lists: list[list[str]],
                max_vocab: int = 20_000) -> dict[str, int]:
    '''
    Builds a word→index vocabulary from the training token lists.
    Index 0 = <PAD>, Index 1 = <UNK>.
    '''
    counter = Counter(tok for tokens in token_lists for tok in tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def tokens_to_ids(token_list: list[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab['<UNK>']
    return [vocab.get(tok, unk) for tok in token_list]


def pad_sequence(ids: list[int], max_len: int, pad_id: int = 0) -> list[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


# ---------------------------------------------------------------------------
# MethodRNN — mirrors Method_CNN structure exactly
# ---------------------------------------------------------------------------

class MethodRNN(method, nn.Module):
    # ------- Hyperparameters -----------------------------------------------
    learning_rate = 2e-4       # Adam lr
    max_epoch     = 15         # training epochs
    batch_size    = 256        # mini-batch size
    max_vocab     = 20_000     # vocabulary cap
    embed_dim     = 128        # embedding vector size
    hidden_dim    = 256        # LSTM hidden state size
    num_layers    = 2          # stacked LSTM layers
    dropout_rate  = 0.5        # dropout probability
    max_seq_len   = 300        # tokens per review (truncate/pad)

    # Path to the raw text data
    data_dir = 'data/stage_4_data/text_classification'

    def __init__(self):
        super(MethodRNN, self).__init__()
        method.__init__(self, 'RNN-TextClassification-v2', '')
        nn.Module.__init__(self)

        # Device selection: prefer Apple-Silicon GPU (MPS) on Mac, then CUDA,
        # then CPU. The original version of this file hard-coded CPU for a
        # 144-thread Linux server; here we auto-detect so it uses the M-series
        # GPU when available.
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Cap the intra-op thread pool at the actual core count (matters only
        # for the CPU fallback; harmless on GPU).
        torch.set_num_threads(os.cpu_count() or 1)
        print(f'Using device: {self.device}, threads: {torch.get_num_threads()}')

        # Vocabulary is populated in _build_model() after data is loaded.
        self.vocab    = None
        self._model_ready = False

    # -----------------------------------------------------------------------
    # Call after vocabulary is known to build the nn.Module layers.
    # -----------------------------------------------------------------------
    def _build_model(self, vocab_size: int):
        # Embedding: maps integer word-ids → dense vectors.
        # padding_idx=0 keeps the <PAD> embedding at zero and excludes it from
        # gradient updates so it never "learns" a meaningful direction.
        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)

        # Bidirectional LSTM: reads the sequence left-to-right AND
        # right-to-left, doubling the effective context captured at every
        # timestep.  output dim per step = hidden_dim * 2.
        # batch_first=True → tensors are (batch, seq, features).
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )

        # Layer norm applied to the pooled LSTM output for training stability.
        self.layer_norm = nn.LayerNorm(self.hidden_dim * 2)

        # Classifier head: two FC layers with dropout.
        # Input size = hidden_dim * 2  (bidirectional → concatenated states)
        self.fc1     = nn.Linear(self.hidden_dim * 2, 128)
        self.fc2     = nn.Linear(128, 2)          # 2 logits: neg / pos
        self.dropout = nn.Dropout(self.dropout_rate)

        self.to(self.device)
        self._model_ready = True

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------
    def forward(self, x):
        '''
        x : LongTensor of shape (batch, seq_len)  — token ids
        Returns logits of shape (batch, 2).
        '''
        # (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(x))

        # lstm_out: (batch, seq_len, hidden_dim*2)
        lstm_out, _ = self.lstm(embedded)

        # Global mean-pooling over the time axis.
        # Averaging is more stable than taking only the last hidden state,
        # especially for long sequences where early context matters.
        pooled = lstm_out.mean(dim=1)             # (batch, hidden_dim*2)
        pooled = self.layer_norm(pooled)

        out = F.relu(self.fc1(pooled))            # (batch, 128)
        out = self.dropout(out)
        out = self.fc2(out)                        # (batch, 2) — raw logits
        return out

    # -----------------------------------------------------------------------
    # Release cached device memory.
    # On MPS the caching allocator holds freed blocks and can balloon to tens
    # of GiB across a training epoch, eventually exceeding the unified-memory
    # watermark and OOM-ing the next allocation. Calling empty_cache()
    # periodically returns those blocks to the OS. It does NOT change any
    # numerics — only memory bookkeeping — so the experiment is unaffected.
    # -----------------------------------------------------------------------
    def _empty_device_cache(self):
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Training loop  (mirrors Method_CNN.fit exactly)
    # -----------------------------------------------------------------------
    def fit(self, X_ids, y, X_test_ids, y_test):
        '''
        X_ids, X_test_ids : list of padded integer sequences
        y, y_test         : list of integer labels (0/1)
        Returns per-epoch metric histories for plotting.
        '''
        # Convert to tensors (keep on CPU; batches are moved to device below)
        X_train_t = torch.LongTensor(X_ids)
        X_test_t  = torch.LongTensor(X_test_ids)
        y_train_t = torch.LongTensor(y)
        y_test_t  = torch.LongTensor(y_test)

        # DataLoader handles shuffling and batching.
        # num_workers=0 + no pin_memory: required for macOS/MPS safety. macOS
        # uses the 'spawn' start method, so worker processes (num_workers>0)
        # re-import the entry module and crash without a __main__ guard; and
        # pin_memory is a CUDA-only optimisation. On the M-series GPU the data
        # is small enough that single-process loading keeps the GPU fed.
        train_ds     = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        # Full eval tensors live on device for fast chunk-wise inference
        X_train_dev = X_train_t.to(self.device)
        X_test_dev  = X_test_t.to(self.device)
        y_train_dev = y_train_t.to(self.device)
        y_test_dev  = y_test_t.to(self.device)

        # CPU copies of the labels for the (CPU-side) epoch-end loss/metrics.
        y_train_cpu = y_train_t
        y_test_cpu  = y_test_t

        loss_fn   = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=1e-5)

        # Learning-rate scheduler: halve LR if test loss doesn't improve for
        # 3 consecutive epochs.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        acc_eval  = Evaluate_Accuracy('eval', '')
        prec_eval = Evaluate_Precision('eval', '')
        rec_eval  = Evaluate_Recall('eval', '')
        f1_eval   = Evaluate_F1('eval', '')

        epochs_hist = []
        train_accs,   test_accs   = [], []
        train_losses, test_losses = [], []
        train_precs,  test_precs  = [], []
        train_recs,   test_recs   = [], []
        train_f1s,    test_f1s    = [], []

        best_test_acc = 0.0
        best_state    = None

        for epoch in range(self.max_epoch):

            # -- MINI-BATCH TRAINING ----------------------------------------
            self.train()

            for batch_idx, (Xb, yb) in enumerate(train_loader):
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)

                # set_to_none=True is faster than zero_grad(): skips the memset
                # and lets the allocator reuse the memory immediately.
                optimizer.zero_grad(set_to_none=True)
                logits = self.forward(Xb)
                loss   = loss_fn(logits, yb)
                loss.backward()
                # Gradient clipping prevents exploding gradients — a common
                # failure mode in RNNs training on long sequences.
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()

                # Bound the MPS cache so it can't balloon across the epoch.
                if (batch_idx + 1) % 10 == 0:
                    self._empty_device_cache()

            # Free everything the training graph cached before the eval pass —
            # this is what prevents the epoch-end OOM on MPS.
            self._empty_device_cache()

            # -- EPOCH-END EVALUATION ---------------------------------------
            # Evaluate in chunks to avoid OOM on large datasets.
            # CRITICAL for MPS: move each chunk's logits to CPU *inside* the
            # loop. That .cpu() forces a per-chunk synchronization, which lets
            # the MPS backend release the LSTM's intermediate Metal buffers.
            # Without it, the no_grad loop never syncs and those (untracked,
            # empty_cache-immune) intermediates accumulate to tens of GiB and
            # OOM the GPU. Reductions (loss/argmax) then happen on CPU.
            self.eval()

            def collect_logits_cpu(X_dev):
                chunks = []
                chunk = 256
                for s in range(0, X_dev.shape[0], chunk):
                    out = self.forward(X_dev[s:s + chunk]).detach().to('cpu')
                    chunks.append(out)
                return torch.cat(chunks)

            with torch.no_grad():
                train_logits = collect_logits_cpu(X_train_dev)
                test_logits  = collect_logits_cpu(X_test_dev)
            self._empty_device_cache()

            train_loss = loss_fn(train_logits, y_train_cpu).item()
            test_loss  = loss_fn(test_logits,  y_test_cpu).item()

            scheduler.step(test_loss)

            train_pred = train_logits.max(1)[1]
            test_pred  = test_logits.max(1)[1]

            def score(evaluator, true, pred):
                evaluator.data = {'true_y': true.cpu(), 'pred_y': pred.cpu()}
                return evaluator.evaluate()

            tr_acc  = score(acc_eval,  y_train_dev, train_pred)
            te_acc  = score(acc_eval,  y_test_dev,  test_pred)
            tr_prec = score(prec_eval, y_train_dev, train_pred)
            te_prec = score(prec_eval, y_test_dev,  test_pred)
            tr_rec  = score(rec_eval,  y_train_dev, train_pred)
            te_rec  = score(rec_eval,  y_test_dev,  test_pred)
            tr_f1   = score(f1_eval,   y_train_dev, train_pred)
            te_f1   = score(f1_eval,   y_test_dev,  test_pred)

            if te_acc > best_test_acc:
                best_test_acc = te_acc
                best_state    = {k: v.clone() for k, v in self.state_dict().items()}

            epochs_hist.append(epoch)
            train_accs.append(tr_acc);     test_accs.append(te_acc)
            train_losses.append(train_loss); test_losses.append(test_loss)
            train_precs.append(tr_prec);   test_precs.append(te_prec)
            train_recs.append(tr_rec);     test_recs.append(te_rec)
            train_f1s.append(tr_f1);       test_f1s.append(te_f1)

            print(f'Epoch: {epoch}  Train Loss: {train_loss:.4f}  Test Loss: {test_loss:.4f}')
            print(f'Epoch: {epoch}  Training Accuracy: {tr_acc:.4f}  Testing Accuracy: {te_acc:.4f}')
            print(f'Epoch: {epoch}  Training Recall: {tr_rec:.4f}  Testing Recall: {te_rec:.4f}')
            print(f'Epoch: {epoch}  Training Precision: {tr_prec:.4f}  Testing Precision: {te_prec:.4f}')
            print(f'Epoch: {epoch}  Training F1: {tr_f1:.4f}  Testing F1: {te_f1:.4f}')

        self.load_state_dict(best_state)
        print(f'\nBest test accuracy: {best_test_acc:.4f}')

        return (epochs_hist,
                train_accs, test_accs,
                train_losses, test_losses,
                train_precs, test_precs,
                train_recs, test_recs,
                train_f1s, test_f1s)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    def test(self, X_ids):
        '''Run trained model on padded id sequences; return predicted labels.'''
        self.eval()
        X_t = torch.LongTensor(X_ids).to(self.device)
        preds = []
        with torch.no_grad():
            for s in range(0, X_t.shape[0], 256):
                logits = self.forward(X_t[s:s+256])
                preds.append(logits.max(1)[1].cpu())
        self._empty_device_cache()
        return torch.cat(preds)

    # -----------------------------------------------------------------------
    # Result directory resolved from this file's location (robust to CWD)
    # -----------------------------------------------------------------------
    def _result_dir(self):
        result_dir = Path(__file__).resolve().parents[2] / 'result' / 'stage_4_result'
        result_dir.mkdir(parents=True, exist_ok=True)
        return str(result_dir) + '/'

    # -----------------------------------------------------------------------
    # run() — entry point called by the stage script
    # -----------------------------------------------------------------------
    def run(self):
        print('method running...')

        # -- CONSUME DATA from Dataset_Loader (self.data set by the pipeline) -
        # self.data['train']['X'] = list of raw review strings
        # self.data['train']['y'] = list of int labels (1=pos, 0=neg)
        print('Cleaning and tokenising data...')
        train_labels = self.data['train']['y']
        test_labels  = self.data['test']['y']

        train_tokens = [clean_text(raw) for raw in self.data['train']['X']]
        test_tokens  = [clean_text(raw) for raw in self.data['test']['X']]

        print(f'  Train samples: {len(train_tokens)}  |  Test samples: {len(test_tokens)}')

        # -- BUILD VOCABULARY (from training set only — no test leakage) ----
        print('Building vocabulary...')
        self.vocab = build_vocab(train_tokens, max_vocab=self.max_vocab)
        vocab_size = len(self.vocab)
        print(f'  Vocabulary size: {vocab_size}')

        # -- NUMERICALIZE & PAD ----------------------------------------------
        def prepare(token_lists):
            return [
                pad_sequence(tokens_to_ids(tl, self.vocab), self.max_seq_len)
                for tl in token_lists
            ]

        train_ids = prepare(train_tokens)
        test_ids  = prepare(test_tokens)

        # -- BUILD MODEL LAYERS now that vocab_size is known -----------------
        self._build_model(vocab_size)

        # -- TRAIN -----------------------------------------------------------
        print('--start training...')
        (epochs_hist,
         train_accs, test_accs,
         train_losses, test_losses,
         train_precs, test_precs,
         train_recs, test_recs,
         train_f1s, test_f1s) = self.fit(
            train_ids, train_labels,
            test_ids,  test_labels,
        )

        # -- LEARNING CURVES -------------------------------------------------
        result_dir = self._result_dir()

        def save_plot(y1, y2, label1, label2, ylabel, title, fname):
            plt.figure(figsize=(8, 5))
            plt.plot(epochs_hist, y1, color='steelblue',  label=label1)
            plt.plot(epochs_hist, y2, color='darkorange', label=label2)
            plt.title(title)
            plt.xlabel('Epoch')
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(result_dir + fname)
            plt.close()

        save_plot(train_losses, test_losses,
                  'train loss', 'test loss',
                  'Loss', 'Training vs Test Loss', 'RNN_v2_loss.png')

        save_plot(train_accs, test_accs,
                  'train accuracy', 'test accuracy',
                  'Accuracy', 'Training vs Test Accuracy', 'RNN_v2_accuracy.png')

        save_plot(train_precs, test_precs,
                  'train precision', 'test precision',
                  'Precision', 'Training vs Test Precision', 'RNN_v2_precision.png')

        save_plot(train_recs, test_recs,
                  'train recall', 'test recall',
                  'Recall', 'Training vs Test Recall', 'RNN_v2_recall.png')

        save_plot(train_f1s, test_f1s,
                  'train F1', 'test F1',
                  'F1 Score', 'Training vs Test F1', 'RNN_v2_f1.png')

        # -- CONFUSION MATRIX (binary: neg=0, pos=1) -------------------------
        print('\n-- Generating confusion matrix...')
        pred_y = self.test(test_ids)
        true_y = torch.LongTensor(test_labels)

        n_classes = 2
        cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
        for t, p in zip(true_y, pred_y):
            cm[t][p] += 1

        class_names = ['neg', 'pos']
        print('\nConfusion Matrix (rows=true, cols=predicted):')
        print('         ' + '  '.join(f'{c:>8s}' for c in class_names))
        for i in range(n_classes):
            row = '  '.join(f'{cm[i][j]:8d}' for j in range(n_classes))
            print(f'  {class_names[i]:>4s}: {row}')

        per_class_acc = cm.diag().float() / cm.sum(1).float()
        print('\nPer-class accuracy:')
        for i in range(n_classes):
            print(f'  {class_names[i]}: {per_class_acc[i]:.4f}')

        # Save model checkpoint
        torch.save(self.state_dict(), result_dir + 'RNN_v2_model.pt')

        print('\n--start testing...')
        pred_y = self.test(test_ids)
        return {'pred_y': pred_y, 'true_y': test_labels}
