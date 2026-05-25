from pathlib import Path
import os
import re
import string
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from code.base_class.method import method

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / 'ecs170_matplotlib'
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_MPLCONFIGDIR))
from matplotlib import pyplot as plt


class Method_RNN(method, nn.Module):
    def __init__(
        self,
        mName='recurrent neural network text generator',
        mDescription='',
        vocab_size=None,
        embed_size=128,
        hidden_size=128,
        rnn_type='lstm',
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
        weight_decay=0.0,
        max_epoch=20,
        batch_size=64,
        early_stopping_patience=2,
        early_stopping_min_delta=0.001,
    ):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
        self.token_to_idx = None
        self.idx_to_token = None

        self.embedding = None
        self.rnn = None
        self.dropout = None
        self.fc = None

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f'using device: {self.device}')

        if self.vocab_size is not None:
            self._build_model(self.vocab_size)

    def _build_model(self, vocab_size):
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, self.embed_size, padding_idx=self.pad_idx)

        if self.rnn_type == 'rnn':
            rnn_class = nn.RNN
        elif self.rnn_type == 'gru':
            rnn_class = nn.GRU
        elif self.rnn_type == 'lstm':
            rnn_class = nn.LSTM
        else:
            raise ValueError("rnn_type must be one of 'rnn', 'gru', or 'lstm'.")

        self.rnn = rnn_class(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.hidden_size, vocab_size)
        self.to(self.device)

    def _configure_from_data(self):
        if self.data is None:
            raise ValueError('Method_RNN.data must be set before training or generation.')

        self.pad_idx = self.data.get('pad_idx', 0)
        self.unk_idx = self.data.get('unk_idx', 1)
        self.bos_idx = self.data.get('bos_idx', 2)
        self.eos_idx = self.data.get('eos_idx', 3)
        self.token_to_idx = self.data['token_to_idx']
        self.idx_to_token = self.data['idx_to_token']

        vocab_size = self.data['vocab_size']
        if self.embedding is None or self.vocab_size != vocab_size:
            self._build_model(vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.dropout(x)
        output, hidden = self.rnn(x, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden

    def _make_loader(self, X, y, shuffle):
        X_tensor = torch.LongTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _run_train_epoch(self, loader, loss_function, optimizer):
        self.train()
        total_loss = 0.0
        total_batches = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            logits, _ = self.forward(batch_X)
            loss = loss_function(logits.reshape(-1, self.vocab_size), batch_y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        return total_loss / max(1, total_batches)

    def _evaluate_loss(self, loader, loss_function):
        self.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                logits, _ = self.forward(batch_X)
                loss = loss_function(logits.reshape(-1, self.vocab_size), batch_y.reshape(-1))
                total_loss += loss.item()
                total_batches += 1

        return total_loss / max(1, total_batches)

    def fit(self, X, y, X_test, y_test):
        self._configure_from_data()

        train_loader = self._make_loader(X, y, shuffle=True)
        train_eval_loader = self._make_loader(X, y, shuffle=False)
        test_loader = self._make_loader(X_test, y_test, shuffle=False)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch)
        loss_function = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        epochs = []
        losses = []
        test_losses = []
        best_state = None
        best_test_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        stopped_epoch = self.max_epoch - 1

        for epoch in range(self.max_epoch):
            batch_train_loss = self._run_train_epoch(train_loader, loss_function, optimizer)
            train_loss = self._evaluate_loss(train_eval_loader, loss_function)
            test_loss = self._evaluate_loss(test_loader, loss_function)
            scheduler.step()

            epochs.append(epoch)
            losses.append(train_loss)
            test_losses.append(test_loss)

            improved = test_loss < best_test_loss - self.early_stopping_min_delta
            if improved:
                best_test_loss = test_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                'Epoch:', epoch,
                'Training Loss:', train_loss,
                'Testing Loss:', test_loss,
                'Batch Training Loss:', batch_train_loss,
            )

            if epochs_without_improvement >= self.early_stopping_patience:
                stopped_epoch = epoch
                print('Early stopping at epoch:', epoch, 'Best Epoch:', best_epoch)
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return epochs, losses, test_losses, best_epoch, best_test_loss, stopped_epoch

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", text.lower())

    def _detokenize(self, tokens):
        output = ''
        punctuation = set(string.punctuation)

        for token in tokens:
            if token in {'<pad>', '<unk>', '<bos>', '<eos>'}:
                continue
            if not output:
                output = token
            elif token in punctuation:
                output += token
            else:
                output += ' ' + token

        return output

    def _sample_next_token(
        self,
        logits,
        generated_indices,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        min_new_tokens,
        new_tokens_generated,
    ):
        logits = logits.squeeze(0).clone()

        for idx in [self.pad_idx, self.unk_idx, self.bos_idx]:
            logits[idx] = -float('inf')
        if new_tokens_generated < min_new_tokens:
            logits[self.eos_idx] = -float('inf')

        if repetition_penalty is not None and repetition_penalty > 1.0:
            for idx in set(generated_indices[-20:]):
                if 0 <= idx < logits.numel():
                    if logits[idx] < 0:
                        logits[idx] *= repetition_penalty
                    else:
                        logits[idx] /= repetition_penalty

        temperature = max(temperature, 1e-5)
        logits = logits / temperature

        if top_k is not None and top_k > 0 and top_k < logits.numel():
            threshold = torch.topk(logits, top_k).values[-1]
            logits[logits < threshold] = -float('inf')

        probabilities = torch.softmax(logits, dim=-1)

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            sorted_probs[remove] = 0.0
            total = sorted_probs.sum()
            if total > 0:
                sorted_probs = sorted_probs / total
                sampled_rank = torch.multinomial(sorted_probs, num_samples=1).item()
                return sorted_indices[sampled_rank].item()

        total = probabilities.sum()
        if not torch.isfinite(total) or total <= 0:
            return torch.argmax(logits).item()

        return torch.multinomial(probabilities, num_samples=1).item()

    def generate(
        self,
        seed_text='',
        max_new_tokens=30,
        temperature=0.55,
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.15,
        min_new_tokens=5,
    ):
        self._configure_from_data()
        self.eval()

        tokens = self._tokenize(seed_text)
        input_indices = [self.bos_idx]
        input_indices.extend(self.token_to_idx.get(token, self.unk_idx) for token in tokens)
        generated_indices = input_indices[:]
        hidden = None

        with torch.no_grad():
            input_tensor = torch.LongTensor([input_indices]).to(self.device)
            logits, hidden = self.forward(input_tensor, hidden)
            next_logits = logits[:, -1, :]

            for new_tokens_generated in range(max_new_tokens):
                next_idx = self._sample_next_token(
                    next_logits,
                    generated_indices,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    min_new_tokens,
                    new_tokens_generated,
                )
                generated_indices.append(next_idx)

                if next_idx == self.eos_idx:
                    break

                input_tensor = torch.LongTensor([[next_idx]]).to(self.device)
                logits, hidden = self.forward(input_tensor, hidden)
                next_logits = logits[:, -1, :]

        generated_tokens = [self.idx_to_token[idx] for idx in generated_indices]
        return self._detokenize(generated_tokens)

    def _save_loss_plot(self, epochs, losses, test_losses):
        result_dir = Path(__file__).resolve().parents[2] / 'result' / 'stage_4_result'
        result_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, color='blue', label='training loss (eval mode)')
        plt.plot(epochs, test_losses, color='orange', label='testing loss')
        plt.title('Epoch vs Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(result_dir / 'Stage_4_loss.png')
        plt.close()

    def _save_report_materials(self, best_epoch, best_test_loss, stopped_epoch, generated_samples):
        result_dir = Path(__file__).resolve().parents[2] / 'result' / 'stage_4_result'
        result_dir.mkdir(parents=True, exist_ok=True)
        perplexity = float(np.exp(min(best_test_loss, 50)))
        report_path = result_dir / 'Stage_4_generation_report_materials.txt'

        lines = [
            'Stage 4 Text Generation Report Materials',
            '',
            'Model architecture:',
            '- Embedding layer: vocab_size x ' + str(self.embed_size),
            '- ' + self.rnn_type.upper() + ' layer: hidden_size=' + str(self.hidden_size)
            + ', num_layers=' + str(self.num_layers),
            '- Dropout: ' + str(self.dropout_rate),
            '- Fully connected output layer: ' + str(self.hidden_size) + ' -> vocab_size',
            '',
            'Training settings:',
            '- Optimizer: Adam',
            '- Learning rate: ' + str(self.learning_rate),
            '- Weight decay: ' + str(self.weight_decay),
            '- Epochs: ' + str(self.max_epoch),
            '- Batch size: ' + str(self.batch_size),
            '- Early stopping patience: ' + str(self.early_stopping_patience),
            '- Loss: CrossEntropyLoss with padding ignored',
            '- Plotted training loss is measured in evaluation mode after each epoch',
            '- Best epoch by testing loss: ' + str(best_epoch),
            '- Stopped epoch: ' + str(stopped_epoch),
            '',
            'Final generation performance:',
            '- Best testing loss: ' + str(best_test_loss),
            '- Testing perplexity: ' + str(perplexity),
            '',
            'Generated samples:',
        ]
        lines.extend('- ' + sample for sample in generated_samples)
        lines.extend([
            '',
            'Generated plot file:',
            '- result/stage_4_result/Stage_4_loss.png',
        ])

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    def test(self, seed_text=''):
        return self.generate(seed_text=seed_text)

    def run(self):
        print('method running...')
        print('--start training...')
        epochs, losses, test_losses, best_epoch, best_test_loss, stopped_epoch = self.fit(
            self.data['train']['X'],
            self.data['train']['y'],
            self.data['test']['X'],
            self.data['test']['y'],
        )
        self._save_loss_plot(epochs, losses, test_losses)

        print('--start generation...')
        seed_text = self.data.get('seed_text', '')
        seed_tokens = self._tokenize(seed_text)[:3]
        seed_three_words = self._detokenize(seed_tokens) or 'why did the'
        generated_samples = [
            self.generate(seed_text='why did the', max_new_tokens=25),
            self.generate(seed_text='what do you', max_new_tokens=25),
            self.generate(seed_text=seed_three_words, max_new_tokens=25),
        ]
        self._save_report_materials(best_epoch, best_test_loss, stopped_epoch, generated_samples)

        model_path = Path(__file__).resolve().parents[2] / 'result' / 'stage_4_result' / 'Stage_4_model.pt'
        torch.save(self.state_dict(), model_path)

        for sample in generated_samples:
            print('Generated:', sample)

        return {
            'generated_text': generated_samples[0],
            'generated_samples': generated_samples,
            'seed_text': seed_text,
            'epochs': epochs,
            'losses': losses,
            'test_losses': test_losses,
            'best_epoch': best_epoch,
            'best_test_loss': best_test_loss,
            'stopped_epoch': stopped_epoch,
            'test_perplexity': float(np.exp(min(best_test_loss, 50))),
            'vocab_size': self.vocab_size,
            'model_settings': {
                'embed_size': self.embed_size,
                'hidden_size': self.hidden_size,
                'rnn_type': self.rnn_type,
                'num_layers': self.num_layers,
                'dropout': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'max_epoch': self.max_epoch,
                'batch_size': self.batch_size,
                'early_stopping_patience': self.early_stopping_patience,
            },
        }
