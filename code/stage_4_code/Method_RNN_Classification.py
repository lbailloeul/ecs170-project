from pathlib import Path
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from code.base_class.method import method

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / 'ecs170_matplotlib'
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_MPLCONFIGDIR))
from matplotlib import pyplot as plt


class Method_RNN_Classification(method, nn.Module):
    def __init__(
        self,
        mName='recurrent neural network text classifier',
        mDescription='',
        vocab_size=None,
        output_size=2,
        embed_size=128,
        hidden_size=128,
        rnn_type='gru',
        num_layers=1,
        dropout=0.4,
        learning_rate=0.001,
        weight_decay=0.0,
        max_epoch=10,
        batch_size=64,
        bidirectional=True,
        pooling='mean_max',
        fc_hidden_size=128,
        early_stopping_patience=2,
        early_stopping_min_delta=0.001,
    ):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.fc_hidden_size = fc_hidden_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.pad_idx = 0
        self.embedding = None
        self.rnn = None
        self.dropout = None
        self.fc = None
        self.classifier = None

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
            bidirectional=self.bidirectional,
        )
        directions = 2 if self.bidirectional else 1
        rnn_output_size = self.hidden_size * directions
        classifier_input_size = rnn_output_size * 3 if self.pooling == 'mean_max' else rnn_output_size
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(classifier_input_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_hidden_size, self.output_size),
        )
        self.to(self.device)

    def _configure_from_data(self):
        if self.data is None:
            raise ValueError('Method_RNN_Classification.data must be set before training.')

        self.pad_idx = self.data.get('pad_idx', 0)
        vocab_size = self.data['vocab_size']
        if self.embedding is None or self.vocab_size != vocab_size:
            self._build_model(vocab_size)

    def _lengths_from_X(self, X):
        pad_idx = self.pad_idx
        return [max(1, sum(1 for token_id in row if token_id != pad_idx)) for row in X]

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        rnn_result = self.rnn(packed)
        packed_output = rnn_result[0]
        if self.rnn_type in {'rnn', 'gru'}:
            hidden = rnn_result[1]
        else:
            hidden = rnn_result[1][0]

        if self.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]

        if self.pooling == 'mean_max':
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=x.size(1),
            )
            time_steps = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            mask = time_steps < lengths.unsqueeze(1)
            mask = mask.unsqueeze(2)

            max_pool = output.masked_fill(~mask, -1e9).max(dim=1).values
            mean_pool = (output * mask.float()).sum(dim=1)
            mean_pool = mean_pool / lengths.clamp(min=1).unsqueeze(1).float()
            features = torch.cat((last_hidden, max_pool, mean_pool), dim=1)
        else:
            features = last_hidden

        return self.classifier(features)

    def _make_loader(self, X, y, lengths, shuffle):
        X_tensor = torch.LongTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))
        length_tensor = torch.LongTensor(np.array(lengths))
        dataset = TensorDataset(X_tensor, y_tensor, length_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    @staticmethod
    def _calculate_metrics(true_y, pred_y):
        return {
            'accuracy': accuracy_score(true_y, pred_y),
            'precision': precision_score(true_y, pred_y, average='macro', zero_division=0),
            'recall': recall_score(true_y, pred_y, average='macro', zero_division=0),
            'f1': f1_score(true_y, pred_y, average='macro', zero_division=0),
        }

    def _run_train_epoch(self, loader, loss_function, optimizer):
        self.train()
        total_loss = 0.0
        total_batches = 0

        for batch_X, batch_y, batch_lengths in loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_lengths = batch_lengths.to(self.device)

            optimizer.zero_grad()
            logits = self.forward(batch_X, batch_lengths)
            loss = loss_function(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        return total_loss / max(1, total_batches)

    def _evaluate(self, loader, loss_function):
        self.eval()
        total_loss = 0.0
        total_batches = 0
        true_y = []
        pred_y = []

        with torch.no_grad():
            for batch_X, batch_y, batch_lengths in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_lengths = batch_lengths.to(self.device)

                logits = self.forward(batch_X, batch_lengths)
                loss = loss_function(logits, batch_y)
                predictions = logits.argmax(dim=1)

                total_loss += loss.item()
                total_batches += 1
                true_y.extend(batch_y.cpu().tolist())
                pred_y.extend(predictions.cpu().tolist())

        return {
            'loss': total_loss / max(1, total_batches),
            'metrics': self._calculate_metrics(true_y, pred_y),
            'true_y': true_y,
            'pred_y': pred_y,
        }

    def fit(self, X, y, X_test, y_test, lengths=None, test_lengths=None):
        self._configure_from_data()

        if lengths is None:
            lengths = self._lengths_from_X(X)
        if test_lengths is None:
            test_lengths = self._lengths_from_X(X_test)

        train_loader = self._make_loader(X, y, lengths, shuffle=True)
        train_eval_loader = self._make_loader(X, y, lengths, shuffle=False)
        test_loader = self._make_loader(X_test, y_test, test_lengths, shuffle=False)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch)
        loss_function = nn.CrossEntropyLoss()

        history = {
            'epochs': [],
            'losses': [],
            'test_losses': [],
            'batch_losses': [],
            'accuracies': [],
            'test_accuracies': [],
            'precisions': [],
            'test_precisions': [],
            'recalls': [],
            'test_recalls': [],
            'f1s': [],
            'test_f1s': [],
        }

        best_state = None
        best_epoch = 0
        best_test_loss = float('inf')
        epochs_without_improvement = 0
        stopped_epoch = self.max_epoch - 1

        for epoch in range(self.max_epoch):
            batch_train_loss = self._run_train_epoch(train_loader, loss_function, optimizer)
            train_eval = self._evaluate(train_eval_loader, loss_function)
            test_eval = self._evaluate(test_loader, loss_function)
            scheduler.step()

            train_metrics = train_eval['metrics']
            test_metrics = test_eval['metrics']

            history['epochs'].append(epoch)
            history['losses'].append(train_eval['loss'])
            history['test_losses'].append(test_eval['loss'])
            history['batch_losses'].append(batch_train_loss)
            history['accuracies'].append(train_metrics['accuracy'])
            history['test_accuracies'].append(test_metrics['accuracy'])
            history['precisions'].append(train_metrics['precision'])
            history['test_precisions'].append(test_metrics['precision'])
            history['recalls'].append(train_metrics['recall'])
            history['test_recalls'].append(test_metrics['recall'])
            history['f1s'].append(train_metrics['f1'])
            history['test_f1s'].append(test_metrics['f1'])

            improved = test_eval['loss'] < best_test_loss - self.early_stopping_min_delta
            if improved:
                best_epoch = epoch
                best_test_loss = test_eval['loss']
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                'Epoch:', epoch,
                'Training Loss:', train_eval['loss'],
                'Testing Loss:', test_eval['loss'],
                'Batch Training Loss:', batch_train_loss,
                'Training Accuracy:', train_metrics['accuracy'],
                'Testing Accuracy:', test_metrics['accuracy'],
            )

            if epochs_without_improvement >= self.early_stopping_patience:
                stopped_epoch = epoch
                print('Early stopping at epoch:', epoch, 'Best Epoch:', best_epoch)
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        final_eval = self._evaluate(test_loader, loss_function)
        history['best_epoch'] = best_epoch
        history['stopped_epoch'] = stopped_epoch
        history['final_test_loss'] = final_eval['loss']
        history['final_test_metrics'] = final_eval['metrics']
        history['final_pred_y'] = final_eval['pred_y']
        history['final_true_y'] = final_eval['true_y']
        return history

    def _result_dir(self):
        result_dir = Path(__file__).resolve().parents[2] / 'result' / 'stage_4_result'
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    def _save_plots(self, history):
        result_dir = self._result_dir()

        plt.figure(figsize=(8, 5))
        plt.plot(history['epochs'], history['losses'], color='blue', label='training loss (eval mode)')
        plt.plot(history['epochs'], history['test_losses'], color='orange', label='testing loss')
        plt.title('Epoch vs Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(result_dir / 'Stage_4_classification_loss.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(history['epochs'], history['accuracies'], color='blue', label='training accuracy')
        plt.plot(history['epochs'], history['test_accuracies'], color='orange', label='testing accuracy')
        plt.title('Epoch vs Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(result_dir / 'Stage_4_classification_accuracy.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(history['epochs'], history['test_precisions'], label='testing precision')
        plt.plot(history['epochs'], history['test_recalls'], label='testing recall')
        plt.plot(history['epochs'], history['test_f1s'], label='testing f1')
        plt.title('Epoch vs Classification Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(result_dir / 'Stage_4_classification_metrics.png')
        plt.close()

    def _save_report_materials(self, history):
        result_dir = self._result_dir()
        metrics = history['final_test_metrics']
        report_path = result_dir / 'Stage_4_classification_report_materials.txt'

        lines = [
            'Stage 4 Text Classification Report Materials',
            '',
            'Model architecture:',
            '- Embedding layer: vocab_size x ' + str(self.embed_size),
            '- ' + self.rnn_type.upper() + ' layer: hidden_size=' + str(self.hidden_size)
            + ', num_layers=' + str(self.num_layers)
            + ', bidirectional=' + str(self.bidirectional),
            '- Pooling: ' + str(self.pooling),
            '- Dropout: ' + str(self.dropout_rate),
            '- Classifier hidden layer: ' + str(self.fc_hidden_size),
            '',
            'Training settings:',
            '- Optimizer: Adam',
            '- Learning rate: ' + str(self.learning_rate),
            '- Weight decay: ' + str(self.weight_decay),
            '- Epochs: ' + str(self.max_epoch),
            '- Batch size: ' + str(self.batch_size),
            '- Early stopping patience: ' + str(self.early_stopping_patience),
            '- Loss: CrossEntropyLoss',
            '- Plotted training loss is measured in evaluation mode after each epoch',
            '- Best epoch by testing loss: ' + str(history['best_epoch']),
            '- Stopped epoch: ' + str(history['stopped_epoch']),
            '',
            'Final test performance from the selected checkpoint:',
            '- Accuracy: ' + str(metrics['accuracy']),
            '- Precision: ' + str(metrics['precision']),
            '- Recall: ' + str(metrics['recall']),
            '- F1: ' + str(metrics['f1']),
            '- Loss: ' + str(history['final_test_loss']),
            '',
            'Generated plot files:',
            '- result/stage_4_result/Stage_4_classification_loss.png',
            '- result/stage_4_result/Stage_4_classification_accuracy.png',
            '- result/stage_4_result/Stage_4_classification_metrics.png',
        ]

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    def test(self, X, lengths=None):
        self._configure_from_data()
        if lengths is None:
            lengths = self._lengths_from_X(X)

        dummy_y = [0] * len(X)
        loader = self._make_loader(X, dummy_y, lengths, shuffle=False)
        loss_function = nn.CrossEntropyLoss()
        return torch.LongTensor(self._evaluate(loader, loss_function)['pred_y'])

    def run(self):
        print('method running...')
        print('--start training...')
        history = self.fit(
            self.data['train']['X'],
            self.data['train']['y'],
            self.data['test']['X'],
            self.data['test']['y'],
            self.data['train']['lengths'],
            self.data['test']['lengths'],
        )

        self._save_plots(history)
        self._save_report_materials(history)

        model_path = self._result_dir() / 'Stage_4_classification_model.pt'
        torch.save(self.state_dict(), model_path)

        return {
            'pred_y': torch.LongTensor(history['final_pred_y']),
            'true_y': torch.LongTensor(history['final_true_y']),
            'history': history,
            'final_test_metrics': history['final_test_metrics'],
            'final_test_loss': history['final_test_loss'],
            'best_epoch': history['best_epoch'],
            'stopped_epoch': history['stopped_epoch'],
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
                'bidirectional': self.bidirectional,
                'pooling': self.pooling,
                'fc_hidden_size': self.fc_hidden_size,
                'early_stopping_patience': self.early_stopping_patience,
            },
        }
