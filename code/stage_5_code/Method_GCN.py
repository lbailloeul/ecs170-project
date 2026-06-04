from pathlib import Path
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from code.base_class.method import method

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / 'ecs170_matplotlib'
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_MPLCONFIGDIR))
from matplotlib import pyplot as plt


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        support = self.linear(x)
        return torch.sparse.mm(adj, support)


class Method_GCN(method, nn.Module):
    def __init__(
        self,
        mName='graph convolutional network',
        mDescription='',
        input_size=None,
        output_size=None,
        hidden_size=16,
        hidden_layers=1,
        dropout=0.5,
        learning_rate=0.01,
        weight_decay=5e-4,
        max_epoch=200,
        early_stopping_patience=10,
        early_stopping_min_delta=0.0,
        print_every=10,
        verbose=True,
        device=None,
    ):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if hidden_layers not in (1, 2):
            raise ValueError('hidden_layers must be 1 or 2 for this stage 5 GCN.')

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.print_every = print_every
        self.verbose = verbose

        self.layers = None
        self.dataset_name = 'dataset'

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if self.input_size is not None and self.output_size is not None:
            self._build_model(self.input_size, self.output_size)

    def _build_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        layers = [GraphConvolution(input_size, self.hidden_size)]
        if self.hidden_layers == 2:
            layers.append(GraphConvolution(self.hidden_size, self.hidden_size))
        layers.append(GraphConvolution(self.hidden_size, output_size))

        self.layers = nn.ModuleList(layers)
        self.to(self.device)
        print('using device:', self.device)

    def _configure_from_data(self):
        if self.data is None:
            raise ValueError('Method_GCN.data must be set before training.')

        graph = self.data['graph']
        input_size = graph['X'].shape[1]
        output_size = graph.get('num_classes', int(graph['y'].max().item()) + 1)
        self.dataset_name = self.data.get('dataset_name', self.dataset_name)

        if self.layers is None or self.input_size != input_size or self.output_size != output_size:
            self._build_model(input_size, output_size)

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = layer(x, adj)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return self.layers[-1](x, adj)

    @staticmethod
    def _calculate_metrics(true_y, pred_y):
        return {
            'accuracy': accuracy_score(true_y, pred_y),
            'precision': precision_score(true_y, pred_y, average='macro', zero_division=0),
            'recall': recall_score(true_y, pred_y, average='macro', zero_division=0),
            'f1': f1_score(true_y, pred_y, average='macro', zero_division=0),
        }

    def _first_layer_l2(self):
        if self.weight_decay <= 0.0 or self.layers is None:
            return 0.0
        return 0.5 * self.weight_decay * torch.sum(self.layers[0].linear.weight.pow(2))

    def _evaluate(self, logits, labels, idx, loss_function):
        loss = loss_function(logits[idx], labels[idx])
        pred_y = logits[idx].argmax(dim=1)
        true_y = labels[idx]
        metrics = self._calculate_metrics(
            true_y.detach().cpu().numpy(),
            pred_y.detach().cpu().numpy(),
        )
        return {
            'loss': loss.item(),
            'metrics': metrics,
            'true_y': true_y.detach().cpu().tolist(),
            'pred_y': pred_y.detach().cpu().tolist(),
        }

    def fit(self, graph, split):
        self._configure_from_data()

        features = graph['X'].to(self.device)
        labels = graph['y'].to(self.device)
        adj = graph['utility']['A'].coalesce().to(self.device)
        idx_train = split['idx_train'].to(self.device)
        idx_val = split['idx_val'].to(self.device)
        idx_test = split['idx_test'].to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        history = {
            'epochs': [],
            'losses': [],
            'val_losses': [],
            'test_losses': [],
            'accuracies': [],
            'val_accuracies': [],
            'test_accuracies': [],
            'precisions': [],
            'val_precisions': [],
            'test_precisions': [],
            'recalls': [],
            'val_recalls': [],
            'test_recalls': [],
            'f1s': [],
            'val_f1s': [],
            'test_f1s': [],
        }

        best_state = None
        best_epoch = 0
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        stopped_epoch = self.max_epoch - 1

        for epoch in range(self.max_epoch):
            self.train()
            optimizer.zero_grad()
            logits = self.forward(features, adj)
            train_loss = loss_function(logits[idx_train], labels[idx_train]) + self._first_layer_l2()
            train_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                eval_logits = self.forward(features, adj)
                train_eval = self._evaluate(eval_logits, labels, idx_train, loss_function)
                val_eval = self._evaluate(eval_logits, labels, idx_val, loss_function)
                test_eval = self._evaluate(eval_logits, labels, idx_test, loss_function)

            train_metrics = train_eval['metrics']
            val_metrics = val_eval['metrics']
            test_metrics = test_eval['metrics']

            history['epochs'].append(epoch)
            history['losses'].append(train_eval['loss'])
            history['val_losses'].append(val_eval['loss'])
            history['test_losses'].append(test_eval['loss'])
            history['accuracies'].append(train_metrics['accuracy'])
            history['val_accuracies'].append(val_metrics['accuracy'])
            history['test_accuracies'].append(test_metrics['accuracy'])
            history['precisions'].append(train_metrics['precision'])
            history['val_precisions'].append(val_metrics['precision'])
            history['test_precisions'].append(test_metrics['precision'])
            history['recalls'].append(train_metrics['recall'])
            history['val_recalls'].append(val_metrics['recall'])
            history['test_recalls'].append(test_metrics['recall'])
            history['f1s'].append(train_metrics['f1'])
            history['val_f1s'].append(val_metrics['f1'])
            history['test_f1s'].append(test_metrics['f1'])

            improved = val_eval['loss'] < best_val_loss - self.early_stopping_min_delta
            if improved:
                best_epoch = epoch
                best_val_loss = val_eval['loss']
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose and (epoch % self.print_every == 0 or epoch == self.max_epoch - 1):
                print(
                    'Epoch:', epoch,
                    'Train Loss:', round(train_eval['loss'], 4),
                    'Val Loss:', round(val_eval['loss'], 4),
                    'Test Loss:', round(test_eval['loss'], 4),
                    'Train Accuracy:', round(train_metrics['accuracy'], 4),
                    'Val Accuracy:', round(val_metrics['accuracy'], 4),
                    'Test Accuracy:', round(test_metrics['accuracy'], 4),
                )

            if epochs_without_improvement >= self.early_stopping_patience:
                stopped_epoch = epoch
                if self.verbose:
                    print('Early stopping at epoch:', epoch, 'Best Epoch:', best_epoch)
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()
        with torch.no_grad():
            final_logits = self.forward(features, adj)
            final_train_eval = self._evaluate(final_logits, labels, idx_train, loss_function)
            final_val_eval = self._evaluate(final_logits, labels, idx_val, loss_function)
            final_test_eval = self._evaluate(final_logits, labels, idx_test, loss_function)

        history['best_epoch'] = best_epoch
        history['stopped_epoch'] = stopped_epoch
        history['best_val_loss'] = best_val_loss
        history['final_train_metrics'] = final_train_eval['metrics']
        history['final_val_metrics'] = final_val_eval['metrics']
        history['final_test_metrics'] = final_test_eval['metrics']
        history['final_train_loss'] = final_train_eval['loss']
        history['final_val_loss'] = final_val_eval['loss']
        history['final_test_loss'] = final_test_eval['loss']
        history['final_pred_y'] = final_test_eval['pred_y']
        history['final_true_y'] = final_test_eval['true_y']
        return history

    def _result_dir(self):
        result_dir = Path(__file__).resolve().parents[2] / 'result' / 'stage_5_result'
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    def _artifact_prefix(self, graph):
        split_seed = graph.get('split_summary', {}).get('seed')
        if split_seed is None:
            return 'Stage_5_' + self.dataset_name
        return 'Stage_5_' + self.dataset_name + '_split' + str(split_seed)

    def _save_plots(self, history, graph):
        result_dir = self._result_dir()
        prefix = self._artifact_prefix(graph)

        plt.figure(figsize=(8, 5))
        plt.plot(history['epochs'], history['losses'], label='training loss')
        plt.plot(history['epochs'], history['val_losses'], label='validation loss')
        plt.plot(history['epochs'], history['test_losses'], label='testing loss')
        plt.title('Epoch vs Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(result_dir / (prefix + '_loss.png'))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(history['epochs'], history['accuracies'], label='training accuracy')
        plt.plot(history['epochs'], history['val_accuracies'], label='validation accuracy')
        plt.plot(history['epochs'], history['test_accuracies'], label='testing accuracy')
        plt.title('Epoch vs Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(result_dir / (prefix + '_accuracy.png'))
        plt.close()

    def _save_report_materials(self, history, graph):
        result_dir = self._result_dir()
        metrics = history['final_test_metrics']
        report_path = result_dir / (self._artifact_prefix(graph) + '_report_materials.txt')

        lines = [
            'Stage 5 GCN Report Materials',
            '',
            'Dataset: ' + self.dataset_name,
            'Split summary: ' + str(graph.get('split_summary', {})),
            '',
            'Model architecture:',
            '- Hidden GCN layers: ' + str(self.hidden_layers),
            '- Hidden units: ' + str(self.hidden_size),
            '- Output classes: ' + str(self.output_size),
            '- Dropout: ' + str(self.dropout_rate),
            '',
            'Training settings:',
            '- Optimizer: Adam',
            '- Learning rate: ' + str(self.learning_rate),
            '- L2 regularization on first GCN layer: ' + str(self.weight_decay),
            '- Maximum epochs: ' + str(self.max_epoch),
            '- Early stopping patience: ' + str(self.early_stopping_patience),
            '- Best epoch by validation loss: ' + str(history['best_epoch']),
            '- Stopped epoch: ' + str(history['stopped_epoch']),
            '',
            'Final selected-checkpoint performance:',
            '- Train accuracy: ' + str(history['final_train_metrics']['accuracy']),
            '- Validation accuracy: ' + str(history['final_val_metrics']['accuracy']),
            '- Test accuracy: ' + str(metrics['accuracy']),
            '- Test precision: ' + str(metrics['precision']),
            '- Test recall: ' + str(metrics['recall']),
            '- Test F1: ' + str(metrics['f1']),
            '- Test loss: ' + str(history['final_test_loss']),
            '',
            'Table 2 reference accuracies from Kipf and Welling GCN:',
            '- Citeseer: 70.3',
            '- Cora: 81.5',
            '- Pubmed: 79.0',
            '- Random splits: 67.9 +/- 0.5, 80.1 +/- 0.5, 78.9 +/- 0.7',
        ]

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    def test(self):
        self._configure_from_data()
        graph = self.data['graph']
        split = self.data['train_test_val']

        features = graph['X'].to(self.device)
        adj = graph['utility']['A'].coalesce().to(self.device)
        idx_test = split['idx_test'].to(self.device)

        self.eval()
        with torch.no_grad():
            logits = self.forward(features, adj)
            pred_y = logits[idx_test].argmax(dim=1)
        return pred_y.detach().cpu()

    def run(self):
        print('method running...')
        print('--start training...')
        graph = self.data['graph']
        split = self.data['train_test_val']
        history = self.fit(graph, split)

        self._save_plots(history, graph)
        self._save_report_materials(history, graph)

        model_path = self._result_dir() / (self._artifact_prefix(graph) + '_model.pt')
        torch.save(self.state_dict(), model_path)

        return {
            'pred_y': torch.LongTensor(history['final_pred_y']),
            'true_y': torch.LongTensor(history['final_true_y']),
            'history': history,
            'final_train_metrics': history['final_train_metrics'],
            'final_val_metrics': history['final_val_metrics'],
            'final_test_metrics': history['final_test_metrics'],
            'final_test_loss': history['final_test_loss'],
            'best_epoch': history['best_epoch'],
            'stopped_epoch': history['stopped_epoch'],
            'split_summary': graph.get('split_summary', {}),
            'model_settings': {
                'hidden_size': self.hidden_size,
                'hidden_layers': self.hidden_layers,
                'dropout': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'max_epoch': self.max_epoch,
                'early_stopping_patience': self.early_stopping_patience,
            },
        }
