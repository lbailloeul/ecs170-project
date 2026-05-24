from pathlib import Path
import argparse
import json
import math
import pickle
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from code.stage_4_code.Method_RNN import Method_RNN
from code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification


ABLATIONS = {
    'unit_rnn': {
        'title': 'Ablation 1 - Vanilla RNN Units',
        'classification': {'rnn_type': 'rnn'},
        'generation': {'rnn_type': 'rnn'},
    },
    'no_dropout': {
        'title': 'Ablation 2 - No Dropout',
        'classification': {'dropout': 0.0},
        'generation': {'dropout': 0.0},
    },
    'short_context': {
        'title': 'Ablation 3 - Shorter Context',
        'classification_loader': {'max_len': 100},
        'generation_loader': {'sequence_length': 6},
    },
}


def round3(value):
    return f'{value:.3f}'


def load_classification_data(loader_updates):
    data_obj = Dataset_Loader_Classification('stage 4 text classification', '')
    data_obj.dataset_source_folder_path = str(REPO_ROOT / 'data' / 'stage_4_data' / 'text_classification') + '/'
    data_obj.max_len = 250
    data_obj.min_freq = 2
    data_obj.max_vocab_size = 20000

    for key, value in loader_updates.items():
        setattr(data_obj, key, value)

    return data_obj.load()


def load_generation_data(loader_updates):
    data_obj = Dataset_Loader('stage 4 text generation', '')
    data_obj.dataset_source_folder_path = str(REPO_ROOT / 'data' / 'stage_4_data' / 'text_generation') + '/'
    data_obj.dataset_file_name = 'data'
    data_obj.sequence_length = 12
    data_obj.train_ratio = 0.8
    data_obj.min_freq = 2
    data_obj.max_vocab_size = 3000

    for key, value in loader_updates.items():
        setattr(data_obj, key, value)

    return data_obj.load()


def run_classification(config):
    data = load_classification_data(config.get('classification_loader', {}))
    params = {
        'embed_size': 128,
        'hidden_size': 128,
        'rnn_type': 'gru',
        'num_layers': 1,
        'dropout': 0.25,
        'max_epoch': 8,
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 0.00001,
        'bidirectional': True,
        'pooling': 'mean_max',
        'fc_hidden_size': 128,
        'early_stopping_patience': 10,
    }
    params.update(config.get('classification', {}))

    model = Method_RNN_Classification(
        'recurrent neural network text classifier',
        '',
        **params,
    )
    model.data = data

    history = model.fit(
        data['train']['X'],
        data['train']['y'],
        data['test']['X'],
        data['test']['y'],
        data['train']['lengths'],
        data['test']['lengths'],
    )

    best_epoch = max(
        range(len(history['test_accuracies'])),
        key=lambda epoch: history['test_accuracies'][epoch],
    )

    return {
        'history': history,
        'selected_epoch': best_epoch,
        'accuracy': {
            'train': history['accuracies'][best_epoch],
            'test': history['test_accuracies'][best_epoch],
        },
        'recall': {
            'train': history['recalls'][best_epoch],
            'test': history['test_recalls'][best_epoch],
        },
        'precision': {
            'train': history['precisions'][best_epoch],
            'test': history['test_precisions'][best_epoch],
        },
        'f1': {
            'train': history['f1s'][best_epoch],
            'test': history['test_f1s'][best_epoch],
        },
        'settings': {
            'loader': config.get('classification_loader', {}),
            'model': params,
        },
    }


def run_generation(config):
    data = load_generation_data(config.get('generation_loader', {}))
    params = {
        'embed_size': 128,
        'hidden_size': 128,
        'rnn_type': 'lstm',
        'dropout': 0.2,
        'max_epoch': 20,
        'batch_size': 64,
        'learning_rate': 0.0015,
        'weight_decay': 0.00001,
        'early_stopping_patience': 25,
    }
    params.update(config.get('generation', {}))

    model = Method_RNN(
        'recurrent neural network text generator',
        '',
        **params,
    )
    model.data = data

    epochs, losses, test_losses, best_epoch, best_test_loss, stopped_epoch = model.fit(
        data['train']['X'],
        data['train']['y'],
        data['test']['X'],
        data['test']['y'],
    )
    final_epoch = epochs[-1]

    return {
        'epochs': epochs,
        'losses': losses,
        'test_losses': test_losses,
        'best_epoch': best_epoch,
        'best_test_loss': best_test_loss,
        'perplexity': math.exp(min(best_test_loss, 50)),
        'final_epoch': final_epoch,
        'final_train_loss': losses[-1],
        'final_test_loss': test_losses[-1],
        'stopped_epoch': stopped_epoch,
        'settings': {
            'loader': config.get('generation_loader', {}),
            'model': params,
        },
    }


def format_result(title, classification, generation):
    epoch = classification['selected_epoch']
    lines = [
        title,
        f"Epoch: {epoch} Training Accuracy: {round3(classification['accuracy']['train'])} Testing Accuracy: {round3(classification['accuracy']['test'])}",
        f"Epoch: {epoch} Training Recall: {round3(classification['recall']['train'])} Testing Recall: {round3(classification['recall']['test'])}",
        f"Epoch: {epoch} Training Precision: {round3(classification['precision']['train'])} Testing Precision: {round3(classification['precision']['test'])}",
        f"Epoch: {epoch} Training F1 : {round3(classification['f1']['train'])} Testing F1 : {round3(classification['f1']['test'])}",
        '',
        f"Text generation dataset: achieved lowest testing loss on epoch {generation['best_epoch']}.",
        f"Epoch: {generation['best_epoch']} Training Loss: {round3(generation['losses'][generation['best_epoch']])} Testing Loss: {round3(generation['test_losses'][generation['best_epoch']])}",
        f"Epoch: {generation['best_epoch']} Testing Perplexity: {round3(generation['perplexity'])}",
        f"Epoch: {generation['final_epoch']} Final Training Loss: {round3(generation['final_train_loss'])} Final Testing Loss: {round3(generation['final_test_loss'])}",
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ablation', choices=sorted(ABLATIONS))
    args = parser.parse_args()

    np.random.seed(2)
    torch.manual_seed(2)

    config = ABLATIONS[args.ablation]
    print(f"************ Start {config['title']} ************", flush=True)
    classification = run_classification(config)
    generation = run_generation(config)
    output_text = format_result(config['title'], classification, generation)

    result_dir = REPO_ROOT / 'result' / 'stage_4_result'
    result_dir.mkdir(parents=True, exist_ok=True)
    result_base = result_dir / f"ablation_{args.ablation}"

    with open(result_base.with_suffix('.txt'), 'w', encoding='utf-8') as f:
        f.write(output_text + '\n')

    with open(result_base.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(
            {
                'title': config['title'],
                'classification': classification,
                'generation': generation,
                'output_text': output_text,
            },
            f,
        )

    print(output_text, flush=True)
    print(f"************ Finish {config['title']} ************", flush=True)


if __name__ == '__main__':
    main()
