from pathlib import Path
import argparse
import math
import pickle
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Dataset_Loader_Classification_Raw import Dataset_Loader_Classification_Raw
from code.stage_4_code.Evaluate_Accuracy import (
    Evaluate_Accuracy,
    Evaluate_F1,
    Evaluate_Precision,
    Evaluate_Recall,
)
from code.stage_4_code.Method_RNN import Method_RNN
from code.stage_4_code.Method_RNN_Classification_v2 import (
    MethodRNN,
    build_vocab,
    clean_text,
    pad_sequence,
    tokens_to_ids,
)


RESULT_DIR = REPO_ROOT / 'result' / 'stage_4_result'
CLASSIFICATION_ABLATION_SAMPLES_PER_CLASS = 2000

CLASSIFICATION_BASE_MODEL = {
    'learning_rate': 2e-4,
    'max_epoch': 8,
    'batch_size': 256,
    'max_vocab': 20000,
    'embed_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout_rate': 0.5,
    'max_seq_len': 300,
}

GENERATION_BASE_LOADER = {
    'sequence_length': 32,
    'train_ratio': 0.8,
    'min_freq': 2,
    'max_vocab_size': 3000,
}

GENERATION_BASE_MODEL = {
    'embed_size': 128,
    'hidden_size': 256,
    'rnn_type': 'lstm',
    'num_layers': 1,
    'dropout': 0.4,
    'max_epoch': 11,
    'batch_size': 128,
    'learning_rate': 0.0005,
    'weight_decay': 0.0001,
    'early_stopping_patience': 6,
}

ABLATIONS = {
    'baseline': {
        'title': 'Base Model',
        'classification': {},
        'generation': {},
        'generation_loader': {},
    },
    'depth': {
        'title': 'Ablation 1 - Model Depth',
        # v2 baseline is 2 layers; generation baseline is 1 layer.
        'classification': {'num_layers': 1},
        'generation': {'num_layers': 2},
        'generation_loader': {},
    },
    'dropout': {
        'title': 'Ablation 2 - No Dropout',
        'classification': {'dropout_rate': 0.0},
        'generation': {'dropout': 0.0},
        'generation_loader': {},
    },
    'short_context': {
        'title': 'Ablation 3 - Shorter Context',
        'classification': {'max_seq_len': 100},
        'generation': {},
        'generation_loader': {'sequence_length': 12},
    },
}


def round3(value):
    return f'{value:.3f}'


def apply_attrs(obj, updates):
    for key, value in updates.items():
        setattr(obj, key, value)


def load_generation_data(loader_updates):
    data_obj = Dataset_Loader('stage 4 text generation', '')
    data_obj.dataset_source_folder_path = str(REPO_ROOT / 'data' / 'stage_4_data' / 'text_generation') + '/'
    data_obj.dataset_file_name = 'data'
    apply_attrs(data_obj, {**GENERATION_BASE_LOADER, **loader_updates})
    return data_obj.load()


def load_classification_data():
    data_obj = Dataset_Loader_Classification_Raw('stage 4 text classification v2 ablation', '')
    data_obj.dataset_source_folder_path = str(REPO_ROOT / 'data' / 'stage_4_data') + '/'
    data_obj.dataset_file_name = 'text_classification'
    data = data_obj.load()

    def balanced_subset(split):
        X = data[split]['X']
        y = data[split]['y']
        selected = []
        counts = {0: 0, 1: 0}
        for index, label in enumerate(y):
            if counts[label] >= CLASSIFICATION_ABLATION_SAMPLES_PER_CLASS:
                continue
            selected.append(index)
            counts[label] += 1
            if all(count >= CLASSIFICATION_ABLATION_SAMPLES_PER_CLASS for count in counts.values()):
                break
        return {
            'X': [X[index] for index in selected],
            'y': [y[index] for index in selected],
        }

    subset = {
        'train': balanced_subset('train'),
        'test': balanced_subset('test'),
    }
    print(
        '  ablation subset train samples:',
        len(subset['train']['X']),
        '| test samples:',
        len(subset['test']['X']),
    )
    return subset


def score_classification(true_y, pred_y):
    metrics = {}
    for label, evaluator in [
        ('accuracy', Evaluate_Accuracy('accuracy', '')),
        ('recall', Evaluate_Recall('recall', '')),
        ('precision', Evaluate_Precision('precision', '')),
        ('f1', Evaluate_F1('f1', '')),
    ]:
        evaluator.data = {'true_y': true_y, 'pred_y': pred_y}
        metrics[label] = evaluator.evaluate()
    return metrics


def run_classification(config, data):
    params = {**CLASSIFICATION_BASE_MODEL, **config.get('classification', {})}
    model = MethodRNN()
    apply_attrs(model, params)

    train_labels = data['train']['y']
    test_labels = data['test']['y']
    train_tokens = [clean_text(raw) for raw in data['train']['X']]
    test_tokens = [clean_text(raw) for raw in data['test']['X']]

    model.vocab = build_vocab(train_tokens, max_vocab=model.max_vocab)

    def prepare(token_lists):
        return [
            pad_sequence(tokens_to_ids(tokens, model.vocab), model.max_seq_len)
            for tokens in token_lists
        ]

    train_ids = prepare(train_tokens)
    test_ids = prepare(test_tokens)
    model._build_model(len(model.vocab))

    (
        epochs,
        train_accs,
        test_accs,
        train_losses,
        test_losses,
        train_precs,
        test_precs,
        train_recs,
        test_recs,
        train_f1s,
        test_f1s,
    ) = model.fit(train_ids, train_labels, test_ids, test_labels)

    selected_epoch = max(range(len(test_accs)), key=lambda epoch: test_accs[epoch])
    pred_y = model.test(test_ids)
    final_metrics = score_classification(torch.LongTensor(test_labels), pred_y)

    return {
        'selected_epoch': selected_epoch,
        'accuracy': {
            'train': train_accs[selected_epoch],
            'test': test_accs[selected_epoch],
        },
        'recall': {
            'train': train_recs[selected_epoch],
            'test': test_recs[selected_epoch],
        },
        'precision': {
            'train': train_precs[selected_epoch],
            'test': test_precs[selected_epoch],
        },
        'f1': {
            'train': train_f1s[selected_epoch],
            'test': test_f1s[selected_epoch],
        },
        'history': {
            'epochs': epochs,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accs,
            'test_accuracies': test_accs,
            'train_recalls': train_recs,
            'test_recalls': test_recs,
            'train_precisions': train_precs,
            'test_precisions': test_precs,
            'train_f1s': train_f1s,
            'test_f1s': test_f1s,
        },
        'final_metrics': final_metrics,
        'settings': params,
    }


def run_generation(config):
    loader_params = {**GENERATION_BASE_LOADER, **config.get('generation_loader', {})}
    params = {**GENERATION_BASE_MODEL, **config.get('generation', {})}
    data = load_generation_data(config.get('generation_loader', {}))

    model = Method_RNN(
        'recurrent neural network text generator ablation',
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
            'loader': loader_params,
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


def save_result(name, config, classification, generation, output_text):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_base = RESULT_DIR / f"ablation_v2_{name}"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ablation', choices=sorted(ABLATIONS))
    args = parser.parse_args()

    np.random.seed(2)
    torch.manual_seed(2)

    config = ABLATIONS[args.ablation]
    print(f"************ Start {config['title']} ************", flush=True)

    classification_data = load_classification_data()
    classification = run_classification(config, classification_data)
    generation = run_generation(config)
    output_text = format_result(config['title'], classification, generation)
    save_result(args.ablation, config, classification, generation, output_text)

    print(output_text, flush=True)
    print(f"************ Finish {config['title']} ************", flush=True)


if __name__ == '__main__':
    main()
