from pathlib import Path
import pickle
import random
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Method_GCN import Method_GCN


DATASETS = ['cora', 'citeseer', 'pubmed']

BASELINE_ACCURACY = {
    'cora': 0.809,
    'citeseer': 0.693,
    'pubmed': 0.798,
}

ABLATIONS = [
    {
        'name': 'model_depth',
        'title': 'Ablation 1 - Model Depth',
        'description': 'Add one extra GCN hidden layer to test depth/oversmoothing risk.',
        'settings': {'hidden_layers': 2, 'hidden_size': 16, 'dropout': 0.5, 'weight_decay': 5e-4},
    },
    {
        'name': 'hidden_dimension',
        'title': 'Ablation 2 - Hidden Dimension',
        'description': 'Increase hidden dimension from 16 to 32.',
        'settings': {'hidden_layers': 1, 'hidden_size': 32, 'dropout': 0.5, 'weight_decay': 5e-4},
    },
    {
        'name': 'no_dropout',
        'title': 'Ablation 3 - Dropout',
        'description': 'Remove dropout regularization to test overfitting on the small labeled set.',
        'settings': {'hidden_layers': 1, 'hidden_size': 16, 'dropout': 0.0, 'weight_decay': 5e-4},
    },
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_name, split_seed):
    data_obj = Dataset_Loader(
        seed=split_seed,
        dName=dataset_name,
        dDescription='stage 5 citation network',
        labels_per_class=20,
        val_size=500,
        test_size=1000,
    )
    data_obj.dataset_source_folder_path = str(REPO_ROOT / 'data' / 'stage_5_data' / dataset_name)
    loaded_data = data_obj.load()
    loaded_data['dataset_name'] = dataset_name
    return loaded_data


def run_ablation(dataset_name, ablation, split_seed=0, model_seed=0):
    loaded_data = load_dataset(dataset_name, split_seed)
    settings = ablation['settings']

    set_seed(model_seed)
    method_obj = Method_GCN(
        'graph convolutional network',
        '',
        hidden_size=settings['hidden_size'],
        hidden_layers=settings['hidden_layers'],
        dropout=settings['dropout'],
        learning_rate=0.01,
        weight_decay=settings['weight_decay'],
        max_epoch=200,
        early_stopping_patience=10,
        print_every=50,
        verbose=False,
    )
    method_obj.data = loaded_data

    history = method_obj.fit(loaded_data['graph'], loaded_data['train_test_val'])
    metrics = history['final_test_metrics']
    train_metrics = history['final_train_metrics']
    return {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'train_accuracy': train_metrics['accuracy'],
        'train_precision': train_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'train_f1': train_metrics['f1'],
        'test_loss': history['final_test_loss'],
        'train_loss': history['final_train_loss'],
        'best_epoch': history['best_epoch'],
        'stopped_epoch': history['stopped_epoch'],
        'settings': settings,
    }


def write_summary(results):
    result_dir = REPO_ROOT / 'result' / 'stage_5_result'
    result_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = result_dir / 'GCN_ablation_summary.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)

    text_path = result_dir / 'GCN_ablation_summary.txt'
    lines = [
        'Stage 5 GCN Ablation Study',
        'Split seed: 0',
        'Model seed: 0',
        '',
        'Baseline: 1 hidden GCN layer, hidden_size=16, dropout=0.5, first-layer L2=5e-4',
        '',
        'Ablation\tCora Acc\tCiteseer Acc\tPubmed Acc',
    ]

    for ablation in ABLATIONS:
        name = ablation['name']
        lines.append(
            name + '\t'
            + str(results[name]['cora']['accuracy']) + '\t'
            + str(results[name]['citeseer']['accuracy']) + '\t'
            + str(results[name]['pubmed']['accuracy'])
        )

    lines.extend(['', 'Detailed Results'])
    for ablation in ABLATIONS:
        name = ablation['name']
        lines.extend(['', ablation['title'], ablation['description'], 'settings: ' + str(ablation['settings'])])
        for dataset_name in DATASETS:
            result = results[name][dataset_name]
            lines.append(
                dataset_name
                + ': train_loss=' + str(result['train_loss'])
                + ', test_loss=' + str(result['test_loss'])
                + ': accuracy=' + str(result['accuracy'])
                + ', precision=' + str(result['precision'])
                + ', recall=' + str(result['recall'])
                + ', f1=' + str(result['f1'])
                + ', best_epoch=' + str(result['best_epoch'])
            )

    lines.extend(['', 'Report Format'])
    for ablation in ABLATIONS:
        name = ablation['name']
        lines.extend(['', ablation['title']])
        for dataset_name in DATASETS:
            result = results[name][dataset_name]
            lines.extend([
                dataset_name.capitalize() + ' node classification dataset: achieved lowest validation loss on epoch '
                + str(result['best_epoch']) + '.',
                'Epoch: ' + str(result['best_epoch'])
                + ' Training Accuracy: ' + format(result['train_accuracy'], '.3f')
                + ' Testing Accuracy: ' + format(result['accuracy'], '.3f'),
                'Epoch: ' + str(result['best_epoch'])
                + ' Training Recall: ' + format(result['train_recall'], '.3f')
                + ' Testing Recall: ' + format(result['recall'], '.3f'),
                'Epoch: ' + str(result['best_epoch'])
                + ' Training Precision: ' + format(result['train_precision'], '.3f')
                + ' Testing Precision: ' + format(result['precision'], '.3f'),
                'Epoch: ' + str(result['best_epoch'])
                + ' Training F1: ' + format(result['train_f1'], '.3f')
                + ' Testing F1: ' + format(result['f1'], '.3f'),
            ])

    with open(text_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('Saved:', text_path)
    print('Saved:', pkl_path)


def main():
    split_seed = 0
    model_seed = 0
    set_seed(model_seed)
    results = {}

    for ablation in ABLATIONS:
        name = ablation['name']
        results[name] = {}
        print('========== Ablation:', name, '==========')
        for dataset_name in DATASETS:
            print('Running', dataset_name)
            result = run_ablation(dataset_name, ablation, split_seed=split_seed, model_seed=model_seed)
            results[name][dataset_name] = result
            print(dataset_name, 'accuracy:', result['accuracy'])

    write_summary(results)

    print('************ Ablation Accuracy Summary ************')
    print('baseline', BASELINE_ACCURACY)
    for ablation in ABLATIONS:
        name = ablation['name']
        print(name, {dataset_name: results[name][dataset_name]['accuracy'] for dataset_name in DATASETS})


if __name__ == '__main__':
    main()
