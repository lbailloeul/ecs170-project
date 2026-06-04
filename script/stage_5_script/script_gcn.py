from pathlib import Path
import argparse
import random
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Method_GCN import Method_GCN
from code.stage_5_code.Result_Saver import Result_Saver


TABLE_2_TARGETS = {
    'citeseer': 0.703,
    'cora': 0.815,
    'pubmed': 0.790,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Run stage 5 GCN node classification.')
    parser.add_argument('--dataset', choices=['all', 'cora', 'citeseer', 'pubmed'], default='all')
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--model-seed', type=int, default=0)
    parser.add_argument('--labels-per-class', type=int, default=20)
    parser.add_argument('--val-size', type=int, default=500)
    parser.add_argument('--test-size', type=int, default=1000)
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--hidden-layers', type=int, choices=[1, 2], default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def run_dataset(dataset_name, args):
    set_seed(args.model_seed)

    data_obj = Dataset_Loader(
        seed=args.split_seed,
        dName=dataset_name,
        dDescription='stage 5 citation network',
        labels_per_class=args.labels_per_class,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    data_obj.dataset_source_folder_path = str(REPO_ROOT / 'data' / 'stage_5_data' / dataset_name)
    loaded_data = data_obj.load()
    loaded_data['dataset_name'] = dataset_name

    set_seed(args.model_seed)
    method_obj = Method_GCN(
        'graph convolutional network',
        '',
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epoch=args.max_epoch,
        early_stopping_patience=args.patience,
        print_every=args.print_every,
        verbose=not args.quiet,
    )
    method_obj.data = loaded_data

    print('************ Start', dataset_name, '************')
    learned_result = method_obj.run()

    result_dir = REPO_ROOT / 'result' / 'stage_5_result'
    result_dir.mkdir(parents=True, exist_ok=True)
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir / ('GCN_' + dataset_name + '_'))
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = 'split' + str(args.split_seed) + '_seed' + str(args.model_seed)
    result_obj.data = learned_result
    result_obj.save()

    metrics = learned_result['final_test_metrics']
    target = TABLE_2_TARGETS[dataset_name]
    print('************ Final Test Performance', dataset_name, '************')
    print('Accuracy:', metrics['accuracy'])
    print('Precision:', metrics['precision'])
    print('Recall:', metrics['recall'])
    print('F1:', metrics['f1'])
    print('Loss:', learned_result['final_test_loss'])
    print('Best Epoch:', learned_result['best_epoch'])
    print('Stopped Epoch:', learned_result['stopped_epoch'])
    print('Table 2 Target:', target)
    print('Gap:', metrics['accuracy'] - target)
    print('************ Finish', dataset_name, '************')
    return learned_result


def main():
    args = parse_args()
    dataset_names = ['cora', 'citeseer', 'pubmed'] if args.dataset == 'all' else [args.dataset]
    results = {}
    for dataset_name in dataset_names:
        results[dataset_name] = run_dataset(dataset_name, args)

    print('************ Stage 5 Summary ************')
    for dataset_name, result in results.items():
        accuracy = result['final_test_metrics']['accuracy']
        print(dataset_name + ':', accuracy, 'target:', TABLE_2_TARGETS[dataset_name])


if __name__ == '__main__':
    main()
