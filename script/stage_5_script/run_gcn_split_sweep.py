from pathlib import Path
import argparse
import pickle
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.stage_5_script.script_gcn import TABLE_2_TARGETS, run_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Run stage 5 GCN across random train/val/test splits.')
    parser.add_argument('--dataset', choices=['all', 'cora', 'citeseer', 'pubmed'], default='all')
    parser.add_argument('--start-split-seed', type=int, default=0)
    parser.add_argument('--num-splits', type=int, default=10)
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
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_names = ['cora', 'citeseer', 'pubmed'] if args.dataset == 'all' else [args.dataset]
    split_seeds = list(range(args.start_split_seed, args.start_split_seed + args.num_splits))
    sweep_summary = {}

    for dataset_name in dataset_names:
        accuracies = []
        runs = []
        for split_seed in split_seeds:
            args.split_seed = split_seed
            print('========== Sweep run:', dataset_name, 'split seed', split_seed, '==========')
            result = run_dataset(dataset_name, args)
            accuracy = result['final_test_metrics']['accuracy']
            accuracies.append(accuracy)
            runs.append({
                'split_seed': split_seed,
                'accuracy': accuracy,
                'best_epoch': result['best_epoch'],
                'stopped_epoch': result['stopped_epoch'],
                'split_summary': result['split_summary'],
            })

        mean_accuracy = statistics.mean(accuracies)
        std_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        best_run = max(runs, key=lambda item: item['accuracy'])
        sweep_summary[dataset_name] = {
            'target': TABLE_2_TARGETS[dataset_name],
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'best_run': best_run,
            'runs': runs,
        }

    result_dir = REPO_ROOT / 'result' / 'stage_5_result'
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / 'GCN_split_sweep_summary.pkl'
    with open(summary_path, 'wb') as f:
        pickle.dump(sweep_summary, f)

    text_path = result_dir / 'GCN_split_sweep_summary.txt'
    lines = ['Stage 5 GCN Split Sweep Summary', '']
    for dataset_name, summary in sweep_summary.items():
        lines.extend([
            dataset_name,
            'target: ' + str(summary['target']),
            'mean_accuracy: ' + str(summary['mean_accuracy']),
            'std_accuracy: ' + str(summary['std_accuracy']),
            'best_split_seed: ' + str(summary['best_run']['split_seed']),
            'best_accuracy: ' + str(summary['best_run']['accuracy']),
            '',
        ])
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('************ Sweep Summary ************')
    for dataset_name, summary in sweep_summary.items():
        print(
            dataset_name,
            'mean:', summary['mean_accuracy'],
            'std:', summary['std_accuracy'],
            'best:', summary['best_run']['accuracy'],
            'target:', summary['target'],
        )
    print('Saved:', text_path)


if __name__ == '__main__':
    main()
