from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN import Method_RNN
from code.stage_4_code.Result_Saver import Result_Saver


if 1:
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('stage 4 text generation', '')
    data_obj.dataset_source_folder_path = str(REPO_ROOT / 'data' / 'stage_4_data' / 'text_generation') + '/'
    data_obj.dataset_file_name = 'data'
    data_obj.sequence_length = 32
    data_obj.train_ratio = 0.8
    data_obj.min_freq = 2
    data_obj.max_vocab_size = 3000

    loaded_data = data_obj.load()

    method_obj = Method_RNN(
        'recurrent neural network text generator',
        '',
        embed_size=128,
        hidden_size=256,
        rnn_type='lstm',
        dropout=0.4,
        max_epoch=40,
        batch_size=128,
        learning_rate=0.0005,
        weight_decay=0.0001,
        early_stopping_patience=6,
    )
    method_obj.data = loaded_data

    result_dir = REPO_ROOT / 'result' / 'stage_4_result'
    result_dir.mkdir(parents=True, exist_ok=True)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir / 'RNN_')
    result_obj.result_destination_file_name = 'generation_result'
    result_obj.fold_count = None

    print('************ Start ************')
    learned_result = method_obj.run()
    result_obj.data = learned_result
    result_obj.save()
    print('************ Generated Samples ************')
    for sample in learned_result['generated_samples']:
        print(sample)
    print('Best Epoch:', learned_result['best_epoch'])
    print('Stopped Epoch:', learned_result['stopped_epoch'])
    print('Best Testing Loss:', learned_result['best_test_loss'])
    print('Testing Perplexity:', learned_result['test_perplexity'])
    print('************ Finish ************')
