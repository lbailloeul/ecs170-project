from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from code.stage_4_code.Result_Saver import Result_Saver


if 1:
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader_Classification('stage 4 text classification', '')
    data_obj.dataset_source_folder_path = str(REPO_ROOT / 'data' / 'stage_4_data' / 'text_classification') + '/'
    data_obj.max_len = 250
    data_obj.min_freq = 2
    data_obj.max_vocab_size = 20000

    loaded_data = data_obj.load()

    method_obj = Method_RNN_Classification(
        'recurrent neural network text classifier',
        '',
        embed_size=128,
        hidden_size=128,
        rnn_type='gru',
        num_layers=1,
        dropout=0.25,
        max_epoch=8,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.00001,
        bidirectional=True,
        pooling='mean_max',
        fc_hidden_size=128,
        early_stopping_patience=2,
    )
    method_obj.data = loaded_data

    result_dir = REPO_ROOT / 'result' / 'stage_4_result'
    result_dir.mkdir(parents=True, exist_ok=True)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir / 'RNN_classification_')
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = None

    print('************ Start ************')
    learned_result = method_obj.run()
    result_obj.data = learned_result
    result_obj.save()

    metrics = learned_result['final_test_metrics']
    print('************ Final Test Performance ************')
    print('Accuracy:', metrics['accuracy'])
    print('Precision:', metrics['precision'])
    print('Recall:', metrics['recall'])
    print('F1:', metrics['f1'])
    print('Loss:', learned_result['final_test_loss'])
    print('Best Epoch:', learned_result['best_epoch'])
    print('Stopped Epoch:', learned_result['stopped_epoch'])
    print('************ Finish ************')
