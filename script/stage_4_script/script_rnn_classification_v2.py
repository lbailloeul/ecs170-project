'''
Runner script for the "v2" Classification RNN (separate from the existing
script_rnn_classification.py).

    data_obj    — knows how to load your dataset (raw IMDb review strings)
    method_obj  — knows how to train and predict (MethodRNN, BiLSTM)
    result_obj  — knows how to save predictions to disk
    setting_obj — orchestrates the three above (train/test split or k-fold)

  Call setting_obj.load_run_save_evaluate() and it handles the rest.
'''

from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.stage_4_code.Dataset_Loader_Classification_Raw import Dataset_Loader_Classification_Raw
from code.stage_4_code.Method_RNN_Classification_v2 import MethodRNN
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_4_code.Setting_Train_Test_File import Setting_Train_Test_File
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy


#---- Recurrent Neural Network script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- object initialization section ---------------
    data_obj = Dataset_Loader_Classification_Raw('stage 4 text classification (v2)', '')
    data_obj.dataset_source_folder_path = str(REPO_ROOT / 'data' / 'stage_4_data') + '/'
    data_obj.dataset_file_name = 'text_classification'

    # ---- Model ---------------
    method_obj = MethodRNN()      # max_epoch = 15 (see class hyperparameters)

    # ---- Result saver ---------------
    result_dir = REPO_ROOT / 'result' / 'stage_4_result'
    result_dir.mkdir(parents=True, exist_ok=True)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir / 'RNN_v2_classification_')
    result_obj.result_destination_file_name = 'prediction_result'

    # ---- Setting (train/test split) ---------------
    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    setting_obj = Setting_Train_Test_File('train test split', '')

    # ---- Evaluator ---------------
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('RNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
