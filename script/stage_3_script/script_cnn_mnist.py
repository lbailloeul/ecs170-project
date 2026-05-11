'''
Runner script for the MNIST CNN baseline.
 
    data_obj    — knows how to load your dataset
    method_obj  — knows how to train and predict
    result_obj  — knows how to save predictions to disk
    setting_obj — orchestrates the three above (train/test split or k-fold)
 
  Call setting_obj.load_run_save_evaluate() and it handles the rest.
  This separation makes it easy to swap in a different dataset, model, or
  evaluation strategy without touching the others.
'''


from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_3_code.Setting_Train_Test_File import Setting_Train_Test_File
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Convolutional Neural Network script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('stage 3 MNIST', '')
    data_obj.dataset_source_folder_path = '/data/stage_3_data/'
    data_obj.dataset_file_name = 'MNIST'


    # ---- Model ---------------
    method_obj = Method_CNN()

    # ---- Result saver ---------------

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '/result/stage_3_result/CNN_'
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
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    