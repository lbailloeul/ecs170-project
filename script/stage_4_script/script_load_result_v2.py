'''
Loads and prints the saved prediction result from the "v2" classification run
(script_rnn_classification_v2.py), which saves with fold_count = None.
'''

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.stage_1_code.Result_Loader import Result_Loader

if 1:
    result_obj = Result_Loader('saver', '')
    result_obj.result_destination_folder_path = str(
        REPO_ROOT / 'result' / 'stage_4_result' / 'RNN_v2_classification_'
    )
    result_obj.result_destination_file_name = 'prediction_result'

    result_obj.fold_count = None
    result_obj.load()
    print('Fold:', None, ', Result:', result_obj.data)
