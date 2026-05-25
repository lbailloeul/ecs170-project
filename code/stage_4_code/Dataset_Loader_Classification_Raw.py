'''
Raw-text Dataset_Loader for the "v2" stage 4 text classification run.

Reads the IMDb folder layout and returns RAW review strings; the cleaning /
tokenising / vocab / padding all happen inside Method_RNN_Classification_v2.

    <dataset_source_folder_path><dataset_file_name>/
        train/pos/*.txt
        train/neg/*.txt
        test/pos/*.txt
        test/neg/*.txt

Returns data in the standard pipeline shape:
    {
        'train': {'X': [str, ...], 'y': [int, ...]},
        'test':  {'X': [str, ...], 'y': [int, ...]},
    }

Labels: 1 = positive, 0 = negative.

Kept separate from the existing Dataset_Loader_Classification (which pre-encodes
to id sequences with lengths) so the two classification implementations don't
interfere.
'''

import os
from code.base_class.dataset import dataset


class Dataset_Loader_Classification_Raw(dataset):

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = None
        self.dataset_file_name = None

    # ------------------------------------------------------------------
    def _load_split(self, split: str) -> tuple[list[str], list[int]]:
        '''Read all .txt files from train/ or test/, both pos and neg.'''
        base = os.path.join(
            self.dataset_source_folder_path,
            self.dataset_file_name,
            split,
        )
        texts, labels = [], []
        for label, folder in [(1, 'pos'), (0, 'neg')]:
            folder_path = os.path.join(base, folder)
            for fname in sorted(os.listdir(folder_path)):
                if not fname.endswith('.txt'):
                    continue
                fpath = os.path.join(folder_path, fname)
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    texts.append(f.read())
                labels.append(label)
        return texts, labels

    # ------------------------------------------------------------------
    def load(self):
        print('loading dataset...')

        train_X, train_y = self._load_split('train')
        test_X,  test_y  = self._load_split('test')

        print(f'  train samples: {len(train_X)}  |  test samples: {len(test_X)}')

        self.data = {
            'train': {'X': train_X, 'y': train_y},
            'test':  {'X': test_X,  'y': test_y},
        }
        return self.data
