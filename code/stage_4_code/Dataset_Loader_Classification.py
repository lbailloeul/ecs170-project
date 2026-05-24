'''
Concrete IO class for the stage 4 text classification dataset.
'''

import re
from collections import Counter
from pathlib import Path

from code.base_class.dataset import dataset


class Dataset_Loader_Classification(dataset):
    data = None
    dataset_source_folder_path = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.max_len = 256
        self.min_freq = 2
        self.max_vocab_size = 20000
        self.max_files_per_label = None
        self.label_to_idx = {'neg': 0, 'pos': 1}
        self.idx_to_label = ['neg', 'pos']

    def _dataset_root(self):
        folder = self.dataset_source_folder_path or '../../data/stage_4_data/text_classification/'
        path = Path(folder)
        if path.exists():
            return path

        repo_root = Path(__file__).resolve().parents[2]
        fallback = repo_root / 'data' / 'stage_4_data' / 'text_classification'
        if fallback.exists():
            return fallback

        return path

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())

    def _read_split(self, root, split_name):
        texts = []
        labels = []

        for label_name in self.idx_to_label:
            label_dir = root / split_name / label_name
            file_paths = sorted(label_dir.glob('*.txt'))
            if self.max_files_per_label is not None:
                file_paths = file_paths[:self.max_files_per_label]

            for file_path in file_paths:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    texts.append(f.read())
                labels.append(self.label_to_idx[label_name])

        if not texts:
            raise ValueError('No text files found for split ' + split_name + ' in ' + str(root))

        return texts, labels

    def _build_vocabulary(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        special_tokens = ['<pad>', '<unk>']
        vocab_tokens = [
            token for token, count in counter.most_common()
            if count >= self.min_freq
        ]
        if self.max_vocab_size is not None:
            vocab_tokens = vocab_tokens[:max(0, self.max_vocab_size - len(special_tokens))]

        idx_to_token = special_tokens + vocab_tokens
        token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
        return token_to_idx, idx_to_token

    def _encode_text(self, text, token_to_idx):
        unk_idx = token_to_idx['<unk>']
        token_ids = [token_to_idx.get(token, unk_idx) for token in self._tokenize(text)]
        if not token_ids:
            token_ids = [unk_idx]

        token_ids = token_ids[:self.max_len]
        length = len(token_ids)
        pad_count = self.max_len - length
        if pad_count > 0:
            token_ids += [token_to_idx['<pad>']] * pad_count

        return token_ids, length

    def _encode_many(self, texts, token_to_idx):
        X = []
        lengths = []

        for text in texts:
            token_ids, length = self._encode_text(text, token_to_idx)
            X.append(token_ids)
            lengths.append(length)

        return X, lengths

    def load(self):
        print('loading data...')

        data_root = self._dataset_root()
        train_texts, y_train = self._read_split(data_root, 'train')
        test_texts, y_test = self._read_split(data_root, 'test')

        token_to_idx, idx_to_token = self._build_vocabulary(train_texts)
        X_train, train_lengths = self._encode_many(train_texts, token_to_idx)
        X_test, test_lengths = self._encode_many(test_texts, token_to_idx)

        return {
            'train': {'X': X_train, 'y': y_train, 'lengths': train_lengths},
            'test': {'X': X_test, 'y': y_test, 'lengths': test_lengths},
            'token_to_idx': token_to_idx,
            'idx_to_token': idx_to_token,
            'vocab_size': len(idx_to_token),
            'pad_idx': token_to_idx['<pad>'],
            'unk_idx': token_to_idx['<unk>'],
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'max_len': self.max_len,
        }
