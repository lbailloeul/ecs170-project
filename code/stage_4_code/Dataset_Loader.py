'''
Concrete IO class for the stage 4 text generation dataset.
'''

import csv
import random
import re
from collections import Counter
from pathlib import Path

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.sequence_length = 12
        self.train_ratio = 0.8
        self.random_seed = 2
        self.min_freq = 1
        self.max_vocab_size = None
        self.max_rows = None
        self.stride = 1

    def _dataset_path(self):
        folder = self.dataset_source_folder_path or '../../data/stage_4_data/text_generation/'
        file_name = self.dataset_file_name or 'data'
        path = Path(folder) / file_name
        if path.exists():
            return path

        repo_root = Path(__file__).resolve().parents[2]
        fallback = repo_root / 'data' / 'stage_4_data' / 'text_generation' / file_name
        if fallback.exists():
            return fallback

        return path

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", text.lower())

    def _read_jokes(self, path):
        jokes = []
        with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                joke = (row.get('Joke') or '').strip()
                if joke:
                    jokes.append(joke)
                if self.max_rows is not None and len(jokes) >= self.max_rows:
                    break
        return jokes

    def _build_vocabulary(self, jokes):
        counter = Counter()
        for joke in jokes:
            counter.update(self._tokenize(joke))

        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        vocab_tokens = [
            token for token, count in counter.most_common()
            if count >= self.min_freq
        ]
        if self.max_vocab_size is not None:
            vocab_tokens = vocab_tokens[:max(0, self.max_vocab_size - len(special_tokens))]

        idx_to_token = special_tokens + vocab_tokens
        token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
        return token_to_idx, idx_to_token

    def _encode_joke(self, joke, token_to_idx):
        unk_idx = token_to_idx['<unk>']
        return (
            [token_to_idx['<bos>']]
            + [token_to_idx.get(token, unk_idx) for token in self._tokenize(joke)]
            + [token_to_idx['<eos>']]
        )

    def _make_sequences(self, jokes, token_to_idx):
        pad_idx = token_to_idx['<pad>']
        X, y = [], []
        stride = max(1, self.stride)

        for joke in jokes:
            encoded = self._encode_joke(joke, token_to_idx)
            if len(encoded) < 2:
                continue

            for start in range(0, len(encoded) - 1, stride):
                input_seq = encoded[start:start + self.sequence_length]
                target_seq = encoded[start + 1:start + self.sequence_length + 1]

                if len(input_seq) < self.sequence_length:
                    input_seq += [pad_idx] * (self.sequence_length - len(input_seq))
                if len(target_seq) < self.sequence_length:
                    target_seq += [pad_idx] * (self.sequence_length - len(target_seq))

                X.append(input_seq)
                y.append(target_seq)

        return X, y

    def load(self):
        print('loading data...')

        data_path = self._dataset_path()
        jokes = self._read_jokes(data_path)
        if not jokes:
            raise ValueError('No jokes were loaded from ' + str(data_path))

        rng = random.Random(self.random_seed)
        shuffled_jokes = jokes[:]
        rng.shuffle(shuffled_jokes)

        split_idx = int(len(shuffled_jokes) * self.train_ratio)
        split_idx = min(max(split_idx, 1), len(shuffled_jokes) - 1)
        train_jokes = shuffled_jokes[:split_idx]
        test_jokes = shuffled_jokes[split_idx:]

        token_to_idx, idx_to_token = self._build_vocabulary(train_jokes)
        X_train, y_train = self._make_sequences(train_jokes, token_to_idx)
        X_test, y_test = self._make_sequences(test_jokes, token_to_idx)

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
            'token_to_idx': token_to_idx,
            'idx_to_token': idx_to_token,
            'vocab_size': len(idx_to_token),
            'pad_idx': token_to_idx['<pad>'],
            'unk_idx': token_to_idx['<unk>'],
            'bos_idx': token_to_idx['<bos>'],
            'eos_idx': token_to_idx['<eos>'],
            'sequence_length': self.sequence_length,
            'train_jokes': train_jokes,
            'test_jokes': test_jokes,
            'seed_text': train_jokes[0],
        }
