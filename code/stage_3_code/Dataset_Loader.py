'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        with open(self.dataset_source_folder_path + self.dataset_file_name, 'rb') as f:
            data = pickle.load(f)

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for instance in data['train']:
            X_train.append(instance['image'])
            y_train.append(instance['label'])

        for instance in data['test']:
            X_test.append(instance['image'])
            y_test.append(instance['label'])

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }