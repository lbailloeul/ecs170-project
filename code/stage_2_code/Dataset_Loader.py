'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_train_file_name = None
    dataset_test_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _load_one(self, filename):
        X = []
        y = []
        f = open(self.dataset_source_folder_path + filename, 'r')
        for line in f:
            line = line.strip('\n').rstrip(',')
            elements = [int(i) for i in line.split(',')]
            y.append(elements[0])
            X.append(elements[1:])
        f.close()
        return X, y

    def load(self):
        print('loading data...')
        X_train, y_train = self._load_one(self.dataset_train_file_name)
        X_test, y_test = self._load_one(self.dataset_test_file_name)
        return {'train': {'X': X_train, 'y': y_train},
                'test': {'X': X_test, 'y': y_test}}