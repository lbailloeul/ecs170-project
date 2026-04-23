'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])

class Evaluate_F1(evaluate):
    data = None

    def evaluate(self):
        return f1_score(self.data['true_y'], self.data['pred_y'], average='macro')

class Evaluate_Precision(evaluate):
    data = None

    def evaluate(self):
        return precision_score(self.data['true_y'], self.data['pred_y'], average='macro')

class Evaluate_Recall(evaluate):
    data = None

    def evaluate(self):
        return recall_score(self.data['true_y'], self.data['pred_y'], average='macro')