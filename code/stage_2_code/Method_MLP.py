'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy, Evaluate_F1, Evaluate_Precision, Evaluate_Recall
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 300
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.fc_layer_1 = nn.Linear(784, 256)
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(256, 128)
        self.activation_func_2 = nn.ReLU()
        self.fc_layer_3 = nn.Linear(128, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        y_pred = self.fc_layer_3(h2)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def fit(self, X, y, X_test, y_test):
        epochs = []
        accuracies, test_accuracies = [], []
        precisions, test_precisions = [], []
        recalls, test_recalls = [], []
        f1s, test_f1s = [], []

        losses, test_losses = [], []
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        precision_evaluator = Evaluate_Precision('training evaluator', '')
        recall_evaluator = Evaluate_Recall('training evaluator', '')
        f1_evaluator = Evaluate_F1('training evaluator', '')

        X_train_tensor = torch.FloatTensor(np.array(X))
        y_train_tensor = torch.LongTensor(np.array(y))
        X_test_tensor = torch.FloatTensor(np.array(X_test))
        y_test_tensor = torch.LongTensor(np.array(y_test))

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            self.train()
            y_pred = self.forward(X_train_tensor)
            train_loss = loss_function(y_pred, y_train_tensor)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch % 10 == 0:
                self.eval()
                y_test_pred = self.forward(X_test_tensor)
                test_loss = loss_function(y_test_pred, y_test_tensor)

                accuracy_evaluator.data = {'true_y': y_train_tensor, 'pred_y': y_pred.max(1)[1]}
                train_acc = accuracy_evaluator.evaluate()

                accuracy_evaluator.data = {'true_y': y_test_tensor, 'pred_y': y_test_pred.max(1)[1]}
                test_acc = accuracy_evaluator.evaluate()

                precision_evaluator.data = {'true_y': y_train_tensor, 'pred_y': y_pred.max(1)[1]}
                train_prec = precision_evaluator.evaluate()

                precision_evaluator.data = {'true_y': y_test_tensor, 'pred_y': y_test_pred.max(1)[1]}
                test_prec = precision_evaluator.evaluate()

                recall_evaluator.data = {'true_y': y_train_tensor, 'pred_y': y_pred.max(1)[1]}
                train_recall = recall_evaluator.evaluate()

                recall_evaluator.data = {'true_y': y_test_tensor, 'pred_y': y_test_pred.max(1)[1]}
                test_recall = recall_evaluator.evaluate()

                f1_evaluator.data = {'true_y': y_train_tensor, 'pred_y': y_pred.max(1)[1]}
                train_f1 = f1_evaluator.evaluate()

                f1_evaluator.data = {'true_y': y_test_tensor, 'pred_y': y_test_pred.max(1)[1]}
                test_f1 = f1_evaluator.evaluate()

                epochs.append(epoch)
                accuracies.append(train_acc)
                precisions.append(train_prec)
                recalls.append(train_recall)
                f1s.append(train_f1)
                losses.append(train_loss.item())

                test_accuracies.append(test_acc)
                test_precisions.append(test_prec)
                test_recalls.append(test_recall)
                test_f1s.append(test_f1)
                test_losses.append(test_loss.item())

                print('Epoch:', epoch, 'Training Accuracy:', train_acc, 'Testing Accuracy:', test_acc)


        print('Epoch:', epoch, 'Training Recall:', train_recall, 'Testing Recall:', test_recall)
        print('Epoch:', epoch, 'Training Precision:', train_prec, 'Testing Precision:', test_prec)
        print('Epoch:', epoch, 'Training F1 :', train_f1, 'Testing F1 :', test_f1)

        return epochs, accuracies, test_accuracies, losses, test_losses, precisions, test_precisions, recalls, test_recalls, f1s, test_f1s

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        epochs, accuracies, test_accuracies, losses, test_losses, precisions, test_precisions, recalls, test_recalls, f1s, test_f1s = self.fit(self.data['train']['X'], self.data['train']['y'], self.data['test']['X'], self.data['test']['y'])
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, accuracies, color='blue', label='training accuracy')
        plt.plot(epochs, test_accuracies, color = 'orange', label='testing accuracy')
        plt.title('Epoch vs Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('../../result/stage_2_result/Stage_2_accuracy.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, precisions, color='blue', label='training precision')
        plt.plot(epochs, test_precisions, color='orange', label='testing precision')
        plt.title('Epoch vs Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig('../../result/stage_2_result/Stage_2_precision.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, recalls, color='blue', label='training recall')
        plt.plot(epochs, test_recalls, color='orange', label='testing recall')
        plt.title('Epoch vs Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig('../../result/stage_2_result/Stage_2_recall.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, f1s, color='blue', label='training f1')
        plt.plot(epochs, test_f1s, color='orange', label='testing f1')
        plt.title('Epoch vs f1')
        plt.xlabel('Epoch')
        plt.ylabel('f1')
        plt.legend()
        plt.savefig('../../result/stage_2_result/Stage_2_f1.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, color='blue', label='training loss')
        plt.plot(epochs, test_losses, color = 'orange', label='testing loss')
        plt.title('Epoch vs Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('../../result/stage_2_result/Stage_2_loss.png')
        plt.close()

        torch.save(self.state_dict(), '../../result/stage_2_result/Stage_2_model.pt')
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}