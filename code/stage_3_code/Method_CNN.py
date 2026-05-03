from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy, Evaluate_F1, Evaluate_Precision, Evaluate_Recall
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

class Method_CNN(method, nn.Module):
    def __init__(self):
        super(Method_CNN, self).__init__()

        method.__init__(self, 'convolutional neural network', '')
        nn.Module.__init__(self)

        self.learning_rate = 0.001
        self.max_epoch = 300

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 25 * 20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 40)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

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

        X_train_tensor = torch.FloatTensor(np.array(X)).permute(0, 3, 1, 2) / 255.0
        X_test_tensor = torch.FloatTensor(np.array(X_test)).permute(0, 3, 1, 2) / 255.0
        y_train_tensor = torch.LongTensor(np.array(y)) - 1
        y_test_tensor = torch.LongTensor(np.array(y_test)) - 1

        # augment: horizontal flip + vertical flip (upside down)
        X_flip_h = torch.flip(X_train_tensor, dims=[3])
        X_flip_v = torch.flip(X_train_tensor, dims=[2])
        X_train_tensor = torch.cat([X_train_tensor, X_flip_h, X_flip_v], dim=0)
        y_train_tensor = torch.cat([y_train_tensor] * 3, dim=0)

        best_test_acc = 0.0
        best_state = None

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

            if epoch % 1 == 0:
                self.eval()
                y_test_pred = self.forward(X_test_tensor)
                test_loss = loss_function(y_test_pred, y_test_tensor)

                accuracy_evaluator.data = {'true_y': y_train_tensor, 'pred_y': y_pred.max(1)[1]}
                train_acc = accuracy_evaluator.evaluate()

                accuracy_evaluator.data = {'true_y': y_test_tensor, 'pred_y': y_test_pred.max(1)[1]}
                test_acc = accuracy_evaluator.evaluate()

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}

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

        self.load_state_dict(best_state)
        return epochs, accuracies, test_accuracies, losses, test_losses, precisions, test_precisions, recalls, test_recalls, f1s, test_f1s

    def test(self, X):
        # do the testing, and result the result
        self.eval()
        X_tensor = torch.FloatTensor(np.array(X)).permute(0, 3, 1, 2) / 255.0

        with torch.no_grad():
            y_pred = self.forward(X_tensor)

        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1] + 1

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
        plt.savefig('../../result/stage_3_result/Stage_3_accuracy.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, precisions, color='blue', label='training precision')
        plt.plot(epochs, test_precisions, color='orange', label='testing precision')
        plt.title('Epoch vs Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig('../../result/stage_3_result/Stage_3_precision.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, recalls, color='blue', label='training recall')
        plt.plot(epochs, test_recalls, color='orange', label='testing recall')
        plt.title('Epoch vs Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig('../../result/stage_3_result/Stage_3_recall.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, f1s, color='blue', label='training f1')
        plt.plot(epochs, test_f1s, color='orange', label='testing f1')
        plt.title('Epoch vs f1')
        plt.xlabel('Epoch')
        plt.ylabel('f1')
        plt.legend()
        plt.savefig('../../result/stage_3_result/Stage_3_f1.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, color='blue', label='training loss')
        plt.plot(epochs, test_losses, color = 'orange', label='testing loss')
        plt.title('Epoch vs Loss (Zoomed)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('../../result/stage_3_result/Stage_3_loss.png')
        plt.close()

        torch.save(self.state_dict(), '../../result/stage_3_result/Stage_3_model.pt')
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}