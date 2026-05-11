'''
CNN method class for MNIST digit classification (0-9, 10 classes).
 
Architecture: LeNet-style, 2 conv layers + 2 FC layers.
Input:  1 × 28 × 28  (grayscale)
Output: 10 logits     (one per digit class)
 
Spatial dimension trace (explains every magic number below):
  Input          :  1 × 28 × 28
  After Conv1(5) :  6 × 24 × 24   (28 - 5 + 1 = 24)
  After Pool(2)  :  6 × 12 × 12   (24 / 2 = 12)
  After Conv2(5) : 16 ×  8 ×  8   (12 - 5 + 1 = 8)
  After Pool(2)  : 16 ×  4 ×  4   ( 8 / 2 = 4)
  Flattened      : 16 * 4 * 4 = 256
'''



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
    # ------- Hyperparameters ---------------------------------------
    learning_rate = 0.001   # Adam lr; 1e-3 is a reliable default
    max_epoch     = 30
    batch_size    = 64      # mini-batch size

    
    
    def __init__(self):
        super(Method_CNN, self).__init__()

        method.__init__(self, 'CNN-MNIST', '')
        nn.Module.__init__(self)


        # -- CONVOLUTIONAL LAYERS --------------------------------------- 
        # Conv2d(in_channels, out_channels, kernel_size)
        # MNIST is grayscale (one value per pixel).
        # A 5×5 filter/kernel is the classic LeNet choice; large enough to
        # capture strokes/curves but small enough to be fast on 28×28 images.
        # 6 then 16 filters.  Stacking more filters in deeper layers
        #   lets the network learn increasingly abstract features (edges →
        #   curves → digit parts).
        self.conv1 = nn.Conv2d(1, 6, 5)   # 1-channel input, 6 feature maps
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6-channel input, 16 feature maps

        # Batch Norm: normalizes activations after each conv so gradients
        # flow more stably + faster convergence.
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)

        # MaxPool2d(kernel, stride): keeps the largest value in each 2×2
        # window. Makes the network slightly translation-
        # invariant (a "2" shifted 1 px still looks like a "2") and halves
        # the spatial size so later layers are cheaper to compute.
        self.pool = nn.MaxPool2d(2, 2)
        
        # -- FULLY-CONNECTED LAYERS ---------------------------------------
        # After the two conv+pool pairs the spatial size is 4×4 with 16
        # channels → 16*4*4 = 256 inputs to the first FC layer.
        self.fc1 = nn.Linear(256, 120)  # 256 → 120
        self.fc2 = nn.Linear(120, 10)   # 120 → 10 (one logit per digit)

        # Dropout randomly zeros activations during training to prevent
        # co-adaptation of neurons --> model doesn't over-fit the training set.
        # p=0.5 means each neuron is kept with probability 0.5.
        # Dropout is automatically disabled during model.eval().
        self.dropout = nn.Dropout(p=0.5)




    # forward pass
    def forward(self, x):
        # ReLU(x) = max(0, x)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # → 6×12×12
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # → 16×4×4

        x = torch.flatten(x, 1)   # → 256
 
        x = F.relu(self.fc1(x))   # → 120
        x = self.dropout(x)
        x = self.fc2(x)            # → 10  (raw logits, no softmax needed
                                   #        because CrossEntropyLoss applies
                                   #        log-softmax internally)
        return x




    # training loop
    def fit(self, X, y, X_test, y_test):
        '''
        X, y : training images and labels (Python lists)
        X_test, y_test : test images and labels
 
        Returns per-epoch metric histories for plotting.
        '''       
 
        # -- CONVERTING TO TENSORS ---------------------------------------
        # np.array() first because stacking a Python list of arrays is
        # much faster than letting torch do it element-by-element.
        X_train_t = torch.FloatTensor(np.array(X)).unsqueeze(1) / 255.0
        X_test_t  = torch.FloatTensor(np.array(X_test)).unsqueeze(1) / 255.0
 
        # LongTensors bc CrossEntropyLoss requires integer class indices,not floats.
        # MNIST labels are already 0-9, so NO subtraction needed here.
        y_train_t = torch.LongTensor(np.array(y))
        y_test_t  = torch.LongTensor(np.array(y_test))
 
        # ── LOSS & OPTIMISER ──────────────────────────────────────────
        # CrossEntropyLoss = log-softmax + negative log-likelihood.
        # NOT MSE bc it treats labels as continuous values (e.g.,
        # "3" is close to "4").  Cross-entropy treats them as independent
        # categories and gives much stronger gradient signal for classification.
        loss_fn = nn.CrossEntropyLoss()
 
        # Adam adapts the learning rate per-parameter using running estimates
        # of the gradient mean and variance.  It converges faster than plain
        # SGD with momentum in most cases and requires less tuning of the learning rate.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
 
        # Evaluators
        acc_eval  = Evaluate_Accuracy('eval', '')
        prec_eval = Evaluate_Precision('eval', '')
        rec_eval  = Evaluate_Recall('eval', '')
        f1_eval   = Evaluate_F1('eval', '')
 
        epochs_hist = []
        train_accs, test_accs = [], []
        train_losses, test_losses = [], []
        train_precs, test_precs = [], []
        train_recs,  test_recs  = [], []
        train_f1s,   test_f1s   = [], []
 
        best_test_acc = 0.0
        best_state    = None
 
        N = X_train_t.shape[0]  # number of training examples
 
        for epoch in range(self.max_epoch):
 
            # -- MINI-BATCH TRAINING ---------------------------------------
            # Gradient estimates from a random subset of data are noisy
            #   but that noise actually helps escape sharp local minima.
            self.train()
            perm = torch.randperm(N)  # shuffle indices each epoch
 
            for start in range(0, N, self.batch_size):
                idx      = perm[start : start + self.batch_size]
                Xb, yb   = X_train_t[idx], y_train_t[idx]
 
                optimizer.zero_grad()          # clear old gradients
                logits = self.forward(Xb)      # forward pass
                loss   = loss_fn(logits, yb)   # compute loss
                loss.backward()                # back-propagate
                optimizer.step()               # update weights
 
            # -- EPOCH-END EVALUATION ------------------------------------
            # Switch to eval mode so dropout is OFF and batch-norm
            # uses its running statistics.
            self.eval()
            with torch.no_grad():
                # Run full datasets through for clean metric reporting.
                train_logits = self.forward(X_train_t)
                test_logits  = self.forward(X_test_t)
 
            train_loss = loss_fn(train_logits, y_train_t).item()
            test_loss  = loss_fn(test_logits,  y_test_t).item()
 
            train_pred = train_logits.max(1)[1]  # argmax → predicted class
            test_pred  = test_logits.max(1)[1]
 
            # Helper to run an evaluator and return its scalar score
            def score(evaluator, true, pred):
                evaluator.data = {'true_y': true, 'pred_y': pred}
                return evaluator.evaluate()
 
            tr_acc  = score(acc_eval,  y_train_t, train_pred)
            te_acc  = score(acc_eval,  y_test_t,  test_pred)
            tr_prec = score(prec_eval, y_train_t, train_pred)
            te_prec = score(prec_eval, y_test_t,  test_pred)
            tr_rec  = score(rec_eval,  y_train_t, train_pred)
            te_rec  = score(rec_eval,  y_test_t,  test_pred)
            tr_f1   = score(f1_eval,   y_train_t, train_pred)
            te_f1   = score(f1_eval,   y_test_t,  test_pred)
 
            # Save the model weights at the best test-accuracy epoch.
            # Bc the very last epoch might have slightly overfit;
            # keeping the checkpoint with the best validation/test score
            # is a common practice called "early stopping by best epoch".
            if te_acc > best_test_acc:
                best_test_acc = te_acc
                best_state    = {k: v.clone() for k, v in self.state_dict().items()}
 
            # Append to history
            epochs_hist.append(epoch)
            train_accs.append(tr_acc);   test_accs.append(te_acc)
            train_losses.append(train_loss); test_losses.append(test_loss)
            train_precs.append(tr_prec); test_precs.append(te_prec)
            train_recs.append(tr_rec);   test_recs.append(te_rec)
            train_f1s.append(tr_f1);     test_f1s.append(te_f1)
 
            print(f'Epoch {epoch:3d} | '
                  f'Loss {train_loss:.4f}/{test_loss:.4f} | '
                  f'Acc {tr_acc:.4f}/{te_acc:.4f}')
 
        # Restore best weights before returning
        self.load_state_dict(best_state)
        print(f'\nBest test accuracy: {best_test_acc:.4f}')
 
        return (epochs_hist,
                train_accs, test_accs,
                train_losses, test_losses,
                train_precs, test_precs,
                train_recs, test_recs,
                train_f1s, test_f1s)


    # testing/inference
    def test(self, X):
        '''Run trained model on X and return predicted labels (0-9).'''
        self.eval()
        X_t = torch.FloatTensor(np.array(X)).unsqueeze(1) / 255.0

        with torch.no_grad():
            logits = self.forward(X_t)
        return logits.max(1)[1]  # argmax — the predicted digit



    def run(self):
        print('method running...')
        print('--start training...')
 
        (epochs_hist,
         train_accs, test_accs,
         train_losses, test_losses,
         train_precs, test_precs,
         train_recs, test_recs,
         train_f1s, test_f1s) = self.fit(
            self.data['train']['X'],
            self.data['train']['y'],
            self.data['test']['X'],
            self.data['test']['y'],
        )
 
        # -- LEARNING CURVES ---------------------------------------
        # plots a metric vs. training time (epochs).
 
        result_dir = '../../result/stage_3_result/'
 
        def save_plot(y1, y2, label1, label2, ylabel, title, fname):
            plt.figure(figsize=(8, 5))
            plt.plot(epochs_hist, y1, color='steelblue',  label=label1)
            plt.plot(epochs_hist, y2, color='darkorange', label=label2)
            plt.title(title); plt.xlabel('Epoch'); plt.ylabel(ylabel)
            plt.legend(); plt.tight_layout()
            plt.savefig(result_dir + fname); plt.close()
 
        save_plot(train_losses, test_losses,
                  'train loss', 'test loss',
                  'Loss', 'Training vs Test Loss', 'MNIST_loss.png')
 
        save_plot(train_accs, test_accs,
                  'train accuracy', 'test accuracy',
                  'Accuracy', 'Training vs Test Accuracy', 'MNIST_accuracy.png')
 
        save_plot(train_precs, test_precs,
                  'train precision', 'test precision',
                  'Precision', 'Training vs Test Precision', 'MNIST_precision.png')
 
        save_plot(train_recs, test_recs,
                  'train recall', 'test recall',
                  'Recall', 'Training vs Test Recall', 'MNIST_recall.png')
 
        save_plot(train_f1s, test_f1s,
                  'train F1', 'test F1',
                  'F1 Score', 'Training vs Test F1', 'MNIST_f1.png')
 
        # ── CONFUSION MATRIX ──────────────────────────────────────────
        # Rows = true class, Columns = predicted class.
        # Diagonal = correct; off-diagonal = mistakes.
        # Common MNIST confusions: 4↔9, 3↔5, 7↔1 (similar shapes).
        print('\n-- Generating confusion matrix...')
        pred_y = self.test(self.data['test']['X'])
        true_y = torch.LongTensor(np.array(self.data['test']['y']))
 
        n_classes = 10
        cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
        for t, p in zip(true_y, pred_y):
            cm[t][p] += 1
 
        print('\nConfusion Matrix (rows=true, cols=predicted):')
        print('     ' + '  '.join(f'{i:4d}' for i in range(n_classes)))
        for i in range(n_classes):
            row = '  '.join(f'{cm[i][j]:4d}' for j in range(n_classes))
            print(f'  {i}: {row}')
 
        per_class_acc = cm.diag().float() / cm.sum(1).float()
        print('\nPer-class accuracy:')
        for i in range(n_classes):
            print(f'  Digit {i}: {per_class_acc[i]:.4f}')
 
        # Save model checkpoint
        torch.save(self.state_dict(), result_dir + 'MNIST_model.pt')
 
        print('\n--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
 
