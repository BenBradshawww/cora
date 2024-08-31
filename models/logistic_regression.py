import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import time
import pandas as pd
import os

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        # out = F.relu(out)
        return F.log_softmax(out, dim=1)


def accuracy(output, label):
    preds = output.max(1)[1].type_as(label)
    correct = preds.eq(label).double()
    correct = correct.sum()
    return correct / len(label)


def f1_metric(output, label):
    preds = output.max(1)[1].type_as(label)
    
    preds_numpy = preds.cpu().numpy()
    labels_numpy = label.cpu().numpy()
    
    f1 = f1_score(labels_numpy, preds_numpy, average='macro')
    
    return f1


def train(model,
            embeddings,
            labels,
            learning_rate,
            idx_train,
            idx_val,
            model_path,
            num_epochs:int,
            output_path:str,
            patience:int=10,
            display_every:int=100,
            seed:int=42):
    
    torch.manual_seed(seed)

    min_delta = 1e-4 
    best_loss = float('inf')

    t = time.time()
    for epoch in range(num_epochs):
        model.train()

        output = model(embeddings[idx_train])
        loss_train = F.nll_loss(output, labels[idx_train])
        acc_train = accuracy(output, labels[idx_train])

        model.zero_grad()

        loss_train.backward()

        # Gradient descent
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        model.eval()
        output = model(embeddings[idx_val])
        loss_val = F.nll_loss(output, labels[idx_val])
        acc_val = accuracy(output, labels[idx_val])

        # Early stopping criterion
        if loss_val.item() < best_loss - min_delta:
            best_loss = loss_val.item()
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                torch.save(best_model_state, model_path)
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Save Loss 
        new_row = pd.DataFrame({'epoch': epoch+1, 'loss_train': [loss_train.item()],'acc_train': [acc_train.item()], 'loss_val':[loss_val.item()], 'acc_val':[acc_val.item()]})
        if epoch == 0 and os.path.isfile(output_path):
            os.remove(output_path)
            new_row.to_csv(output_path, mode='a', header=True, index=False)
        else:
            new_row.to_csv(output_path, mode='a', header=False, index=False)

        if ((epoch+1) % display_every) == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))
    
    t_total = time.time()
    
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


def test(model, embeddings, labels, idx_test):
    model.eval()
    output = model(embeddings[idx_test])
    loss_test = F.nll_loss(output, labels[idx_test])
    acc_test = accuracy(output, labels[idx_test])
    f1_test = f1_metric(output, labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "f1= {:.4f}".format(f1_test.item()))
