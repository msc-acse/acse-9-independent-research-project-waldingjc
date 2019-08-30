#%%
%matplotlib inline
from sklearn.metrics import accuracy_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit

from livelossplot import PlotLosses
from pycm import *
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

import math

import pandas as pd

device = 'cpu'

class TipDataset(Dataset):

    def __init__(self, csv_file):
        self.tips = pd.read_csv(csv_file) # accesses datafile
        self.data = np.array(self.tips.values)[1:,1:-4] # excludes the header line, id and targets
        self.data = torch.from_numpy(self.data) # transforms numpy array into torch tensor
        self.data[:,3:6] = self.data[:,3:6] / 1e5
        self.labels = np.array(self.tips.values)[1:,-4] # excludes header line, id and features
        self.labels = torch.from_numpy(self.labels) # transforms numpy array into torch tensor
        self.labels = self.labels / 1e5


    def __len__(self):
        return len(self.tips)

    def __getitem__(self, idx):
        tip_id = self.tips.iloc[idx, 0]
        features = self.tips.iloc[idx, 1:-4]
        features = np.array([features])
        target = self.tips.iloc[idx, -4]
        target = np.array([target])
        sample = {'tip_id': tip_id, 'features': features, 'target': target}

        return sample

class FirstNet(nn.Module):
    def __init__(self):
        super(FirstNet, self).__init__()
        self.linear_1 = nn.Linear(7, 25)
        self.linear_2 = nn.Linear(25, 25)
        self.linear_3 = nn.Linear(25, 1)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        z1 = self.linear_1(x)
        a1 = self.activation(z1)
        z2 = self.linear_2(a1)
        a2 = self.activation(z2)
        z3 = self.linear_3(a2)
        return z3

class SecondNet(nn.Module):
    def __init__(self):
        super(SecondNet, self).__init__()
        self.linear_1 = nn.Linear(7, 25)
        self.linear_2 = nn.Linear(25, 25)
        self.linear_3 = nn.Linear(25, 10)
        self.linear_4 = nn.Linear(10, 1)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        z1 = self.linear_1(x)
        a1 = self.activation(z1)
        z2 = self.linear_2(a1)
        a2 = self.activation(z2)
        z3 = self.linear_3(a2)
        a3 = self.activation(z3)
        z4 = self.linear_4(a3)
        return z4

class ThirdNet(nn.Module):
    def __init__(self):
        super(ThirdNet, self).__init__()
        self.linear_1 = nn.Linear(7, 25)
        self.linear_2 = nn.Linear(25, 100)
        self.linear_3 = nn.Linear(100, 200)
        self.linear_4 = nn.Linear(200, 50)
        self.linear_5 = nn.Linear(50, 1)
        #self.activation = nn.ReLU()
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        z1 = self.linear_1(x)
        a1 = self.activation(z1)
        z2 = self.linear_2(a1)
        a2 = self.activation(z2)
        z3 = self.linear_3(a2)
        a3 = self.activation(z3)
        z4 = self.linear_4(a3)
        a4 = self.activation(z4)
        z5 = self.linear_5(a4)
        return z5

def dataloader_make(ICGT_tips_train):

    shuffler = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(ICGT_tips_train.data, ICGT_tips_train.labels) # shuffles training set to extract a validation set
    indices = [(train_idx, val_idx) for train_idx, val_idx in shuffler][0]

    f_train, t_train = ICGT_tips_train.data[indices[0]].float(), ICGT_tips_train.labels[indices[0]].float()
    f_val, t_val = ICGT_tips_train.data[indices[1]].float(), ICGT_tips_train.labels[indices[1]].float()

    shuffler = ShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(f_val, t_val) # shuffles validation set to extract a test set
    indices = [(train_idx, val_idx) for train_idx, val_idx in shuffler][0]
    
    f_val, t_val = ICGT_tips_train.data[indices[0]].float(), ICGT_tips_train.labels[indices[0]].float()
    f_test, t_test = ICGT_tips_train.data[indices[1]].float(), ICGT_tips_train.labels[indices[1]].float()

    ICGT_tips_train = TensorDataset(f_train, t_train.float())
    ICGT_tips_val = TensorDataset(f_val, t_val.float())
    ICGT_tips_test = TensorDataset(f_test, t_test.float())

    train_loader = DataLoader(ICGT_tips_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ICGT_tips_val, batch_size=test_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ICGT_tips_test, batch_size=test_batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, f_test, t_test.float()

def train(model, optimiser, criterion, data_loader):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in data_loader:
        #X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        a2 = model(X)
        #print(a2[0], y[0])
        loss = criterion(a2, y.unsqueeze(1))
        loss.backward()
        train_loss += loss*X.size(0)
        y_pred = a2
        train_acc += explained_variance_score(y.cpu().numpy(), y_pred.detach().cpu().numpy())*X.size(0)
        optimiser.step()  
        
    return train_loss/len(data_loader.dataset), train_acc/len(data_loader.dataset)

def validate(model, criterion, data_loader):
    model.eval()
    val_loss, val_acc = 0., 0.
    for X, y in data_loader:
        with torch.no_grad():
            #X, y = X.to(device), y.to(device)
            a2 = model(X)
            loss = criterion(a2, y.unsqueeze(1))
            val_loss += loss*X.size(0)
            y_pred = a2
            val_acc += explained_variance_score(y.unsqueeze(1).cpu().numpy(), y_pred.cpu().numpy())*X.size(0)
            
    return val_loss/len(data_loader.dataset), val_acc/len(data_loader.dataset)


seed = 42
lr = 1e-3
mom = 0.9
batch_size = 1000
test_batch_size = 1000
n_epochs = 20

#model = FirstNet()
#model = SecondNet()
model = ThirdNet()
#optimiser = torch.optim.SGD(model.parameters(), lr=lr)#, momentum=mom)
optimiser = torch.optim.Adam(model.parameters())
criterion = nn.SmoothL1Loss()

training_file = "Data/fracture_k_sequence_all3_edit.csv"
#test_file = "Data/fracture_k_sequence_all_edit_test.csv"
#training_file = "Data/fracture_k_sequence1.csv"
#test_file = "Data/fracture_k_sequence1_test.csv"
#training_file = "proto.csv"
#test_file = "proto.csv" # placeholder csv
ICGT_tips_train = TipDataset(training_file) #get the data from ICGT for tip features as a csv
#ICGT_tips_test = TipDataset(test_file) #same as above, except a different set for testing
train_loader, val_loader, f_test, t_test = dataloader_make(ICGT_tips_train)#, ICGT_tips_test)

liveloss = PlotLosses()
for epoch in range(n_epochs):
    logs = {}
    train_loss, train_acc = train(model, optimiser, criterion, train_loader)

    logs['' + 'log loss'] = train_loss.item()
    logs['' + 'accuracy'] = train_acc.item()
    
    val_loss, val_acc = validate(model, criterion, val_loader)
    logs['val_' + 'log loss'] = val_loss.item()
    logs['val_' + 'accuracy'] = val_acc.item()
    
    #print(epoch)
    liveloss.update(logs)
    liveloss.draw()
    
model.eval()
#output = model(ICGT_tips_test.data.float())
#truth = ICGT_tips_test.labels
output = model(f_test)
truth = t_test
errors = []
outputs = []
truths = []
avg_error = 0
avg_value = 0
max_error = 0
bad_index = 0
for i in range(len(truth)):
    error = abs(truth[i] - output[i])
    error = error.item()
    errors.append(error)
    outputs.append(output[i].item())
    truths.append(truth[i].item())
    avg_error += error
    avg_value += abs(truth[i].item())
    if error > max_error:
        max_error = error
        bad_index = i
        
avg_error /= len(truth)
avg_value /= len(truth)

truths, outputs = (list(t) for t in zip(*sorted(zip(truths, outputs))))

print()
print("xxxxxxxxxxxxxxxxxxxxx")
print('Bad Output  : ' + str(output[bad_index].item()))
print('Bad Truth   : ' + str(truth[bad_index].item()))
print('Max Error   : ' + str(max_error))
print("xxxxxxxxxxxxxxxxxxxxx")
print('Avg Value   : ' + str(avg_value))
print('Avg Error   : ' + str(avg_error))
print('Avg % Error : ' + str(100*avg_error/avg_value))
print("xxxxxxxxxxxxxxxxxxxxx")
plt.hist(errors, 100)
plt.show()
plt.plot(outputs, truths, '+')
mini = min(np.minimum(outputs, truths))
maxi = max(np.maximum(outputs, truths))
plt.plot([mini, maxi], [mini, maxi])
plt.show()
plt.plot(outputs, '+')
plt.plot(truths)
plt.show()
#for i in range(len(truth)):
#    print(output[i].item(), truth[i].item(), errors[i])



#%%
