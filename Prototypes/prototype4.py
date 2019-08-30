#%%
#%matplotlib inline
from sklearn.metrics import explained_variance_score as evs
from sklearn.model_selection import ShuffleSplit

from livelossplot import PlotLosses
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

import math

import pandas as pd

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name())
    print("CUDA")
else:
    device = torch.device("cpu")
    print("CPU")

class TipDataset(Dataset):
    '''
    Custom dataset to read and hold the tip data as data (features) and lbls (targets/labels)
    Applies divisor to SIF values to bring down to order 10s

    Arguments:
    csv_file    : file address for the csv containing the data      // string
    sif         : selects SIF element [1, 2, 3] for training        // integer // default: 0
    divisor     : amount by which to divide the SIF elements        // integer // default: 1e-5

    Parameters:
    sif_idx     : the actual index in the csv for the selected SIF  // integer
    select      : allows selection of non-contiguous features       // array

    Attributes:
    tips        : the raw csv file read by pandas                   // pandas dataframe
    data        : the features from the csv                         // tensor
    ndat        : the number of features in the set                 // integer
    lbls        : the targets (labels) from the csv                 // tensor

    Returns:
    null
    '''

    def __init__(self, csv_file, sif=0, divisor=1e-5):
        sif_idx = 0
        if (sif == 1) or (sif == 2) or (sif == 3): # converts SIF selection into csv index
            sif_idx = (5 - sif) * -1
        else:
            print("No SIF element selected, defaulting to SIF I")
            sif_idx = -4
        
        self.tips = pd.read_csv(csv_file) # accesses datafile
        select = np.r_[0:3,-8:-4] # feat1
        #select = np.r_[-11:-4] # feat2
        #select = np.r_[0:3,-11:-4] # feat3
        #select = np.r_[3:6,-8:-4] # feat4
        self.data = np.array(self.tips.values)[:,select] # excludes the header line, id and targets
        #self.data = np.array(self.tips.values)[:,:-4]
        self.data = torch.from_numpy(self.data) # transforms numpy array into torch tensor
        self.data[:,3:6] = self.data[:,3:6] * divisor # divides features to ensure the net can handle them
        self.ndat = len(self.data[0]) # number of features
        self.lbls = np.array(self.tips.values)[:,sif_idx] # excludes header line, id, features and undesired targets
        #self.lbls = np.array(self.tips.values)[:,sif_idx]
        self.lbls = torch.from_numpy(self.lbls) # transforms numpy array into torch tensor
        self.lbls = self.lbls * divisor # divides targets in line with features as above


    # def __len__(self):
    #     return len(self.tips)

    # def __getitem__(self, idx):
    #     tip_id = self.tips.iloc[idx, 0]
    #     features = self.tips.iloc[idx, 1:-4]
    #     features = np.array([features])
    #     target = self.tips.iloc[idx, -4]
    #     target = np.array([target])
    #     sample = {'tip_id': tip_id, 'features': features, 'target': target}

    #     return sample

class FirstNet(nn.Module):
    '''
    A simple neural network for training

    Architecture: input -> 25(activation) -> 25(activation) -> 1

    Arguments:
    features    : number of training features       // integer // default: 7

    Parameters:
    z1          : first hidden layer                // tensor
    a1          : first activation layer            // tensor
    z2          : second hidden layer               // tensor
    a2          : second activation layer           // tensor
    z3          : final hidden layer                // tensor

    Attributes:
    lin_1       : linear layer of features -> 25    // nn.Linear
    lin_2       : linear layer of 25 -> 25          // nn.Linear
    lin_3       : linear layer of 25 -> 1           // nn.Linear
    activ       : activation layer, here LeakyReLU  // nn.LeakyReLU

    Returns:
    z3          : final hidden layer                // tensor
    '''

    def __init__(self, features=7):
        super(FirstNet, self).__init__()
        self.lin_1 = nn.Linear(features, 25) # structures for the network
        self.lin_2 = nn.Linear(25, 25)
        self.lin_3 = nn.Linear(25, 1)
        self.activ = nn.LeakyReLU() # activation function
        
    def forward(self, x):
        z1 = self.lin_1(x)
        a1 = self.activ(z1) # activation after each fully connected layer
        z2 = self.lin_2(a1)
        a2 = self.activ(z2)
        z3 = self.lin_3(a2)
        return z3

class SecondNet(nn.Module):
    '''
    A simple neural network for training, but with an extra hidden layer

    Architecture: input -> 25(activation) -> 25(activation) -> 10(activation) -> 1

    Arguments:
    features    : number of training features       // integer // default: 7

    Parameters:
    z1          : first hidden layer                // tensor
    a1          : first activation layer            // tensor
    z2          : second hidden layer               // tensor
    a2          : second activation layer           // tensor
    z3          : third hidden layer                // tensor
    a3          : third activation layer            // tensor
    z4          : final hidden layer                // tensor

    Attributes:
    lin_1       : linear layer of features -> 25    // nn.Linear
    lin_2       : linear layer of 25 -> 25          // nn.Linear
    lin_3       : linear layer of 25 -> 10          // nn.Linear
    lin_4       : linear layer of 10 -> 1           // nn.Linear
    activ       : activation layer, here LeakyReLU  // nn.LeakyReLU

    Returns:
    z4          : final hidden layer                // tensor
    '''

    def __init__(self, features=7):
        super(SecondNet, self).__init__()
        self.lin_1 = nn.Linear(features, 25) # structures for the network
        self.lin_2 = nn.Linear(25, 25)
        self.lin_3 = nn.Linear(25, 10)
        self.lin_4 = nn.Linear(10, 1)
        self.activ = nn.LeakyReLU() # activation function
        
    def forward(self, x):
        z1 = self.lin_1(x)
        a1 = self.activ(z1) # activation after each fully connected layer
        z2 = self.lin_2(a1)
        a2 = self.activ(z2)
        z3 = self.lin_3(a2)
        a3 = self.activ(z3)
        z4 = self.lin_4(a3)
        return z4

class ThirdNet(nn.Module):
    '''
    A simple neural network for training, but with massively increased neurons

    Architecture: input -> 25(activation) -> 100(activation) -> 200(activation) -> 50(activation) -> 1

    Arguments:
    features    : number of training features       // integer // default: 7

    Parameters:
    z1          : first hidden layer                // tensor
    a1          : first activation layer            // tensor
    z2          : second hidden layer               // tensor
    a2          : second activation layer           // tensor
    z3          : third hidden layer                // tensor
    a3          : third activation layer            // tensor
    z4          : fourth hidden layer               // tensor
    a4          : fourth activation layer           // tensor
    z5          : final hidden layer                // tensor

    Attributes:
    lin_1       : linear layer of features -> 25    // nn.Linear
    lin_2       : linear layer of 25 -> 100         // nn.Linear
    lin_3       : linear layer of 100 -> 200        // nn.Linear
    lin_4       : linear layer of 200 -> 50         // nn.Linear
    lin_5       : linear layer of 50 -> 1           // nn.Linear
    activ       : activation layer, here LeakyReLU  // nn.LeakyReLU

    Returns:
    z5          : final hidden layer                // tensor
    '''

    def __init__(self, features=7):
        super(ThirdNet, self).__init__()
        self.lin_1 = nn.Linear(features, 25) # structures for the network
        self.lin_2 = nn.Linear(25, 100)
        self.lin_3 = nn.Linear(100, 200)
        self.lin_4 = nn.Linear(200, 50)
        self.lin_5 = nn.Linear(50, 1)
        #self.activ = nn.Sigmoid()
        #self.activ = nn.ReLU()
        self.activ = nn.LeakyReLU() # activation function
        
    def forward(self, x):
        z1 = self.lin_1(x)
        a1 = self.activ(z1) # activation after each fully connected layer
        z2 = self.lin_2(a1)
        a2 = self.activ(z2)
        z3 = self.lin_3(a2)
        a3 = self.activ(z3)
        z4 = self.lin_4(a3)
        a4 = self.activ(z4)
        z5 = self.lin_5(a4)
        return z5

def dataldr_make(ICGT_tips, trn_batch_size, tst_batch_size):
    '''
    A routine to construct the dataloaders for training, validation and testing, as well as features and targets for testing
    Uses ShuffleSplit to create the sets

    Arguments:
    ICGT_tips       : holds the total dataset                   // tensor dataset
    trn_batch_size  : the training batch size                   // integer
    tst_batch_size  : the test batch size                       // integer

    Parameters:
    ss              : the ShuffleSplit object                   // ShuffleSplit
    idx             : the indices returned by the shuffle split // [integer, integer]
    f_trn           : the training features                     // tensor
    t_trn           : the training targets                      // tensor
    f_vtt           : the validation and testing features       // tensor
    t_vtt           : the validation and testing targets        // tensor
    f_val           : the validation features                   // tensor
    t_val           : the validation targets                    // tensor
    f_tst           : the testing features                      // tensor
    t_tst           : the testing targets                       // tensor
    ICGT_tips_trn   : the training tensor dataset               // tensor dataset
    ICGT_tips_val   : the validation tensor dataset             // tensor dataset
    ICGT_tips_tst   : the testing tensor dataset                // tensor dataset
    trn_ldr         : the training dataloader                   // dataloader
    val_ldr         : the validation dataloader                 // dataloader
    tst_ldr         : the testing dataloader [vestigial]        // dataloader

    Returns:
    trn_ldr         : the training dataloader                   // dataloader
    val_ldr         : the validation dataloader                 // dataloader
    tst_ldr         : the testing dataloader [vestigial]        // dataloader
    f_tst           : the testing features                      // tensor
    t_tst           : the testing targets                       // tensor
    '''

    ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(ICGT_tips.data, ICGT_tips.lbls) # shuffles training set to extract a validation set
    idx = [(trn_idx, val_idx) for trn_idx, val_idx in ss][0] # return shuffled indices

    f_trn = ICGT_tips.data[idx[0]].float() # training features
    t_trn = ICGT_tips.lbls[idx[0]].float() # training targets
    f_vtt = ICGT_tips.data[idx[1]].float() # testing and validation features
    t_vtt = ICGT_tips.lbls[idx[1]].float() # testing and validation targets

    ss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(f_vtt, t_vtt) # shuffles validation set to extract a test set
    idx = [(trn_idx, val_idx) for trn_idx, val_idx in ss][0] # return shuffled indices
    
    f_val = f_vtt[idx[0]].float() # validation features
    t_val = t_vtt[idx[0]].float() # validation targets
    f_tst = f_vtt[idx[1]].float() # testing features
    t_tst = t_vtt[idx[1]].float() # testing targets

    ICGT_tips_trn = TensorDataset(f_trn, t_trn.float()) # training tensor dataset
    ICGT_tips_val = TensorDataset(f_val, t_val.float()) # validation tensor dataset
    ICGT_tips_tst = TensorDataset(f_tst, t_tst.float()) # testing tensor dataset

    trn_ldr = DataLoader(ICGT_tips_trn, batch_size=trn_batch_size, shuffle=True, num_workers=0) # training dataloader
    val_ldr = DataLoader(ICGT_tips_val, batch_size=tst_batch_size, shuffle=False, num_workers=0) # validation dataloader
    tst_ldr = DataLoader(ICGT_tips_tst, batch_size=tst_batch_size, shuffle=False, num_workers=0) # testing dataloader

    return trn_ldr, val_ldr, tst_ldr, f_tst, t_tst

def trn(mdl, opti, crit, ldr):
    '''
    This routine handles the training loop

    Arguments:
    mdl         : the model to be trained                       // nn.Module
    opti        : the optimiser object                          // optim
    crit        : the criterion (loss) function                 // nn loss function
    ldr         : the dataloader for training                   // dataloader

    Parameters:
    trn_los     : tracks the training loss                      // float
    trn_acc     : tracks the training accuracy using evs        // float
    X           : the feature set for the current datapoint     // tensor
    y           : the target for the current datapoint          // tensor
    y_pred      : the predictions returned by the model         // tensor
    los         : the calculated loss for the current datapoint // nn loss object

    Returns:
    trn_los_avg : the averaged training loss                    // float
    trn_acc_avg : the averaged training accuracy                // float
    '''

    mdl.train() # set model to training mode
    trn_los, trn_acc = 0, 0 # initialise training loss and accuracy metrics
    for X, y in ldr:
        X, y = X.to(device), y.to(device)
        opti.zero_grad() # set optimiser to zero grad for training
        y_pred = mdl(X) # return model predictions
        los = crit(y_pred, y.unsqueeze(1)) # return loss from criterion function
        los.backward() # implement backpropagation
        trn_los += los*X.size(0)
        trn_acc += evs(y.cpu().numpy(), y_pred.detach().cpu().numpy())*X.size(0) # use explained_variance_score for accuracy
        opti.step() # step the optimiser
        
    return trn_los/len(ldr.dataset), trn_acc/len(ldr.dataset) # take averages for the loss and accuracy

def val(mdl, crit, ldr):
    '''
    This routine handles the validation loop

    Arguments:
    mdl         : the model to be validated                     // nn.Module
    crit        : the criterion (loss) function                 // nn loss function
    ldr         : the dataloader for validation                 // dataloader

    Parameters:
    val_los     : tracks the validation loss                    // float
    val_acc     : tracks the validation accuracy using evs      // float
    X           : the feature set for the current datapoint     // tensor
    y           : the target for the current datapoint          // tensor
    y_pred      : the predictions returned by the model         // tensor
    los         : the calculated loss for the current datapoint // nn loss object

    Returns:
    val_los_avg : the averaged validation loss                  // float
    val_acc_avg : the averaged validation accuracy              // float
    '''

    mdl.eval() # set model to evaluation mode for validation
    val_los, val_acc = 0, 0 # initialise validation loss and accuracy metrics
    for X, y in ldr:
        with torch.no_grad(): # set torch to not calculate gradient for validation
            X, y = X.to(device), y.to(device)
            y_pred = mdl(X) # return model predictions
            los = crit(y_pred, y.unsqueeze(1)) # return loss from criterion function
            val_los += los*X.size(0)
            val_acc += evs(y.unsqueeze(1).cpu().numpy(), y_pred.cpu().numpy())*X.size(0) # use explained_variance_score for accuracy
            
    return val_los/len(ldr.dataset), val_acc/len(ldr.dataset) # take averages for the loss and accuracy

def execute(model, n_epochs, trn_ldr, val_ldr, opti, crit, plot):
    '''
    This routine is responsible for the entire training process, and handles in-training plotting

    Arguments:
    model       : the model to be trained                                   // nn.Module
    n_epochs    : the number of epochs the model should be trained for      // integer
    trn_ldr     : the training dataloader                                   // dataloader
    val_ldr     : the validation dataloader                                 // dataloader
    opti        : the optimiser object                                      // optim
    crit        : the criterion (loss) function                             // nn loss function
    plot        : a flag denoting whether in-training plotting should occur // boolean

    Parameters:
    liveloss    : responsible for in-training plotting, activated by plot   // PlotLosses() object
    epoch       : the current epoch number                                  // integer
    logs        : holds the log data for the current epoch                  // dict
    trn_los     : the training loss for the current epoch                   // float
    trn_acc     : the training accuracy for the current epoch               // float
    val_los     : the validation loss for the current epoch                 // float
    val_acc     : the validation accuracy for the current epoch             // float

    Returns:
    model       : the final, trained model                                  // nn.Module
    '''

    if plot:
        liveloss = PlotLosses() # initialise liveloss if plotting flag true

    for epoch in range(n_epochs):
        logs = {}

        trn_los, trn_acc = trn(model, opti, crit, trn_ldr) # run the training cycle
        logs['' + 'log loss'] = trn_los.item()
        logs['' + 'accuracy'] = trn_acc.item() # update the logs
        
        val_los, val_acc = val(model, crit, val_ldr) # run the validation cycle
        logs['val_' + 'log loss'] = val_los.item()
        logs['val_' + 'accuracy'] = val_acc.item() # update the logs
        
        if plot:
            liveloss.update(logs)
            liveloss.draw() # print the plots if flag is true
        if not plot:
            print("Epoch: " + str(epoch)) # if not plotting, print epoch number for tracking
    
    return model # return finished trained model

def output(model, f_tst, t_tst):
    '''
    This routine handles the error analysis and output

    Arguments:
    model       : the trained model to be analysed                  // nn.Module
    f_tst       : the features for testing                          // tensor
    t_tst       : the targets for testing                           // tensor

    Parameters:
    output      : the output from the model when given f_tst        // tensor
    truth       : the true values for f_tst, from t_tst             // tensor
    errors      : a list of the absolute errors for each datapoint  // [float]
    outputs     : a list version of output                          // [float]
    truths      : a list version of truth                           // [float]
    avg_error   : the average absolute error                        // float
    avg_value   : the average of the absolute targets               // float
    max_error   : the largest absolute error in the dataset         // float
    bad_index   : the index for the max_error datapoint             // integer
    datalen     : the length of the dataset, for averaging          // integer
    i           : indexing for the loop                             // integer
    error       : the absolute error for the current datapoint      // float
    mini        : the minimum value of both truths and outputs      // float
    maxi        : the maximum value of both truths and outputs      // float

    Returns:
    null
    '''

    model.eval() # set model to evaluation mode
    output = model(f_tst) # return trained model's predictions
    truth = t_tst # return the base truth values for the features given to the model

    errors = []
    outputs = []
    truths = []

    avg_error = 0
    avg_value = 0
    max_error = 0
    bad_index = 0 # initialising metrics

    datalen = len(truth) # number of datapoints in evaluation set

    for i in range(datalen):
        outputs.append(output[i].item())
        truths.append(truth[i].item()) # converts tensor dataset to simple list

        error = abs(truths[i] - outputs[i]) # calculates absolute error for each datapoint
        errors.append(error)
        avg_error += error
        
        avg_value += abs(truths[i]) # takes the absolute value for average so average doesnt go to zero

        if error > max_error:
            max_error = error # ascertains the worst misprediction
            bad_index = i
            
    avg_error /= datalen
    avg_value /= datalen # calculates averages

    truths, outputs = (list(t) for t in zip(*sorted(zip(truths, outputs)))) # sorts lists for plotting

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

    errors = sorted(errors)
    plt.hist(errors[:-round(0.005*datalen)], 50) # exclude the worst 0.5% of errors as they ruin the plot
    plt.axvline(avg_value, color="r", linestyle="dashed", label="Average Value : " + str(avg_value)[:4])
    plt.plot([], [], ' ', label="Average Error : " + str(avg_error)[:4])
    plt.plot([], [], ' ', label="Percent Error : " + str(100*avg_error/avg_value)[:4] + "%")
    plt.title("Error Histogram")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    plt.plot(outputs, truths, '+')
    mini = min(np.minimum(outputs, truths))
    maxi = max(np.maximum(outputs, truths)) # calculates the range for the 1:1 line
    plt.plot([mini, maxi], [mini, maxi])
    plt.title("Prediction against Truth")
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.show()

    plt.plot(outputs, '+', label="Prediction")
    plt.plot(truths, label="Truth")
    plt.title("SIF - Prediction and Truth")
    plt.xlabel("Datapoint")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def main(file_name, mdl=3, n_epochs=500, sif=0, plot=True, test=True, save=False, mID=1, load=False, ld_mdl=[3,1,1]):
    '''
    This is the core routine that handles everything else

    Arguments:
    file_name       : the file address for the data csv                              // string
    mdl             : the ID for the model to be used                                // integer      // default: 3
    n_epochs        : the number of epochs to run training for                       // integer      // default: 500
    sif             : which stress intensity factor element to be predicted     // integer      // default: 0
    plot            : flag that determines whether to do mid-training plotting      // boolean      // default: True
    test            : flag that determines whether to evaluate the trained model // boolean      // default: True
    save            : flag that determines whether to save the trained model         // boolean      // default: False
    mID             : model ID to be affixed to the saved model's filename      // to-string    // default: 1
    load            : flag that determines whether to train or load a model         // boolean      // default: False
    ld_mdl          : contains the required data for loading a model                 // list         // default: [3,1,1]
    
    Parameters:
    trn_batch_size  : the batch size for training                                   // integer
    tst_batch_size  : the batch size for testing                                    // integer
    model           : the selected model architecture                               // nn.Module
    opti            : the optimiser function                                        // torch.optim object
    crit            : the criterion function                                        // nn object
    ICGT_tips       : the custom dataset that holds the input                       // Dataset
    trn_ldr         : the dataloader for training                                   // dataloader
    val_ldr         : the dataloader for validation                                 // dataloader
    tst_ldr         : the dataloader for testing [vestigial]                        // dataloader
    f_tst           : input features for testing                                    // tensor
    t_tst           : targets for testing                                           // tensor
    cancel          : flag set to true if load operation fails                      // boolean

    Returns:
    null
    '''

    if not load: # this is the training routine
        lr = 1e-3
        trn_batch_size = 1000
        tst_batch_size = 1000

        if (sif == 1) or (sif == 2) or (sif == 3): # SIF selection routine
            ICGT_tips = TipDataset(file_name, sif)
        else:
            print("No SIF element selected, defaulting to SIF I")
            ICGT_tips = TipDataset(file_name, 1)
            sif = 1 # sets SIF id to 1 after defaulting

        if mdl == 1: # model selection statements
            model = FirstNet(ICGT_tips.ndat)
        elif mdl == 2:
            model = SecondNet(ICGT_tips.ndat)
        elif mdl == 3:
            model = ThirdNet(ICGT_tips.ndat)
        else:
            print("Invalid model selected, defaulting to Model 3")
            model = ThirdNet(ICGT_tips.ndat)
            mdl = 3 # sets model id to 3 after defaulting
        
        #if torch.cuda.device_count() > 1:
            #model = nn.DataParallel(model)
        model.to(device)

        #opti = torch.optim.SGD(model.parameters(), lr=lr)
        #opti = torch.optim.RMSprop(model.parameters())
        opti = torch.optim.Adam(model.parameters()) # optimiser definition
        #crit = nn.MSELoss()
        crit = nn.L1Loss()
        #crit = nn.SmoothL1Loss() # criterion definition

        trn_ldr, val_ldr, tst_ldr, f_tst, t_tst = dataldr_make(ICGT_tips, trn_batch_size, tst_batch_size)

        model = execute(model, n_epochs, trn_ldr, val_ldr, opti, crit, plot) # actually trains the model

        if save: # if the save flag is true, saves the model dict
            torch.save(model.state_dict(), "Models/model" + str(mdl) + "_SIF" + str(sif) + "_" + str(mID) + ".pt")

    elif load: # loads a previously trained model for testing

        cancel = False # if loading fails, set to true to avoid failed evaluation
        test = True

        mdl = ld_mdl[0]
        sif = ld_mdl[1]
        mID = ld_mdl[2] # decodes loading id

        if mdl == 1:
            model = FirstNet(7)
        elif mdl == 2:
            model = SecondNet(7)
        elif mdl == 3:
            model = ThirdNet(7)
        else:
            print("Invalid model, cancelling load operation")
            cancel = True
            test = False # sets flags to false to avoid evaluation without a load

        try: # tries the load operation
            model.load_state_dict(torch.load("Models/model" + str(mdl) + "_SIF" + str(sif) + "_" + str(mID) + ".pt"))
        except:
            print("Load operation failed, cancelling operation")
            cancel = True
            test = False # sets flags to false to avoid evaluation without a load
        
        if not cancel:
            ICGT_tips = TipDataset(file_name, sif)
            f_tst = ICGT_tips.data.float()
            t_tst = ICGT_tips.lbls.float() # assigns values for output routine

    if test:
        output(model, f_tst, t_tst)

main("Data/fracture_k_set.csv", mdl=3, n_epochs=1000, sif=1, plot=False, test=False, save=True, mID = "res1", load=True, ld_mdl=[3,1,"res1"])

main("Data/fracture_k_set.csv", mdl=3, n_epochs=1000, sif=2, plot=False, test=False, save=True, mID = "res1", load=True, ld_mdl=[3,2,"res1"])

main("Data/fracture_k_set.csv", mdl=3, n_epochs=1000, sif=3, plot=False, test=False, save=True, mID = "res1", load=True, ld_mdl=[3,3,"res1"])


#%%
