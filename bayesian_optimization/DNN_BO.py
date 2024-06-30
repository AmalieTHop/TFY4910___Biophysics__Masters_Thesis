# import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import os
import copy
import warnings

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import algorithms.utils as from_utils


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction = 'none')
        self.eps = eps
        self.rmse = None
        
    def forward(self, yhat, y, reduction='mean'):
        se = self.mse(yhat,y)                            # sqared error
        self.rmse = torch.sqrt(torch.mean(se, dim=-1))   # rms error
        if reduction=='none':
            return self.rmse
        elif reduction=='mean':
            mean_rmse = torch.mean(self.rmse)            # mean rms error for batch
            return mean_rmse
    


# Define the neural network.
class Net(nn.Module):
    def __init__(self, bvalues, net_pars, parameters):

        super(Net, self).__init__()
        self.net_pars = net_pars
        self.bvalues = bvalues
        self.hparams = parameters

        # define number of parameters being estimated
        self.est_pars = 4

        # define width of network
        width_hidden = self.hparams.get("width")

        # define module lists
        self.fc_layers0 = nn.ModuleList()
        self.fc_layers1 = nn.ModuleList()
        self.fc_layers2 = nn.ModuleList()
        self.fc_layers3 = nn.ModuleList()

        # loop over the layers
        width_input = len(bvalues)
        for i in range(self.hparams.get("depth")):
            # extend with a fully-connected linear layer
            self.fc_layers0.extend([nn.Linear(width_input, width_hidden)])
            self.fc_layers1.extend([nn.Linear(width_input, width_hidden)])
            self.fc_layers2.extend([nn.Linear(width_input, width_hidden)])
            self.fc_layers3.extend([nn.Linear(width_input, width_hidden)])
            width_input = width_hidden
            # add batch normalisation
            self.fc_layers0.extend([nn.BatchNorm1d(width_hidden)])
            self.fc_layers1.extend([nn.BatchNorm1d(width_hidden)])
            self.fc_layers2.extend([nn.BatchNorm1d(width_hidden)])
            self.fc_layers3.extend([nn.BatchNorm1d(width_hidden)])

            # add ELU units for non-linearity
            self.fc_layers0.extend([nn.ELU()])
            self.fc_layers1.extend([nn.ELU()])
            self.fc_layers2.extend([nn.ELU()])
            self.fc_layers3.extend([nn.ELU()])

            # add dropout regularisation
            dropout_p = self.hparams.get("dropout_p")
            if (i != (self.hparams.get("depth") - 1)):
                self.fc_layers0.extend([nn.Dropout(dropout_p)])
                self.fc_layers1.extend([nn.Dropout(dropout_p)])
                self.fc_layers2.extend([nn.Dropout(dropout_p)])
                self.fc_layers3.extend([nn.Dropout(dropout_p)])
        
        # final layer yielding output
        self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(width_hidden, 1))
        self.encoder1 = nn.Sequential(*self.fc_layers1, nn.Linear(width_hidden, 1))
        self.encoder2 = nn.Sequential(*self.fc_layers2, nn.Linear(width_hidden, 1))
        self.encoder3 = nn.Sequential(*self.fc_layers3, nn.Linear(width_hidden, 1))


    def forward(self, X):
        params0 = self.encoder0(X)
        params1 = self.encoder1(X)
        params2 = self.encoder2(X)
        params3 = self.encoder3(X)

        X_temp=[]
        const = 1.0
        constraint = self.hparams.get("constraint")
        if constraint == 'relu6':
            output_activation = nn.ReLU6()
            const = 6.0
        elif constraint == 'sigmoid':
            output_activation = nn.Sigmoid()
        else:
            raise Exception('the chosen parameter constraint is not implemented. Try ''relu6'' or ''sigmoid''.')
        
        Dt_norm = output_activation(params0[:, 0].unsqueeze(1))/const
        Fp_norm = output_activation(params1[:, 0].unsqueeze(1))/const
        Dp_norm = output_activation(params2[:, 0].unsqueeze(1))/const
        S0_norm = output_activation(params3[:, 0].unsqueeze(1))/const


        [Dt_unorm, Fp_unorm, Dp_unorm, S0_unorm] = from_utils.unormalise_params([Dt_norm, Fp_norm, Dp_norm, S0_norm], self.net_pars.bounds)

        X_temp.append(S0_unorm * (Fp_unorm * torch.exp(-self.bvalues * Dp_unorm) + (1 - Fp_unorm) * torch.exp(-self.bvalues * Dt_unorm)))
        X = torch.cat(X_temp, dim=1)

        params_norm = torch.hstack((Dt_norm, Fp_norm, Dp_norm, S0_norm))
        params_unorm = torch.hstack((Dt_unorm, Fp_unorm, Dp_unorm, S0_unorm))
        return X, params_unorm, params_norm



def learn_selfsupervised(X, bvalues, arg, parameters):

    torch.backends.cudnn.benchmark = True
    arg = checkarg(arg)

    # normalise the signal to b=0
    X = normalise(X, bvalues, arg, min(bvalues))

    # initialising the network of choice using the input argument arg
    bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
    net = Net(bvalues, arg.net_pars, parameters).to(arg.train_pars.device)

    # splitting data into learning and validation set; subsequently initialising the Dataloaders
    X_train, X_validation = train_test_split(X, test_size = 1 - arg.train_pars.split, random_state=64)

    # train loader loads the training data and the validation loader loads the validation loader
    trainloader = create_loader_from_np(X_train, labels=False, batch_size=arg.train_pars.batch_size, shuffle=True)
    validationloader = create_loader_from_np(X_validation, labels=False, batch_size=(min(len(X_validation), 32*arg.train_pars.batch_size)), shuffle=False)

    # number of iterations en each epoch for training and validation
    num_training_its = np.min([arg.train_pars.maxit, len(X_train)// arg.train_pars.batch_size])
    num_validation_its = len(X_validation) // (min(len(X_validation), 32*arg.train_pars.batch_size))

    # defining optimiser
    optimizer = load_optimizer(net, arg, parameters)

    # define loss function
    criterion = define_loss_fun(arg, parameters.get("loss_fun"))


    # Initialising parameters
    best_validation_loss = 1e16
    num_bad_epochs = 0
    avg_epoch_train_losses = []
    avg_epoch_validation_losses = []
    final_model = copy.deepcopy(net.state_dict())

    ## Train
    for epoch in range(5000):
        print("-----------------------------------------------------------------")
        print(f"Epoch: {epoch}; Bad epochs: {num_bad_epochs}")
        net.train()

        train_loss_vals = np.zeros(num_training_its)
        for i, [X_train] in enumerate(tqdm(trainloader, position=0, leave=True, total=num_training_its), 1):
            # have a maximum number of batches per epoch to ensure regular updates of whether we are improving
            if i > num_training_its:
                break

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # put batch on GPU if pressent
            X_train = X_train.to(arg.train_pars.device)

            ## forwardprop
            X_pred, _, _ = net(X_train)
            
            # determine loss for batch; note that the loss is determined by the difference between the predicted signal and the actual signal. The loss does not look at Dt, Dp or Fp.
            train_loss = criterion(X_pred, X_train)
            train_loss_vals[i-1] = train_loss

            # backprop + updating network
            train_loss.backward()
            optimizer.step()

        avg_epoch_train_loss = np.mean(train_loss_vals)
        avg_epoch_train_losses.append(avg_epoch_train_loss)
        print(f'train loss: {avg_epoch_train_loss}')


        # Validation
        net.eval()
        validation_loss_vals = np.zeros(num_validation_its)
        for i, [X_validation] in enumerate(tqdm(validationloader, position=0, leave=True), 1):
            # zero the parameter gradients (from previous training)
            optimizer.zero_grad()

            # put batch on GPU if pressent
            X_validation = X_validation.to(arg.train_pars.device)
            
            # do prediction, only look at predicted IVIM signal
            X_pred, _, _ = net(X_validation)

            # validation loss
            validation_loss = criterion(X_pred, X_validation)
            validation_loss_vals[i-1] = validation_loss

        avg_epoch_validation_loss = np.mean(validation_loss_vals)
        avg_epoch_validation_losses.append(avg_epoch_validation_loss)
        print(f'validation loss: {avg_epoch_validation_loss}')

        if arg.train_pars.select_best:
            if avg_epoch_validation_loss < best_validation_loss:
                print('\n############### Saving good model ###############################')
                final_model = copy.deepcopy(net.state_dict())
                best_validation_loss = avg_epoch_validation_loss
                num_bad_epochs = 0
            else:
                num_bad_epochs = num_bad_epochs + 1
                if num_bad_epochs == arg.train_pars.patience:
                    print(f'\nDone, best validation loss: {best_validation_loss}')
                    break

    print("Done")
    
    # Restore best model
    if arg.train_pars.select_best:
        net.load_state_dict(final_model)
    del trainloader
    del validationloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()


    return net, avg_epoch_train_losses, avg_epoch_validation_losses, best_validation_loss



def learn_supervised(X, y, bvalues, arg, parameters):

    torch.backends.cudnn.benchmark = True
    arg = checkarg(arg)

    # normalise the signal to b=0
    X = normalise(X, bvalues, arg, min(bvalues))

    # initialising the network
    bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
    net = Net(bvalues, arg.net_pars, parameters).to(arg.train_pars.device)

    # splitting data into learning and validation set
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 1-arg.train_pars.split, random_state=64)    ###

    # train loader loads the training data and the validation loader loads the validation loader
    trainloader = create_loader_from_np(X_train, y_train, labels=True, batch_size=arg.train_pars.batch_size, shuffle=True)
    validationloader = create_loader_from_np(X_validation, y_validation, labels=True, batch_size=(min(len(X_validation), 32*arg.train_pars.batch_size)), shuffle=False)
    
    # number of iterations en each epoch for training and validation
    num_training_its = np.min([arg.train_pars.maxit, len(X_train)// arg.train_pars.batch_size])
    num_validation_its = len(X_validation) // (min(len(X_validation), 32*arg.train_pars.batch_size))

    # defining optimiser
    optimizer = load_optimizer(net, arg, parameters)

    # define loss function
    criterion = define_loss_fun(arg, parameters.get("loss_fun"))


    # Initialising parameters
    best_validation_loss = 1e16
    num_bad_epochs = 0
    avg_epoch_train_losses = []
    avg_epoch_validation_losses = []
    final_model = copy.deepcopy(net.state_dict())

    ## Train
    for epoch in range(5000):
        print('-----------------------------------------------------------------')
        print(f'Epoch: {epoch}; Bad epochs: {num_bad_epochs}')
        net.train()

        train_loss_vals = np.zeros(num_training_its)
        for i, [X_train, y_train] in enumerate(tqdm(trainloader, position=0, leave=False, total=num_training_its), 1):
            # have a maximum number of batches per epoch to ensure regular updates of whether we are improving
            if i > num_training_its:
                break

            # zero the parameter gradients
            optimizer.zero_grad()

            # put batch on GPU if pressent
            X_train = X_train.to(arg.train_pars.device)
            y_train = y_train.to(arg.train_pars.device)

            # forwardprop
            _, _, params_pred_norm = net(X_train)

            # loss
            train_loss = criterion(params_pred_norm, y_train)
            train_loss_vals[i-1] = train_loss

            # backprop + updating network
            train_loss.backward()
            optimizer.step()

        avg_epoch_train_loss = np.mean(train_loss_vals)
        avg_epoch_train_losses.append(avg_epoch_train_loss)
        print(f'train loss: {avg_epoch_train_loss}')


        # Validation
        net.eval()
        validation_loss_vals = np.zeros(num_validation_its)
        for i, [X_validation, y_validation] in enumerate(tqdm(validationloader, position=0, leave=False), 1):
            # zero the parameter gradients (from previous training)
            optimizer.zero_grad()

            # put batch on GPU if pressent
            X_validation = X_validation.to(arg.train_pars.device)

            # forward
            _, _, params_pred_norm = net(X_validation)

            # validation
            validation_loss = criterion(params_pred_norm, y_validation)
            validation_loss_vals[i-1] = validation_loss
        
        avg_epoch_validation_loss = np.mean(validation_loss_vals)
        avg_epoch_validation_losses.append(avg_epoch_validation_loss)
        print(f'validation loss: {avg_epoch_validation_loss}')

        if arg.train_pars.select_best:
            if avg_epoch_validation_loss < best_validation_loss:
                print('\n############### Saving good model ###############################')
                final_model = copy.deepcopy(net.state_dict())
                best_validation_loss = avg_epoch_validation_loss
                num_bad_epochs = 0
            else:
                num_bad_epochs = num_bad_epochs + 1
                if num_bad_epochs == arg.train_pars.patience:
                    print(f'\nDone, best validation loss: {best_validation_loss}')
                    break

    print('Done')
    
    # Restore best model
    if arg.train_pars.select_best:
        net.load_state_dict(final_model)
    del trainloader
    del validationloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()

    return net, avg_epoch_train_losses, avg_epoch_validation_losses, best_validation_loss




def predict_IVIM(data, bvalues, net, arg, parameters):
    arg = checkarg(arg)

    ## normalise the signal to b=0 and remove data with nans
    data = normalise(data, bvalues, arg)
    mylist = isnan(np.mean(data, axis=1))
    sels = [not i for i in mylist]
    # remove data with non-IVIM-like behaviour. Estimating IVIM parameters in these data is meaningless anyways.
    sels = sels & (np.percentile(data[:, bvalues < 50], 0.95, axis=1) < 1.3) & (
                   np.percentile(data[:, bvalues > 50], 0.95, axis=1) < 1.2) & (
                   np.percentile(data[:, bvalues > 150], 0.95, axis=1) < 1.0)
    
    # we need this for later
    lend = len(data)
    data = data[sels]

    # tell net it is used for evaluation
    net.eval()
    # initialise parameters and data
    Dp = np.array([])
    Dt = np.array([])
    Fp = np.array([])
    S0 = np.array([])
    measured_signal_rmse = np.array([])

    # defining the loss function; signal-RMSE
    rmse_criterion = RMSELoss().to(arg.train_pars.device)

    # initialise dataloader. Batch size can be way larger as we are still training.
    testloader = create_loader_from_np(data, labels=False, batch_size=min(len(data), arg.train_pars.batch_size*16), shuffle=False, drop_last=False)
    num_test_its = int(np.ceil(len(data) / (min(len(data), arg.train_pars.batch_size*16))))
    
    # start predicting
    with torch.no_grad():
        test_loss_vals = np.zeros(num_test_its)
        for i, [X_test] in enumerate(tqdm(testloader, position=0, leave=True), 1):
            # put batch on GPU if pressent
            X_test = X_test.to(arg.train_pars.device)

            # forward   
            X_test_pred, params_unorm_pred, _ = net(X_test)

            # scaled predicted parameters and signal-rmse
            Dt_unorm = params_unorm_pred[:, 0].cpu().numpy()
            Fp_unorm = params_unorm_pred[:, 1].cpu().numpy()
            Dp_unorm = params_unorm_pred[:, 2].cpu().numpy()
            S0_unorm = params_unorm_pred[:, 3].cpu().numpy()
            rmses = rmse_criterion(X_test_pred, X_test, reduction='none').cpu().numpy()

            # signal loss
            test_loss = np.mean(rmses)
            test_loss_vals[i-1] = test_loss

            try:
                S0 = np.append(S0, S0_unorm)
            except:
                S0 = np.append(S0, S0_unorm)
            Dt = np.append(Dt, Dt_unorm)
            Fp = np.append(Fp, Fp_unorm)
            Dp = np.append(Dp, Dp_unorm)
            measured_signal_rmse = np.append(measured_signal_rmse, rmses)
    avg_test_loss = np.mean(test_loss_vals)

    # The 'abs' and 'none' constraint networks have no way of figuring out what is D and D* a-priori. However, they do
    # tend to pick one output parameter for D or D* consistently within the network. If the network has swapped D and
    # D*, we swap them back here.
    if np.mean(Dp) < np.mean(Dt):
        Dp_temp = copy.deepcopy(Dt)
        Dt = copy.deepcopy(Dp)
        Dp = copy.deepcopy(Dp_temp)
        Fp = 1 - Fp
    
    # correcting for the data that initially was removed as it did not have IVIM behaviour, by returning zero estimates
    Dptrue = np.zeros(lend)
    Dttrue = np.zeros(lend)
    Fptrue = np.zeros(lend)
    S0true = np.zeros(lend)
    measured_signal_rmse_true = np.zeros(lend)
    Dptrue[sels] = Dp
    Dttrue[sels] = Dt
    Fptrue[sels] = Fp
    S0true[sels] = S0
    measured_signal_rmse_true[sels] = measured_signal_rmse
    
    del testloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()

    return [Dttrue, Fptrue, Dptrue, S0true, measured_signal_rmse_true], avg_test_loss





def create_loader_from_np(X, y = None, labels = True, batch_size = 128, shuffle= True, drop_last = True):

    if labels:
        dataset = utils.TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.float))
    else:
        dataset = utils.TensorDataset(torch.from_numpy(X).type(torch.float))
    
    loader = utils.DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle, 
                              drop_last=drop_last)
    
    return loader



def load_optimizer(net, arg, parameters):
    optimizer_name = parameters.get("optimizer")
    lr = parameters.get("lr")
    if optimizer_name == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-4)
        return optimizer
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-4)
        return optimizer
    else:
        print('Invalid optimizer: choose adam or adamw')



def define_loss_fun(arg, loss_fun):
    if loss_fun == 'rmse':
        criterion = RMSELoss().to(arg.train_pars.device) 
        return criterion
    elif loss_fun == 'mse':
        criterion = nn.MSELoss().to(arg.train_pars.device)
        return criterion
    elif loss_fun == 'mae':
        criterion = nn.L1Loss().to(arg.train_pars.device)
        return criterion
    else:
        print('Invalid loss function: choose mse, rmse or mae')



def normalise(X_train, bvalues, arg, bref=0):
    try:
        ## normalise the signal to b=0 and remove data with nans
        if arg.norm_data_full:
            S0 = np.mean(X_train, axis=1).astype('<f')
        else:
            S0 = np.mean(X_train[:, bvalues == bref], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        np.delete(X_train, isnan(np.mean(X_train, axis=1)), axis=0)
        # normalise neighbours
    except:
        S0 = torch.mean(X_train[:, bvalues == bref], axis=1)
        X_train = X_train / S0[:, None]
        np.delete(X_train, isnan(torch.mean(X_train, axis=1)), axis=0)
    return X_train


def isnan(x):
    # this program indicates what are NaNs 
    return x != x




def checkarg_train_pars(arg):
    if not hasattr(arg, 'patience'):
        warnings.warn('arg.train.patience not defined. Using default value 10')
        arg.patience = 10
    if not hasattr(arg,'batch_size'):
        warnings.warn('arg.train.batch_size not defined. Using default value 128')
        arg.batch_size = 128
    if not hasattr(arg,'maxit'):
        warnings.warn('arg.train.maxit not defined. Using default value 500')
        arg.maxit = 500
    if not hasattr(arg,'split'):
        warnings.warn('arg.train.split not defined. Using default value 0.9')
        arg.split = 0.9
    if not hasattr(arg,'use_cuda'):
        arg.use_cuda = torch.cuda.is_available()
    if not hasattr(arg, 'device'):
        arg.device = torch.device("cuda:0" if arg.use_cuda else "cpu")
    if not hasattr(arg, 'select_best'):
        warnings.warn('arg.train.select_best not defined. Using default of True')
        arg.select_best = True
    return arg


def checkarg_net_pars(arg):
    if not hasattr(arg,'bounds'):
        warnings.warn('arg.net_pars.bounds not defined. Using default values [0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]')
        arg.bounds = np.array([[0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]])
    return arg


def checkarg(arg):
    if not hasattr(arg,'net_pars'):
        warnings.warn('arg no net_pars. Using default initialisation')
        arg.net_pars = net_pars()
    if not hasattr(arg, 'train_pars'):
        warnings.warn('arg no train_pars. Using default initialisation')
        arg.train_pars = train_pars()
    if not hasattr(arg, 'norm_data_full'):
        warnings.warn('arg no norm_data_full. Using default of False')
        arg.norm_data_full = False
    arg.net_pars = checkarg_net_pars(arg.net_pars)
    arg.train_pars = checkarg_train_pars(arg.train_pars)
    return arg



class train_pars:
    def __init__(self):
        self.patience = 10
        self.batch_size = 128
        self.maxit = 500
        self.split = 0.9
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = True


class net_pars:
    def __init__(self):
        self.bounds = np.array([[0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]])
