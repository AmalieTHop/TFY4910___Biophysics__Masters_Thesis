"""
September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Built on code by Sebastiano Barbieri: https://github.com/sebbarb/deep_ivim

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved physics-informed deep learning of the intravoxel-incoherent motion model: accurate, unique and consistent. MRM 2021)
If this code was useful, please cite:
https://doi.org/10.1002/mrm.27910
"""

"""
Modified:
June 2024 by Amalie Toftum Hop
https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis

Code is uploaded as part of a Master’s thesis: 
Amalie Toftum Hop. “Deep Learning-Based Intravoxel Incoherent Motion Modelling of
Diffusion-Weighted MRI in Head and Neck Cancer: In Silico and In Vivo Studies.
Master thesis. Norwegian University of Science and Technology, 2024.



Comment: This source code is similar to algorithms/DNN.py, but is adapted
for model selection using bayesian optimisation as described in the thesis. 
The parameter 'parameters' is used to handle the different hyperparameters 
that are being tuned.
"""


# import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import warnings

from sklearn.model_selection import train_test_split
import algorithms.utils as from_utils
import algorithms.DNN.DNN as from_DNN



# define the neural network.
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

        # output activation function
        X_temp=[]
        const = 1.0
        constraint = self.hparams.get("constraint")
        if constraint == 'relu6':
            output_activation = nn.ReLU6()
            const = 6.0
        elif constraint == 'sigmoid':
            output_activation = nn.Sigmoid()
        else:
            raise Exception('The chosen parameter constraint is not implemented. Try ''relu6'' or ''sigmoid''.')
        
        # normalised ivim parameter outputs
        Dt_norm = output_activation(params0[:, 0].unsqueeze(1))/const
        Fp_norm = output_activation(params1[:, 0].unsqueeze(1))/const
        Dp_norm = output_activation(params2[:, 0].unsqueeze(1))/const
        S0_norm = output_activation(params3[:, 0].unsqueeze(1))/const

        # scaled ivim parameter outputs
        [Dt_unorm, Fp_unorm, Dp_unorm, S0_unorm] = from_utils.unormalise_params([Dt_norm, Fp_norm, Dp_norm, S0_norm], self.net_pars.bounds)

        # uses the ivim model to recondtruct the ivim signal sequences
        X_temp.append(S0_unorm * (Fp_unorm * torch.exp(-self.bvalues * Dp_unorm) + (1 - Fp_unorm) * torch.exp(-self.bvalues * Dt_unorm)))
        
        X = torch.cat(X_temp, dim=1)
        params_norm = torch.hstack((Dt_norm, Fp_norm, Dp_norm, S0_norm))
        params_unorm = torch.hstack((Dt_unorm, Fp_unorm, Dp_unorm, S0_unorm))
        return X, params_unorm, params_norm



def learn_selfsupervised(X, bvalues, arg, parameters):
    """
    Trains self-supervised DNN with hyperparameters given by 'arg' using the training data 'X'.
    """

    torch.backends.cudnn.benchmark = True
    arg = checkarg_BO(arg)

    # normalise to S(b=0)
    X, _ = from_DNN.normalise(X, bvalues, min(bvalues))

    # initialising the network of choice using the input argument arg
    bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
    net = Net(bvalues, arg.net_pars, parameters).to(arg.train_pars.device)

    # splitting data into learning and validation set; subsequently initialising the dataloaders
    X_train, X_validation = train_test_split(X, test_size = 1 - arg.train_pars.split, random_state=64)

    # train loader loads the training data and the validation loader loads the validation loader
    trainloader = from_DNN.create_loader_from_np(X_train, labels=False, batch_size=arg.train_pars.batch_size, shuffle=True)
    validationloader = from_DNN.create_loader_from_np(X_validation, labels=False, batch_size=(min(len(X_validation), 32*arg.train_pars.batch_size)), shuffle=False)

    # number of iterations in each epoch for training and validation
    num_training_its = np.min([arg.train_pars.maxit, len(X_train)// arg.train_pars.batch_size])
    num_validation_its = len(X_validation) // (min(len(X_validation), 32*arg.train_pars.batch_size))

    # defining optimiser
    optimizer = load_optimizer(net, parameters)

    # define loss function
    criterion = define_loss_fun(arg, parameters)


    # initialising parameters
    best_validation_loss = 1e16
    num_bad_epochs = 0
    avg_epoch_train_losses = []
    avg_epoch_validation_losses = []
    final_model = copy.deepcopy(net.state_dict())

    # training
    for epoch in range(5000):
        print('-----------------------------------------------------------------')
        print(f'Epoch: {epoch}; Bad epochs: {num_bad_epochs}')
        net.train()

        train_loss_vals = np.zeros(num_training_its)
        for i, [X_train] in enumerate(tqdm(trainloader, position=0, leave=True, total=num_training_its), 1):

            # keeps track of the number of iterations per epoch
            if i > num_training_its:
                break

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # put batch on GPU if present
            X_train = X_train.to(arg.train_pars.device)

            ## forwardprop
            X_pred, _, _ = net(X_train)
            
            # determine training loss for batch; note that the loss is determined by the difference between the predicted signal and the true signal. The loss does not look at Dt, Dp or Fp.
            train_loss = criterion(X_pred, X_train)
            train_loss_vals[i-1] = train_loss

            # backprop + updating network
            train_loss.backward()
            optimizer.step()

        # training loss for epoch
        avg_epoch_train_loss = np.mean(train_loss_vals)
        avg_epoch_train_losses.append(avg_epoch_train_loss)
        print(f'train loss: {avg_epoch_train_loss}')


        # validation
        net.eval()
        validation_loss_vals = np.zeros(num_validation_its)
        for i, [X_validation] in enumerate(tqdm(validationloader, position=0, leave=True), 1):

            # zero the parameter gradients (from previous training)
            optimizer.zero_grad()

            # put batch on GPU if present
            X_validation = X_validation.to(arg.train_pars.device)
            
            # do prediction, only look at predicted IVIM signal
            X_pred, _, _ = net(X_validation)

            # determine validation loss for batch
            validation_loss = criterion(X_pred, X_validation)
            validation_loss_vals[i-1] = validation_loss

        # validation loss for epoch
        avg_epoch_validation_loss = np.mean(validation_loss_vals)
        avg_epoch_validation_losses.append(avg_epoch_validation_loss)
        print(f'validation loss: {avg_epoch_validation_loss}')

        # early stopping
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

    
    # restore best model
    if arg.train_pars.select_best:
        net.load_state_dict(final_model)

    del trainloader
    del validationloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()

    print('Sucessfull training')
    return net, avg_epoch_train_losses, avg_epoch_validation_losses, best_validation_loss





def learn_supervised(X, y, bvalues, arg, parameters):
    """
    Trains supervised DNN with hyperparameters given by 'arg' using the training inputs 'X' and its 
    corresponding training lables 'y'.
    """

    torch.backends.cudnn.benchmark = True
    arg = checkarg_BO(arg)

    # normalise to S(b=0)
    X, _ = from_DNN.normalise(X, bvalues, min(bvalues))

    # initialising the network
    bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
    net = Net(bvalues, arg.net_pars, parameters).to(arg.train_pars.device)

    # splitting data into learning and validation set
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 1-arg.train_pars.split, random_state=64)    ###

    # train loader loads the training data and the validation loader loads the validation loader
    trainloader = from_DNN.create_loader_from_np(X_train, y_train, labels=True, batch_size=arg.train_pars.batch_size, shuffle=True)
    validationloader = from_DNN.create_loader_from_np(X_validation, y_validation, labels=True, batch_size=(min(len(X_validation), 32*arg.train_pars.batch_size)), shuffle=False)
    
    # number of iterations en each epoch for training and validation
    num_training_its = np.min([arg.train_pars.maxit, len(X_train)// arg.train_pars.batch_size])
    num_validation_its = len(X_validation) // (min(len(X_validation), 32*arg.train_pars.batch_size))

    # defining optimiser
    optimizer = load_optimizer(net, parameters)

    # define loss function
    criterion = define_loss_fun(arg, parameters)


    # initialising parameters
    best_validation_loss = 1e16
    num_bad_epochs = 0
    avg_epoch_train_losses = []
    avg_epoch_validation_losses = []
    final_model = copy.deepcopy(net.state_dict())

    # training
    for epoch in range(5000):
        print('-----------------------------------------------------------------')
        print(f'Epoch: {epoch}; Bad epochs: {num_bad_epochs}')
        net.train()

        train_loss_vals = np.zeros(num_training_its)
        for i, [X_train, y_train] in enumerate(tqdm(trainloader, position=0, leave=False, total=num_training_its), 1):

            # keeps track of the number of iterations per epoch
            if i > num_training_its:
                break

            # zero the parameter gradients (from previous training)
            optimizer.zero_grad()

            # put batch on GPU if present
            X_train = X_train.to(arg.train_pars.device)
            y_train = y_train.to(arg.train_pars.device)

            # forwardprop
            _, _, params_pred_norm = net(X_train)

            # determine training loss for batch; note that the loss is determined by the difference between the predicted ivim parameters and the true ivim parameters
            train_loss = criterion(params_pred_norm, y_train)
            train_loss_vals[i-1] = train_loss

            # backprop + updating network
            train_loss.backward()
            optimizer.step()

        # training loss for epoch
        avg_epoch_train_loss = np.mean(train_loss_vals)
        avg_epoch_train_losses.append(avg_epoch_train_loss)
        print(f'train loss: {avg_epoch_train_loss}')


        # validation
        net.eval()
        validation_loss_vals = np.zeros(num_validation_its)
        for i, [X_validation, y_validation] in enumerate(tqdm(validationloader, position=0, leave=False), 1):
            
            # zero the parameter gradients (from previous training)
            optimizer.zero_grad()

            # put batch on GPU if present
            X_validation = X_validation.to(arg.train_pars.device)

            # forward
            _, _, params_pred_norm = net(X_validation)

            # determine validation loss for batch
            validation_loss = criterion(params_pred_norm, y_validation)
            validation_loss_vals[i-1] = validation_loss
        
        # validation loss for epoch
        avg_epoch_validation_loss = np.mean(validation_loss_vals)
        avg_epoch_validation_losses.append(avg_epoch_validation_loss)
        print(f'validation loss: {avg_epoch_validation_loss}')

        # early stopping
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
    
    # restore best model
    if arg.train_pars.select_best:
        net.load_state_dict(final_model)

    del trainloader
    del validationloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()

    print('Sucessfull training')
    return net, avg_epoch_train_losses, avg_epoch_validation_losses, best_validation_loss






# utilities to DNN

def load_optimizer(net, parameters):
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



def define_loss_fun(arg, parameters):
    loss_fun = parameters.get("loss_fun")
    print(loss_fun)
    if loss_fun == 'rmse':
        criterion = from_utils.RMSELoss().to(arg.train_pars.device) 
        return criterion
    elif loss_fun == 'mse':
        criterion = nn.MSELoss().to(arg.train_pars.device)
        return criterion
    elif loss_fun == 'mae':
        criterion = nn.L1Loss().to(arg.train_pars.device)
        return criterion
    else:
        print('Invalid loss function: choose mse, rmse or mae')





# check that all arguments are given

def checkarg_train_pars_BO(arg):
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


def checkarg_net_pars_BO(arg):
    if not hasattr(arg,'bounds'):
        warnings.warn('arg.net_pars.bounds not defined. Using default values [0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]')
        arg.bounds = np.array([[0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]])
    return arg


def checkarg_BO(arg):
    if not hasattr(arg,'net_pars'):
        warnings.warn('arg no net_pars. Using default initialisation')
        arg.net_pars = net_pars_BO()
    if not hasattr(arg, 'train_pars'):
        warnings.warn('arg no train_pars. Using default initialisation')
        arg.train_pars = train_pars_BO()
    arg.net_pars = checkarg_net_pars_BO(arg.net_pars)
    arg.train_pars = checkarg_train_pars_BO(arg.train_pars)
    return arg



class train_pars_BO:
    def __init__(self):
        self.patience = 10
        self.batch_size = 128
        self.maxit = 500
        self.split = 0.9
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = True


class net_pars_BO:
    def __init__(self):
        self.bounds = np.array([[0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]])
