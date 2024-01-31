"""
Project: 
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
Provides a unified file for training pytorch models built with the ann.py file.
For examples of how to use this file see ann_tester.py or gan_tester.py.

Functions:
 - get_losses(names=False)
 - get_optimizers(names=False)
 - plot_history(history, multiplot=False, save=False, save_path=False)
 - test_helper(model, x, y, threshold, confusion)
 - train_helper (model, x, y, loss_func, threshold, training=True)

DNN
 - attribute_get(name)
 - attribute_set(name, value)
 - test(data, headers=None, threshold=0.5)
 - train(train_data, valid_data, num_epochs, shuffle=False, threshold=0.5, headers=None, validation=True, verbose=0)

GAN
 - attribute_get(purpose, name)
 - attribute_set(name, value)
 - discriminator_set_input_shape(batch_size, features_shape, purpose='train')
 - generator_set_input_shape(batch_size, features_shape, purpose='train')
 - test(test, threshold=0.5, classifier=False, load_gan=None, prop_gan=False)
 - train(num_epochs, train, threshold=0.5, generator_class=None, classifier=False, headers=None, save=False, save_path='../Data/generated/', overwrite=False, shuffle=False, verbose=0)

Included with: 
 - ann.py 
 - ann_tester.py
 - common.py 
 - data_generator.py
 - dataloader.py
 - gan_tester.py
 - labeler.py
 - metrics.py
 - plotter.py
 - preprocessor.py 
 - stats.py
 - trainer.py (current file)
 - transitioner.py
 
Notes:
For more information about the project contact 
 - Dr. Suresh Muknahallipatna -> sureshm@uwyo.edu
 - Josh Blaney -> jblaney1@uwyo.edu
"""

# Outside Dependencies
import copy
import glob
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In House Dependencies
import common

__name__ = 'trainer'


"""
    get_losses(names=False)
    A function to return either the dictionary of available losses from pytorch
    or the names of the available losses from pytorch. Used by the training
    methods to link the loss function to the model based solely on the name of 
    the loss function.
    
    inputs:
     - names (bool): Return a list of loss names?
    outptus:
     - losses (list/dict): The available loss functions
"""
def get_losses(names=False):
    try:
        losses = {'l1loss':torch.nn.L1Loss,
                  'mseloss':torch.nn.MSELoss,
                  'crossentropyloss':torch.nn.CrossEntropyLoss,
                  'ctcloss':torch.nn.CTCLoss,
                  'nllloss':torch.nn.NLLLoss,
                  'poissonnllloss':torch.nn.PoissonNLLLoss,
                  'gaussiannllloss':torch.nn.GaussianNLLLoss,
                  'kldivloss':torch.nn.KLDivLoss,
                  'bceloss':torch.nn.BCELoss,
                  'bcewithlogitsloss':torch.nn.BCEWithLogitsLoss,
                  'marginrankingloss':torch.nn.MarginRankingLoss,
                  'hingeembeddingloss':torch.nn.HingeEmbeddingLoss,
                  'multilabelmarginloss':torch.nn.MultiLabelMarginLoss,
                  'huberloss':torch.nn.HuberLoss,
                  'smoothl1loss':torch.nn.SmoothL1Loss,
                  'softmarginloss':torch.nn.SoftMarginLoss,
                  'multilabelsoftmarginloss':torch.nn.MultiLabelSoftMarginLoss,
                  'cosineembeddingloss':torch.nn.CosineEmbeddingLoss,
                  'multimarginloss':torch.nn.MultiMarginLoss,
                  'tripletmarginloss':torch.nn.TripletMarginLoss,
                  'tribletmarginwithdistanceloss':torch.nn.TripletMarginWithDistanceLoss}
        
        losses = list(losses.keys()) if names else losses
        return losses
    
    except Exception as e:
        common.Print_Error('Trainer -> get losses', e)
    
    
"""
    get_optimizers(names=False)
    A function to return either the dictionary of available optimizers from 
    pytorch or the names of the available optimizers from pytorch. Used by the 
    training methods to link the optimizer to the model based solely on the name 
    of the optimizer.
    
    inputs:
     - names (bool): Return a list of optimizer names?
    outptus:
     - optimizers (list/dict): The available optimizers
"""
def get_optimizers(names=False):
    try:
        optimizers = {'adadelta':torch.optim.Adadelta,
                      'adagrad':torch.optim.Adagrad,
                      'adam':torch.optim.Adam,
                      'adamw':torch.optim.AdamW,
                      'sparseadam':torch.optim.SparseAdam,
                      'adamax':torch.optim.Adamax,
                      'asgd':torch.optim.ASGD,
                      'lbfgs':torch.optim.LBFGS,
                      'nadam':torch.optim.NAdam,
                      'radam':torch.optim.RAdam,
                      'rmsprop':torch.optim.RMSprop,
                      'rprop':torch.optim.Rprop,
                      'sgd':torch.optim.SGD}
    
        optimizers = list(optimizers.keys()) if names else optimizers
        return optimizers
    
    except Exception as e:
        common.Print_Error('Trainer -> get optimizers', e)


"""
     plot_history(history, multiplot=False, save=False, save_path=None)
     A function to automate the plotting of training history. Either all tracked
     parameters can be plotted on one graph (multiplot=True) or on separate 
     graphs (multiplot=False).
     
 inputs:
     - history (dictionary): The training history with strings as keys
     - multiplot (bool): Should each parameter get its own plot?
     - save (bool): Save the plot?
     - save_path (string): Where to save the plot
 outputs:
     - 
 """
def plot_history(history, multiplot=False, save=False, save_path=None):        
    try:
        key = list(history.keys())[0]
        x = range(len(history[key]))
 
        if multiplot:
            for key in history.keys():
                y = history[key]
                fig, ax = plt.subplots()
                ax.plot(x,y)
                ax.grid(True)
                ax.set_ylabel(key)
                ax.set_xlabel('Epochs')
        
        else:
            fig, ax = plt.subplots()
            for key in history.keys():
                y = history[key]
                ax.plot(x, y, label=key)
            
            ax.grid(True)
            ax.set_ylabel('History')
            ax.set_xlabel('Epochs')
            ax.legend()
    
        if save:
            plt.savefig(fname=save_path+'history.png')

    except Exception as e:
        common.Print_Error('Trainer -> plot history', e)

      
"""
    test_helper(model, x, y, threshold, confusion)
    A function to automate most of the testing sequence including generating predictions
    and computing accuracy.

    inputs:
     - model (pytorch model): The model to make predictions with
     - x (tensor): A batch of inputs
     - y (tensor): A batch of labels
     - threshold (float): The classification threshold for computing accuracy
     - confusion (bool): Create a confusion matrix?
    outputs:
     - 
"""
def test_helper(model, x, y, threshold, confusion):
    try:
        pred = torch.reshape(model(x), y.shape)

        if pred.shape[-1] > 1 and len(pred.shape) > 1:
            is_correct = (torch.argmax(pred,dim=1) == torch.argmax(y,dim=1)).float()
        else:
            is_correct = ((pred>threshold).int() == y.int()).float()

        acc = is_correct.sum().item()
        if confusion:
            confusion_matrix = common.Update_Confusion_Matrix(pred.cpu(), y.cpu())
            return acc, confusion_matrix
        else:
            return acc
    
    except Exception as e:
        common.Print_Error('Trainer -> test helper', e)
        
    
"""
    train_helper (model, x, y, loss_func, threshold)
    A function to streamline the majority of the training loop. This 
    function uses the model to produce an output, computes the loss, and 
    updates the optimizer.
    
    inputs:
     - model (pytorch model): The model to use for predictions
     - x (tensor): A batch of inputs
     - y (tensor): A batch of labels
     - loss_func (pytorch function): The loss function to use for backprop
     - threshold (float): The classification threshold for computing accuracy
     - training (bool): Is this in a training step? Backprop?
    outputs:
     - ls (float): Loss between predictions and labels
     - acc (float): The accuracy
"""
def train_helper(model, x, y, loss_func, threshold, training=True):
    try:
        # Predict our class based on the current state of our model 
        pred = torch.reshape(model(x), y.shape)

        # Compute the loss
        loss = loss_func(pred, y) 

        # Backpropagate our loss  
        if training:
            loss.backward()

        # Calculate the accuracy of the prediction
        if pred.shape[-1] > 1 and len(pred.shape) > 1:
            is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)).float()

        else:
            is_correct = ((pred > threshold).int() == y.int()).float()

        return loss.item()*y.shape[0], is_correct.sum().item()

    except Exception as e:
        common.Print_Error('Trainer -> train helper', e)
        
        
"""
    DNN_Trainer
    A class to hold all of the necessary training functions for Classification
    and Regression. This class is designed to hold a single pytorch model for 
    training, built by the ann.py file. For examples of how to use this class
    see ann_test.py. See the GAN class below for an example of training more 
    complicated architectures.
"""
class DNN():
    def __init__(self, ann):
        self.ann = ann                              # Store the input model, from ann.py, locally
        self.report = ann.report                    # Store the input model report locally
        
        self.history = {}                           # A dictionary to save the training results
        self.attribute_set('confusion matrix', None)    # An empty confusion matrix used during testing
    
        
    """
        attribute_get(purpose, name)
        A function to retrieve values stored in the report
        
        inputs:
            - name (string): The key to retrieve the information from
        outputs:
            -
    """
    def attribute_get(self,name):
        try:
            return self.report[name]
        except Exception as e:
            common.Print_Error('DNN -> attribute get', e)
    
    
    """
        attribute_set(name, value)
        A function to update the report entry specified by name with value.
        
        inputs:
            - name (string): The entry to update
            - value (): The new value to store at the entry
        outputs:
            -
    """
    def attribute_set(self, name, value):
        try:
            self.report[name] = value
        except Exception as e:
            common.Print_Error('DNN -> attribute set', e)


    """
         test(path, headers=None, threshold=0.5)
         A function to automate the testing process for ML models. The accuracy
         is the only tracked metric and it is stored in a history dictionary.
         
         inputs:
          - data (tuple): A tuple with [features, labels] of batched data
          - headers (list): The headers to use when loading the dataset
          - threshold (float): The classification threshold for computing accuracy
         outptus:
          - acc (float): Accuracy on the test set
    """
    def test(self, data, headers=None, threshold=0.5):
        try:            
            # Init local variables
            acc = 0.0
            labels = data[1].float()
            features = data[0].float()
            batch_size = features.shape[1]
            test_batches = features.shape[0]
            num_inputs = batch_size * test_batches

            binary = True if len(labels.shape) <= 2 else labels.shape[-1] < 2
            confusion_shape = (2,2) if binary else (labels.shape[-1], labels.shape[-1])
            confusion_matrix = np.zeros(confusion_shape)

            # Transfer models to the specified device
            device = torch.device(self.report['device'])
            model = self.ann.model.to(device)

            # Set the trained models for evaluation
            model.eval()
       
            # Explicitly disable gradient computation
            with torch.no_grad():
                for i in range(test_batches):
                    # Transfer data to the same device as the models
                    y_batch = labels[i].to(device)
                    x_batch = features[i].to(device)

                    # Compute the testing results for this batch of real data
                    temp_acc, temp_confusion_matrix = test_helper(model, x_batch, y_batch, threshold, confusion=True)

                    # Store the testing results locally
                    acc += temp_acc
                    confusion_matrix += temp_confusion_matrix

            acc /= num_inputs

            return acc, confusion_matrix

        except Exception as e:
            common.Print_Error('ANN -> test', e)
                
                
    """
        train(path, num_epochs, shuffle=False, threshold=0.5, headers=None, validation=True, verbose=0)
        A function to train DNN. Specifically a function to train individual regression of classificaiton
        models on one set of training and validation datasets.
        
        inputs:
         - train_data (np array): The training dataset
         - valid_data (np array): The validation dataset
         - num_epochs (int): The max number of epochs to train for
         - shuffle (bool): Shuffle the training data each epoch?
         - threshold (float): The binary classification threshold to calculate accuracy
         - headers (list): The headers to use when loading the dataset
         - validation (bool): Should a validation dataset be loaded
         - verbose (int): How much information should be printed?
        outputs:
         - history (dictionary of lists): A dictionary of training metrics
    """
    def train(self, train_data, valid_data, num_epochs, shuffle=False, threshold=0.5, headers=None, validation=True, verbose=0):
        try:            
            # Init local variables
            early_stop_loss = 1e9
            early_stop_index = 0
            train_labels = train_data[1].float()
            train_features = train_data[0].float()
            valid_labels = valid_data[1].float() if valid_data is not None else None
            valid_features = valid_data[0].float() if valid_data is not None else None

            batch_size = train_features.shape[1]
            num_batches_train = train_features.shape[0]
            num_batches_valid = valid_labels.shape[0] if valid_data is not None else 0
            max_index = num_batches_train if num_batches_valid == 0 else num_batches_train - 1
            input_count_train = batch_size * num_batches_train
            input_count_valid = batch_size * num_batches_valid
            
            # Establish dictionaries for training metrics and validation metrics
            history_train = {'loss': [], 'accuracy': [], 'loss length': [], 'accuracy length': []}
            history_valid = {'validation loss': [], 'validation accuracy': [], 'validation loss length': [], 'validation accuracy length': []}
            
            # Move the model to the device and finish model setup (loss and optimizer)
            device = torch.device(self.report['device'])
            model = self.ann.model.to(device)
            weights = None if self.report['weight'] is None else torch.tensor(self.report['weight']) 

            if self.report['loss'] == 'bcewithlogitsloss':
                loss_func = get_losses(names=False)[self.report['loss']]() if weights is None else get_losses(names=False)[self.report['loss']](pos_weight=weights.to(device)) 
            else:
                loss_func = get_losses(names=False)[self.report['loss']]() if weights is None else get_losses(names=False)[self.report['loss']](weight=weights.to(device))

            self.report['lr'] = self.report['lr'] if self.report['lr'] is not None else 0.001
            optimizer = get_optimizers(names=False)[self.report['optimizer']](model.parameters(), lr=self.report['lr'])

            key_width = 5 if verbose < 2 else None
            percent_width = 0 if verbose < 2 else 20
            common.Print('[INFO] Starting Training Process')           
            # Initiate training for the planned number of epochs
            for epoch in range(num_epochs):
                # Evaluate earlystopping criteria
                if self.report['patience'] > 0 and num_batches_valid > 0 and epoch > 0:
                    if history_valid['validation loss'][-1] < early_stop_loss:
                        best_model = copy.deepcopy(model)
                        early_stop_loss = history_valid['validation loss'][-1]
                        early_stop_index = 0

                    else:
                        early_stop_loss = early_stop_loss
                        early_stop_index += 1

                    if early_stop_index >= self.report['patience']:
                        model = copy.deepcopy(best_model)
                        break

                if shuffle:
                    train_features, train_labels = common.Shuffle(train_features, train_labels)

                # Set the current history entry to zero
                for key in history_train.keys():
                    history_train[key].append(0.0)
                    
                ls = 0.0
                acc = 0.0
                index = 0
                # Iterate through our batches (housed within our training DataLoader object)
                for i in range(num_batches_train):
                    # Ensure the model is in training mode even if there is no valdaition set
                    model.train()

                    # Load a batch of data and transfer it to the device
                    y_batch = train_labels[i].to(device)
                    x_batch = train_features[i].to(device)

                    # Zero the optimizer gradients
                    optimizer.zero_grad()
                    
                    # Train the model using a helper function 
                    ls, acc = train_helper(model, x_batch, y_batch, loss_func, threshold)
                    optimizer.step()

                    # Record the training results
                    history_train['loss'][epoch] += ls
                    history_train['accuracy'][epoch] += acc

                    # Track the number of examples processed
                    for key in history_train.keys():
                        if 'length' in key:
                            history_train[f'{key}'][epoch] += len(x_batch)

                    # Update the status printout
                    if index < max_index:
                        common.Print_Status(f'Epoch {epoch} Training', index, num_batches_train, history_train, key_width, percent_width)

                    index += 1

                # Process the validation data
                if num_batches_valid > 0:
                    # Set model to evaluation mode to stop processing weight updates
                    model.eval()
                    
                    # Initialize this epochs metrics in the history dictionary
                    for key in history_valid.keys():
                        history_valid[key].append(0.0)
                    
                    ls = 0.0
                    acc = 0.0
                    index = 0
                    
                    # Process each batch of the validation data
                    for j in range(num_batches_valid):
                        # Transfer the data to the same device as the model
                        y_batch = valid_labels[j].to(device)
                        x_batch = valid_features[j].to(device)        
                        
                        # Use the helper function to test the model
                        ls, acc = train_helper(model, x_batch, y_batch, loss_func, threshold, training=False)
                        
                        # Save the metrics in the history dictionary
                        history_valid['validation loss'][epoch] += ls
                        history_valid['validation accuracy'][epoch] += acc
                        
                        # Track the number of samples processed
                        for key in history_valid.keys():
                            if 'length' in key:
                                history_valid[f'{key}'][epoch] += len(x_batch)

                        # Update status printout
                        if index < num_batches_valid-1:
                            common.Print_Status(f'Epoch {epoch} Validation', index, num_batches_valid, history_valid, key_width, percent_width)
                            
                        index += 1

                    # Update total training printout
                    common.Print_Status(f'Epoch {epoch}', num_batches_train, num_batches_train, {**history_train, **history_valid}, key_width, percent_width)
                                    
                # Normalize training metrics post epoch
                for key in history_train.keys():
                    history_train[key][epoch] /= input_count_train
                
                # Normalize validation metrics post epoch
                if num_batches_valid > 0:
                    for key in history_valid.keys():
                        history_valid[key][epoch] /= input_count_valid 
            
            # Combine training and validation metrics into one dictionary
            temp = {**history_train, **history_valid} if num_batches_valid > 0 else history_train
            history = {}

            for key in temp.keys():
                if 'length' not in key:
                    history[key] = temp[key]
 
            # Transfer model and history back to ann object
            self.ann.model = copy.deepcopy(model.cpu())
            self.ann.history = history
                
            return history
        
        except Exception as e:
            common.Print_Error('ANN -> train',e)
                
        
"""
    GAN
    A trainer class which contains the necessary functionality to train and test
    gan models which have been created using the ANN file. The python file
    gan_tester.py implements several gan using this class.
"""
class GAN():
    def __init__(self, gan):
        self.gan = gan              # A gan from the ann.py file.
        self.report = gan.report    # The report of the gan for synchronization purposes
        
    
    """
        attribute_get(purpose, name)
        A function to retrieve values stored in the report
        
        inputs:
            - name (string): The key to retrieve the information from
        outputs:
            -
    """
    def attribute_get(self, purpose, name):
        try:
            return self.report[name]
        except Exception as e:
            common.Print_Error('DNN -> attribute get', e)
    
    
    """
        attribute_set(name, value)
        A function to update the report entry specified by name with value.
        
        inputs:
            - name (string): The entry to update
            - value (): The new value to store at the entry
        outputs:
            -
    """
    def attribute_set(self, name, value):
        try:
            self.report[name] = value
        except Exception as e:
            common.Print_Error('DNN -> attribute set', e)
            

    """
        discriminator_set_input_shape(batch_size, features_shape, purpose='train')
        A function to set the expected input shape for the generator
        based on the type of discriminator layers used.

        inputs:
         - batch_size (int): The size of one batch
         - features_shape (tuple): The shape of the inputs features
         - purpose (string): dictionary key modifier for train and testing differentiation
        outputs:
         -
    """
    def discriminator_set_input_shape(self, batch_size, features_shape, purpose='train'):
        if self.report['discriminator type'] == 1: # LINEAR DISCRIMINATOR
            self.report['discriminator input shape ' + purpose] = (batch_size, int(self.report['discriminator inputs']))
            input_dim = 1

        elif self.report['discriminator type'] == 2: # RECURRENT DISCRIMINATOR                
            self.report['discriminator input shape ' + purpose] = (batch_size, features_shape[-2], int(self.report['discriminator inputs']))
            input_dim = 2

        elif self.report['discriminator type'] == 3: # CONVOLUTION DISCRIMINATOR
            self.report['discriminator input shape ' + purpose] = (batch_size, int(self.report['discriminator inputs']), features_shape[-1])
            input_dim = 1

        return input_dim
            
            
    """
        generator_set_input_shape(batch_size, features_shape, purpose='train')
        A function to set the expected input shape for the generator
        based on the type of generator layers used.

        inputs:
         - batch_size (int): The size of one batch
         - features_shape (tuple): The shape of the input features
         - purpose (string): dictionary key modifier for train and testing differentiation
        outputs:
         - generator_sequence_length (int): The expected sequence length of the generator inputs
    """
    def generator_set_input_shape(self, batch_size, features_shape, purpose='train'):
        if self.report['generator type'] == 1: # LINEAR GENERATOR
            self.report['generator input shape ' + purpose] = (batch_size, int(self.report['generator inputs']))
            generator_sequence_length = 0

        elif self.report['generator type'] == 2: # RECURRENT GENERATOR
            self.report['generator sequence length'] = features_shape[-1]
            generator_sequence_length = self.report['generator sequence length']
            self.report['generator input shape ' + purpose] = (batch_size, generator_sequence_length, int(self.report['generator inputs']))

        elif self.report['generator type'] == 3: # CONVOLUTION GENERATOR
            self.report['generator sequence length'] = features_shape[-2] if self.report['discriminator type'] == 2 else features_shape[-1]
            generator_sequence_length = self.report['generator sequence length']
            self.report['generator input shape ' + purpose] = (batch_size, int(self.report['generator inputs']), generator_sequence_length)

        return generator_sequence_length
    
    
    """
        test(test, threshold, classifier)
        A function to automate the testing process for ML models. The accuracy
        is the only tracked metric and it is stored in a history dictionary.
        
        inputs:
         - test (dataset): The dataset to test the model with
         - threshold (float): The classification threshold for computing accuracy
         - classifier (bool): Is the discriminator performing multi-class classification?
         - load_gan (string): For two input discriminators use a pretrained gan for feature. 
         - prop_gan (bool): Should the synthetic data propagate through the gans? 
        outputs:
         - acc (float): Accuracy on the test set
    """
    def test(self, test, threshold=0.5, classifier=False, load_gan=None, prop_gan=False):
        try:
            # Init local variables
            acc = 0.0
            num_inputs = 0
            labels = test[1].float()
            features = test[0].float()
            batch_size = features.shape[1]
            test_batches = features.shape[0] 

            binary = True if len(labels.shape) == 1 else labels.shape[-1] < 2
            num_classes = 1 if binary else labels.shape[-1]
            confusion_shape = (2,2) if binary else (labels.shape[-1], labels.shape[-1])
            confusion_matrix = np.zeros(confusion_shape)

            input_dim = self.discriminator_set_input_shape(batch_size, features.shape, purpose='test')
            self.generator_set_input_shape(batch_size, features.shape, purpose='test') 
            generator_output_shape = list(self.report['discriminator input shape test'])

            # Transfer models to the specified device
            device = torch.device(self.report['device'])
            gene_models = []

            for i, model in enumerate(self.gan.generators):
                gene_models.append(model.to(device))
                gene_models[i].eval()

            if len(gene_models) > 1:
                generator_output_shape[input_dim] = 1

            disc_model = self.gan.discriminator.to(device)    

            # Set the trained models for evaluation
            disc_model.eval()

            # Explicitly disable gradient computation
            with torch.no_grad():
                for i in range(test_batches):

                    # Transfer data to the same device as the models
                    y_batch = labels[i].to(device)
                    x_batch = features[i].to(device)

                    num_inputs += y_batch.shape[0]

                    # Compute the testing results for this batch of real data
                    temp_acc, temp_confusion_matrix = test_helper(disc_model, x_batch, y_batch, threshold, confusion=True)

                    # Store the testing results locally
                    acc += temp_acc
                    confusion_matrix += temp_confusion_matrix

                    # Get a batch of random inputs for the generator
                    inputs = torch.normal(mean=0, std=1, size=self.report['generator input shape test']).to(device) 

                    # Generate a batch of SYNTHETIC data using the generator
                    gene_batch = gene_models[0](inputs).detach()

                    # Reshape the generator output to the shape the discriminator expects
                    if len(gene_models) > 1:
                        gene_batch = torch.reshape(gene_batch, generator_output_shape)

                        for i in range(1, len(gene_models)):
                            next_batch = gene_models[i](inputs).detach()
                            next_batch = torch.reshape(next_batch, generator_output_shape)
                            gene_batch = torch.cat((gene_batch, next_batch), dim=input_dim)
                    else:
                        gene_batch = torch.reshape(gene_batch, self.report['discriminator input shape test'])

                    # Automatically create the labels based on if the discriminator is performing
                    # multi-class or binary classification
                    if num_classes > 1: # multi-class
                        targets = torch.zeros([batch_size, num_classes], dtype=torch.float).to(device)
                        targets[:,-1] += 1
                    else: # binary
                        targets = torch.zeros([batch_size, 1], dtype=torch.float).to(device)

                    num_inputs += targets.shape[0]

                    # Compute the testing results for this batch of synthetic data
                    temp_acc, temp_confusion_matrix = test_helper(disc_model, gene_batch, targets, threshold, confusion=True)

                    # Store the testing results locally
                    acc += temp_acc
                    confusion_matrix += temp_confusion_matrix
                    
            acc /= num_inputs 

            return acc, confusion_matrix
        
        except Exception as e:
            common.Print_Error('GAN -> test', e)
    
    
    """
        train(num_epochs, train, threshold=0.5, generator_class=None, classifier=False, headers=None, save=False, save_path='../Data/generated/', overwrite=False, shuffle=False, verbose=0)
        A function to perform machine learning model training on GAN's. This function expects the the data
        to be shaped (num_batch, batch_size, ...) so that it can access the first dimension of the input
        dataset to grab a batch of data, this data format is the default output of the included dataloader
        file. Training is broken into three parts: first train the discriminator on real data, second train
        the discriminator on synthetic data, third train the generator. The forward pass and backprop have 
        been separated into a helper function (train_helper) so streamline this function.

        inputs:
         - num_epochs (int): The number of epochs to train for
         - train (tensor): The training dataset with features in the first dimension and labels in the second
         - threshold (float): The binary classification threshold to calculate accuracy with
         - generator_class (int): Desired class for the generator to emulate
         - classifier (bool): Is the discriminator performing multi-class classification?
         - headers (list): The headers which data is being generated for, used for saving
         - save (bool): Should the generator output be saved after each epoch
         - save_path (string): The path to the folder to save generator output to
         - overwrite (bool): Overwrite existing prediciton files?
         - shuffle (bool): Shuffle the training data each epoch?
         - verbose (int): How much information should be printed? [0,1,2]
        outputs:
         - history (dictionary of lists): A dictionary of training metrics
    """
    def train(self,  
              num_epochs, 
              train, 
              threshold=0.5, 
              generator_class=None, 
              classifier=False,
              headers=None, 
              save=False, 
              save_path='../Data/generated/', 
              overwrite=False, 
              shuffle=False, 
              verbose=0):
        
        try:
            # Init local variables
            temp = {}
            history = {}
            labels = train[1].float()
            features = train[0].float()
            num_classes = labels.shape[-1]
            inputs = int(self.report['discriminator inputs'])

            batch_size = features.shape[1]
            train_batches = features.shape[0]

            input_dim = self.discriminator_set_input_shape(batch_size, features.shape)
            generator_sequence_length = self.generator_set_input_shape(batch_size, features.shape)
            generator_output_shape = list(self.report['discriminator input shape train'])

            # Establish one loss function for both generator and discriminator
            loss_func = get_losses(names=False)[self.report['loss']]()

            # Transfer models to the specified device
            device = torch.device(self.report['device'])
            gene_models = []
            gene_optimizers = []

            for index, model in enumerate(self.gan.generators):
                gene_models.append(model.to(device))
                if self.gan.train[index]:
                   gene_models[index].train()
                   gene_optimizers.append(get_optimizers(names=False)[self.report['generator optimizer']](gene_models[index].parameters(), lr=self.report['generator lr']))
                else:
                   gene_models[index].eval()
                   gene_optimizers.append(None)

            if len(gene_models) > 1:
               generator_output_shape[input_dim] = 1

            disc_model = self.gan.discriminator.to(device)

            # Setup optimizers for transfered models
            self.report['generator lr'] = self.report['generator lr'] if self.report['generator lr'] is not None else 0.001
            self.report['discriminator lr'] = self.report['discriminator lr'] if self.report['discriminator lr'] is not None else 0.001
            disc_optimizer = get_optimizers(names=False)[self.report['discriminator optimizer']](disc_model.parameters(), lr=self.report['discriminator lr'])
            
            # Init history dictionary
            temp['discriminator loss'] = []
            temp['discriminator acc'] = []
            temp['generator loss'] = []
            
            for key in temp.keys():
                history[f'{key}'] = []
                history[f'{key} length'] = [] 
            
            # Remove existing files if overwriting
            if save and overwrite:
               save = True
               common.Remove_Dir(save_path)
               common.Validate_Dir(f'{save_path}')

            elif save:
               file_count = len(glob.glob(save_path))
               if file_count > 0:
                  message = '[WARNING] Unable to save generator output\n'
                  message += f'        {file_count} files already exists at specified location'
                  common.Print(message)
                  save = False
               else:
                  save = True

            # Display all parameters if the verbose is set above 1
            if verbose > 1: 
               generator_input_shape = self.report['generator input shape train']
               discriminator_input_shape = self.report['discriminator input shape train']

               message = '[INFO] Training parameters:\n'
               message += f'\tFeatures Shape: {features.shape}\n'
               message += f'\tLabels Shape: {labels.shape}\n'
               message += f'\tSize of Batches: {batch_size}\n'
               message += f'\tNumber of Batches: {train_batches}\n'
               message += f'\tGenerator Input Shape: {generator_input_shape}\n'
               message += f'\tGenerator Sequence Shape: {generator_sequence_length}\n'
               message += f'\tDiscriminator Input Shape: {discriminator_input_shape}\n'
               message += f'\tLoss Function: {loss_func}\n'
               message += f'Generator Optimizer: {gene_optimizers}\n'
               message += f'Discriminator Optimizer: {disc_optimizer}'
               common.Print(message)

               common.Print('[INFO] Starting Training Process')

            key_width = 5 if verbose < 2 else None
            percent_width = 0 if verbose < 2 else 20
            width = len(str(num_epochs)) + 1

            # Initiate training for the specified number of epochs
            for epoch in range(num_epochs):
                name = f'Training Epoch {epoch:>{width}}'

                # Expand history for the new epoch
                for key in history.keys():
                    history[key].append(0.0)

                # Reset local variables at the start of each epoch
                ls = 0.0
                acc = 0.0
                index = 0
                real_inputs = 0
                gene_inputs = 0
 
                if shuffle:
                    features, labels = common.Shuffle(features, labels)

                # Iterate through our batches (housed within our training DataLoader object)
                for i in range(train_batches):

                    # Zero the discriminator optimizer gradients
                    disc_optimizer.zero_grad()

                    real_inputs += labels[i].shape[0]

                    # Transfer the data to the same device as the models 
                    y_batch = labels[i].to(device)
                    x_batch = features[i].to(device)

                    # Train the discriminator on REAL data               
                    ls, acc = train_helper(disc_model, x_batch, y_batch, loss_func, threshold)

                    # Store the training results in the history
                    history['discriminator loss'][epoch] += ls
                    history['discriminator acc'][epoch] += acc
                    for key in history.keys():
                       if 'discriminator' in key and 'length' in key:
                           history[f'{key}'][epoch] += y_batch.shape[0]

                    # Get a batch of random inputs for the generator
                    inputs = torch.normal(mean=0, std=1, size=self.report['generator input shape train']).to(device)

                    # Generate a batch of SYNTHETIC data using the generator
                    gene_batch = gene_models[0](inputs).detach()

                    # Reshape the generator output to the shape the discriminator expects
                    if len(gene_models) > 1:
                        gene_batch = torch.reshape(gene_batch, generator_output_shape)

                        for i in range(1,len(gene_models)):
                            next_batch = gene_models[i](inputs).detach()
                            next_batch = torch.reshape(next_batch, generator_output_shape)
                            gene_batch = torch.cat((gene_batch, next_batch), dim=input_dim)
                    else:
                        gene_batch = torch.reshape(gene_batch, self.report['discriminator input shape train'])

                    # Automatically create the labels based on if the discriminator is performing
                    # multi-class or binary classification
                    if num_classes > 1: # multi-class
                        targets = torch.zeros([batch_size, num_classes], dtype=torch.float).to(device)
                        targets[:,-1] += 1
                    else: # binary
                        targets = torch.zeros([batch_size, 1], dtype=torch.float).to(device)

                    # Train the discriminator on the SYNTHETIC data
                    ls, acc = train_helper(disc_model, gene_batch, targets, loss_func, threshold)

                    # Update the discriminators weights
                    disc_optimizer.step()

                    # Store the training results in the history
                    history['discriminator loss'][epoch] += ls
                    history['discriminator acc'][epoch] += acc
                    for key in history.keys():
                        if 'discriminator' in key and 'length' in key:
                            history[f'{key}'][epoch] += targets.shape[0]

                    # Zero the generator optimizer gradients
                    for optimizer in gene_optimizers:
                        if optimizer is not None:
                            optimizer.zero_grad()

                    # Get a batch of random inputs for the generator
                    inputs = torch.normal(mean=0, std=1, size=self.report['generator input shape train']).to(device)

                    # Generate a batch of SYNTHETIC data using the generator
                    gene_batch = gene_models[0](inputs)

                    # Reshape the generator output to the shape the discriminator expects
                    if len(gene_models) > 1:
                        gene_batch = torch.reshape(gene_batch, generator_output_shape)

                        for i in range(1, len(gene_models)):
                            next_batch = gene_models[i](inputs)
                            next_batch = torch.reshape(next_batch, generator_output_shape)
                            gene_batch = torch.cat((gene_batch, next_batch), dim=input_dim)
                    else:
                        gene_batch = torch.reshape(gene_batch, self.report['discriminator input shape train'])

                    # Automatically create the labels based on if the discriminator is performing
                    # multi-class or binary classification
                    if num_classes > 1: # multi-class
                        targets = torch.zeros([batch_size, num_classes], dtype=torch.float).to(device) 
                        targets[:,generator_class] += 1
                    else: # binary
                        targets = torch.ones([batch_size, 1], dtype=torch.float).to(device)

                    gene_inputs += targets.shape[0]

                    # Train the generator using the discriminator output
                    ls, acc = train_helper(disc_model, gene_batch, targets, loss_func, threshold)

                    # Update the generator weights
                    for optimizer in gene_optimizers:
                        if optimizer is not None:
                           optimizer.step()

                    # Store the training results in the history 
                    history['generator loss'][epoch] += ls
                    for key in history.keys():
                        if 'generator' in key and 'length' in key:
                            history[f'{key}'][epoch] += targets.shape[0]

                    # Update the training status bar
                    if verbose > 0: common.Print_Status(name, index, train_batches, history, key_width, percent_width)
                    index += 1

                # Store the epoch results in the history
                history['discriminator loss'][epoch] /= (real_inputs + gene_inputs)
                history['discriminator acc'][epoch] /= (real_inputs + gene_inputs)
                history['generator loss'][epoch] /= gene_inputs

                # Optionally save an output from the generator
                if save:
                    preds = {}
                    path = f'{save_path}{epoch}.csv'
                    stable_input = 0.42 * torch.ones(self.report['generator input shape train'])
                    stable_input = stable_input.to(device)

                    for i, model in enumerate(gene_models):
                        preds[headers[i]] = model(stable_input).detach().cpu().numpy()[0,::]

                    preds['Label'] = [generator_class]*len(preds[headers[0]])

                    df = pd.DataFrame(preds)
                    df.to_csv(path, index=False)

            # Transfer the models back to the CPU
            for i, model in enumerate(gene_models):
                self.gan.generators[i] = copy.deepcopy(model.cpu())

            self.gan.discriminator = copy.deepcopy(disc_model.cpu())

            # Remove length information from history before output
            temp = {}
            for key in history.keys():
                if 'length' not in key:
                    temp[key] = history[key]
             
            return temp
        
        except Exception as e:
            common.Print_Error('GAN -> train', e)
    
    

