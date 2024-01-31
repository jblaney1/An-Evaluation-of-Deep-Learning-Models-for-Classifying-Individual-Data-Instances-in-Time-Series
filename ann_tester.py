"""
Project: 
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
Implements torch models through ann.py and trains them using train.py

Functions:
 - cnn_ann()
 - linear_ann()
 - rnn_ann()
 - test_model(dictionary, model)
 - train_model(dictionary, model)

Included with: 
 - ann.py 
 - ann_tester.py (current file)
 - common.py
 - data_generator.py
 - dataloader.py
 - gan_tester.py
 - labeler.py
 - metrics.py
 - plotter.py
 - preprocessor.py 
 - stats.py
 - trainer.py
 - transitioner.py
 
Notes:
For more information about the project contact 
 - Dr. Suresh Muknahallipatna -> sureshm@uwyo.edu
 - Josh Blaney -> jblaney1@uwyo.edu
"""

# Outside Dependencies
import torch
import random
import datetime
import numpy as np


# In House Dependencies
import ann
import common
import trainer
import dataloader as dl
        

"""
    cnn_ann(inputs, outputs, weights=None)
    Implements a cnn. Change the parameters directly in this function to 
    change the model. The input and output are automatically computed from 
    dictionary parameters are the bottom of this file.
    
inputs:
    - inputs (int): The number of input neurons
    - outputs (int): The number of output neurons
    - weights (list): The weights for the loss function
outputs:
    - model (uwyo dnn): A DNN object which combines a pytorch model and report dict
"""
def cnn_ann(inputs, outputs, weights=None):
    try:
        loss = 'crossentropyloss' if outputs > 2 else 'bcewithlogitsloss'
        optimizer = 'adam'

        dnn = ann.DNN(name='cnn', ann_type=3)
        dnn.model, report = ann.create_model(model_type='cnn',
                                             inputs=inputs, 
                                             outputs=outputs, 
                                             neurons=[512, 256, 128, 64, 32, 16], 
                                             activations=['leakyrelu', 'leakyrelu', 'softmax'], 
                                             linear_batch_normalization=True, 
                                             linear_dropout=0.50,
                                             cnn_type='1d', 
                                             channels=[128,],
                                             kernels=(3,), 
                                             strides=None, 
                                             paddings=None, 
                                             pooling=None, 
                                             pooling_kernel=None, 
                                             cnn_batch_normalization=True, 
                                             cnn_dropout=0.10, 
                                             cnn_sequence_length=dictionary['sequence length'])
        
        dnn.attribute_set('loss', loss)
        dnn.attribute_set('weight', weights)
        dnn.attribute_set('optimizer', optimizer)

        dnn.report = {**dnn.report, **report}

        return dnn
    except Exception as e:
        common.Print_Error('Test ANN -> CNN ANN', e)


"""
    linear_ann(inputs, outputs, weights=None)
    Implements a linear model. Change the parameters directly in this function
    to change the model. The input and output are automatically computed from 
    dictionary parameters are the bottom of this file.
    
inputs:
    - inputs (int): The number of input neurons
    - outputs (int): The number of output neurons
    - weights (list): The weights for the loss function
outputs:
    - model (uwyo dnn): A DNN object which combines a pytorch model and report dict
"""
def linear_ann(inputs, outputs, weights=None):
    try:
        loss = 'crossentropyloss' if outputs > 2 else 'bcewithlogitsloss'
        optimizer = 'rmsprop'

        dnn = ann.DNN('linear', ann_type=1)
        dnn.model, report = ann.create_model(model_type='linear', 
                                             inputs=inputs, 
                                             outputs=outputs, 
                                             neurons=[64, 16], 
                                             activations=['relu', 'sigmoid'], 
                                             linear_batch_normalization=True, 
                                             linear_dropout=0.1)
        
        dnn.attribute_set('loss', loss)
        dnn.attribute_set('weight', weights)
        dnn.attribute_set('optimizer', optimizer)
        
        dnn.report = {**dnn.report, **report}

        return dnn 
 
    except Exception as e:
        common.Print_Error('Test ANN -> Linear ANN', e)
        

"""
    rnn_ann(inputs, outputs, weights=None)
    Implements an rnn. Change the parameters directly in this funciton to 
    change the model. The input and output are automatically computed from 
    dictionary parameters are the bottom of this file.
    
inputs:
    - inputs (int): The number of input neurons
    - outputs (int): The number of output neurons
    - weights (list): The weights for the loss function
outputs:
    - model (uwyo dnn): A DNN object which combines a pytorch model and report dict
"""
def rnn_ann(inputs, outputs, weights=None):
    try:
        loss = 'crossentropyloss' if outputs > 2 else 'bcewithlogitsloss'
        optimizer = 'adam'

        dnn = ann.DNN('rnn', ann_type=0)
        dnn.model, report = ann.create_model(model_type='rnn', 
                                             inputs=inputs, 
                                             outputs=outputs, 
                                             neurons=[128, 64, 32, 16, 8, 4],
                                             activations=['leakyrelu', 'softmax'], 
                                             linear_batch_normalization=True, 
                                             linear_dropout=0.1,
                                             rnn_type='gru',
                                             hidden_size=128,
                                             num_layers=4, 
                                             bias=None, 
                                             batch_first=True, 
                                             rnn_dropout=0.25,
                                             bidirectional=False, 
                                             proj_size=None,
                                             recurrent_batch_normalization=True)

        dnn.attribute_set('loss', loss)
        dnn.attribute_set('weight', weights)
        dnn.attribute_set('optimizer', optimizer)
        
        dnn.report = {**dnn.report, **report}

        return dnn
 
    except Exception as e:
        common.Print_Error('Test ANN -> RNN ANN', e)
        
        
"""
    test_model(dictionary, model)
    Uses the trainer.py functions to test the provided ann.py model.
"""
def test_model(dictionary, model):
    try:
        model.attribute_set('data type', dictionary['data type'])
        model.attribute_set('tests limit', dictionary['tests limit'])

        report = model.report
        name = report['name']
        model_path = f'../Models/{name}/' if dictionary['model path'] is None else dictionary['model path']
        data_path = '../Data/Labeled/processed-low-high/' if dictionary['data path'] is None else dictionary['data path']

        desired_num_classes = 4 if report['restrict class'] == 5 else 2
        one_hot = report['outputs'] > 1 or report['restrict class'] > 5

        tests_data = dl.Preload_Dataset(data_path + '/testing/*.csv',
                                        ann_type=report['ann type'],
                                        data_type=report['data type'],
                                        batch_size=report['batch size'],
                                        mean=report['mean'],
                                        std=report['std'],
                                        sequence_length=report['sequence length'],
                                        limit=report['tests limit'],
                                        headers=report['headers'],
                                        restrict_class=report['restrict class'],
                                        shuffle=False,
                                        one_hot=one_hot,
                                        desired_num_classes=desired_num_classes)

        ann_trainer = trainer.DNN(ann=model)

        time_start = datetime.datetime.now()
        acc, confusion_matrix = ann_trainer.test(data=tests_data, threshold=dictionary['threshold'])

        time_stop = datetime.datetime.now()
        time_elapsed = time_stop - time_start
        message = f'[INFO] Testing started at: {time_start}\n'
        message += f'\tTesting completed at: {time_stop}\n'
        message += f'\tTesting Elapsed Time: {time_elapsed}\n'
        message += f'\tTesting Prediction Count: {int(sum(sum(confusion_matrix)))}\n'
        message += f'\tTesting Accuracy: {acc}\n'
        message += '\tTesting Confusion Matrix:\n'
        message += f'{common.Build_Confusion_Matrix_Output(confusion_matrix)}'
        common.Print(message)  

        model.report['testing accuracy'] = acc

        common.Save_Results(f'{model_path}results.txt', model.report, confusion_matrix)

    except Exception as e:
        common.Print_Error('Test ANN -> test model', e)
        
        
"""
    train_model(dictionary, model)
    Uses the trainer.py functions to train the provided ann.py model
"""
def train_model(dictionary, model):
    try:
        name = model.report['name']
        model.attribute_set('std', dictionary['std'])
        model.attribute_set('mean', dictionary['mean'])
        model.attribute_set('epochs', dictionary['epochs'])
        model.attribute_set('headers', dictionary['headers'])
        model.attribute_set('data type', dictionary['data type'])
        model.attribute_set('batch size', dictionary['batch size'])
        model.attribute_set('load shuffle', dictionary['load shuffle'])
        model.attribute_set('train shuffle', dictionary['train shuffle'])
        model.attribute_set('restrict class', dictionary['restrict class'])
        model.attribute_set('sequence length', dictionary['sequence length'])

        model.attribute_set('device', dictionary['device'])

        model.attribute_set('lr', dictionary['lr'])
        model.attribute_set('patience', dictionary['patience'])
 
        model.attribute_set('train limit', dictionary['train limit'])
        model.attribute_set('valid limit', dictionary['valid limit'])
        
        data_path = '../Data/Labeled/processed-low-high/' if dictionary['data path'] is None else dictionary['data path']
        model_path = f'../Models/{name}/' if dictionary['model path'] is None else dictionary['model path']
        
        report = model.report
        desired_num_classes = 4 if report['restrict class'] == 5 else 2
        one_hot = report['outputs'] > 1 or report['restrict class'] > 5
        
        train_data = dl.Preload_Dataset(data_path + '/training/*.csv',
                                        ann_type=report['ann type'],
                                        data_type=report['data type'],
                                        batch_size=report['batch size'],
                                        mean=report['mean'],
                                        std=report['std'],
                                        sequence_length=report['sequence length'],
                                        limit=report['train limit'],
                                        headers=report['headers'],
                                        restrict_class=report['restrict class'],
                                        shuffle=report['load shuffle'],
                                        one_hot=one_hot,
                                        desired_num_classes=desired_num_classes)
        
        if dictionary['valid']:
            valid_data = dl.Preload_Dataset(data_path + '/validation/*.csv',
                                            ann_type=report['ann type'],
                                            data_type=report['data type'],
                                            batch_size=report['batch size'],
                                            mean=report['mean'],
                                            std=report['std'],
                                            sequence_length=report['sequence length'],
                                            limit=report['valid limit'],
                                            headers=report['headers'],
                                            restrict_class=report['restrict class'],
                                            shuffle=report['load shuffle'],
                                            one_hot=one_hot,
                                            desired_num_classes=desired_num_classes)
        else:
            valid_data = None
        
        common.Validate_Dir(model_path)

        ann_trainer = trainer.DNN(ann=model)
        
        time_start = datetime.datetime.now()
        message = f'[INFO] Training started at: {time_start}'
        common.Print(message)
        
        ann_trainer.train(train_data=train_data,
                          valid_data=valid_data,
                          num_epochs=dictionary['epochs'],
                          shuffle=dictionary['train shuffle'],
                          verbose=dictionary['verbose'])

        time_stop = datetime.datetime.now()
        time_elapsed = time_stop - time_start
        time_per_epoch = time_elapsed / dictionary['epochs']
        message += f'\n\tTraining completed at: {time_stop}\n'
        message += f'\tElapsed time: {time_elapsed}\n'
        message += f'\tTime per epoch: {time_per_epoch}'
        common.Print(message)
        
        model = ann_trainer.ann
        
        trainer.plot_history(model.history, save=True, save_path=model_path)
        model.save_model(model_path + 'model.pt')
        common.Save_History(f'{model_path}history.txt', model.history)
        common.Save_Report(f'{model_path}results.txt', model.report)

        return model
        
    except Exception as e:
        common.Print_Error('Test ANN -> train model', e)    
    
    
if __name__=='__main__':
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    time_modifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    device = 'cuda:0'
    restrict_class = 5 
    model_type = 3
    model_names = {1:'linear', 2:'rnn', 3:'cnn'}
    model_name = model_names[model_type]

    dictionary = {# [64, 32, 26, 22, 13, 16] Max values by feature, same order as headers
                  'std': [64, 22],          # The variance to z-normalize with
                  'mean': [0, 0],           # The mean to z-normalize with
                  'epochs': 500,           # The max number of training epochs
                  'data type': 0,          # See dataloader for more information
                  'batch size': 128,       # The size of one training batch
                  # Sequence length [294, 110, 400, 170, 32, 32, 32, 32] -> restrict class [0, 1, 2, 3, 5, 6, 7, 8]
                  'sequence length': 32, 
                  # Restrict the data to a class for training models to predict a reduced class set
                  # 0 -> 3 | Binary classifier, reduce to a single class for classification (used for GAN)
                  # 5      | Multi-class classifier for all classes
                  # 6 -> 8 | Binary classifier, Reduce to a split, so classes 0 and 1 (6), 1 and 2 (7), 2 and 3 (8)
                  'restrict class': restrict_class, 
                  'device': device,      # The device to train and test on
                  'lr': 0.0001,          # Learning rate
                  'patience': 25,        # Patience for early stopping
                  # Where is the data located?
                  'data path':'../Data/Labeled/processed-low-high/',
                  'valid': True,         # Include Validaiton?
                  'train limit': 10000,  # Limit training files
                  'valid limit': 1000,   # Limit validation files
                  'tests limit': 1000,   # Limit testing files
                  # Where should the model be saved?
                  'model path': f'../Models/{model_name}/{restrict_class}/{time_modifier}/',
                  # What headers should the model load?
                  # ['Depth mm', 'Depth Delta', 'Depth Accel', 'Force lbf', 'Force Delta', 'Force Accel']
                  'headers': ['Depth mm', 'Force lbf'],
                  'load shuffle': True,  # Shuffle window while loading?
                  'train shuffle': True, # Shuffle batches during training?
                  'verbose':1,           # Print nothing [0], training info [1], everything [2]
                  'threshold':0.5}       # Binary classification threshold

    if dictionary['restrict class'] == 6:
       weights = [1, 4]
    elif dictionary['restrict class'] == 7:
       weights = [3, 1]
    elif dictionary['restrict class'] == 8:
       weights = [1, 3]
    elif dictionary['restrict class'] == 5:
        weights = [1, 5, 1, 3]
    else:
        weights = None

    outputs = 4 if dictionary['restrict class'] == 5 else 2 
    inputs = dictionary['sequence length'] * len(dictionary['headers']) if model_type == 1 else len(dictionary['headers'])
    
    if model_type == 1:    
        model = linear_ann(inputs=inputs, outputs=outputs, weights=weights)
    elif model_type == 2:
        model = rnn_ann(inputs=inputs, outputs=outputs, weights=weights)
    elif model_type == 3:
        model = cnn_ann(inputs=inputs, outputs=outputs, weights=weights)
    
    model = train_model(dictionary, model)
    test_model(dictionary, model)
        
