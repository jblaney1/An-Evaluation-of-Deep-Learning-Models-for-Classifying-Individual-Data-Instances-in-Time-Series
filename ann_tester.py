"""
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
Implements torch models through ann.py and trains them using train.py

Functions:
 - cnn_ann()
 - residual_ann()
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
        dnn.model, report = ann.create_model(model_type='cnn', # cuda 
                                             inputs=inputs, 
                                             outputs=outputs, 
                                             neurons=[128, 64, 32, 16, 8],  
                                             activations=['leakyrelu', 'leakyrelu', 'softmax'], 
                                             linear_batch_normalization=True, 
                                             linear_dropout=0.25,
                                             cnn_type='1d', 
                                             channels=[64]*2, 
                                             kernels=(3,), 
                                             strides=1, 
                                             paddings=None, 
                                             pooling='adaptivemaxpool1d', 
                                             pooling_kernel=1, 
                                             cnn_batch_normalization=True, 
                                             cnn_dropout=0.25, 
                                             cnn_sequence_length=dictionary['sequence length'])
        
        dnn.attribute_set('loss', loss)
        dnn.attribute_set('weight', weights)
        dnn.attribute_set('optimizer', optimizer)

        dnn.report = {**dnn.report, **report}

        return dnn
    except Exception as e:
        common.Print_Error('Test ANN -> CNN ANN', e)


"""
    residual_ann(inputs, outputs, weights=None)
    Implements a residual model. Change the parameters directly in this function
    to change the model. The input and output are automatically computed from 
    dictionary parameters are the bottom of this file.
    
inputs:
    - inputs (int): The number of input neurons
    - outputs (int): The number of output neurons
    - weights (list): The weights for the loss function
outputs:
    - model (uwyo dnn): A DNN object which combines a pytorch model and report dict
"""
def residual_ann(inputs, outputs, weights=None):
    try:
        loss = 'crossentropyloss' if outputs > 2 else 'bcewithlogitsloss'
        optimizer = 'adam'

        dnn = ann.DNN('cnn', ann_type=3)
        dnn.model, report = ann.create_model(model_type='residual', # cuda 
                                             inputs=inputs, 
                                             outputs=outputs, 
                                             neurons=[16], 
                                             activations=['relu', 'softmax'], 
                                             linear_batch_normalization=True, 
                                             linear_dropout=0.25,
                                             residual_type='resnet',
                                             residual_inputs=[64]*2,
                                             downsample=None,
                                             residual_norm_layer=None,
                                             strides=None,
                                             paddings=None,
                                             bottleneck_channels=None,
                                             residual_activation=None,
                                             residual_operation=None,
                                             cnn_type='1d',
                                             cnn_sequence_length=dictionary['sequence length'])
        
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
        dnn.model, report = ann.create_model(model_type='rnn', # cuda
                                             inputs=inputs, 
                                             outputs=outputs, 
                                             neurons=[32, 16, 8],
                                             activations=['leakyrelu', 'softmax'], 
                                             linear_batch_normalization=True, 
                                             linear_dropout=0.25,
                                             rnn_type='gru',
                                             hidden_size=128,
                                             num_layers=2, 
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

        model.attribute_set('testing duration', time_elapsed)
        model.attribute_set('testing accuracy', acc)

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
        model.report = {**model.report, **dictionary} 

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
                                        desired_num_classes=desired_num_classes,
                                        synth_limit=report['synth limit'],
                                        synthetic_path=report['synthetic path'])
        
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

        model = ann_trainer.ann
        keys = list(model.history.keys())

        time_elapsed = time_stop - time_start
        time_per_epoch = time_elapsed / len(model.history[keys[0]])
        message += f'\n\tTraining completed at: {time_stop}\n'
        message += f'\tElapsed time: {time_elapsed}\n'
        message += f'\tTime per epoch: {time_per_epoch}'
        common.Print(message)
        
        model = ann_trainer.ann

        model.attribute_set('training start time', time_start)
        model.attribute_set('training end time', time_stop)
        model.attribute_set('training duration', time_elapsed) 
        model.attribute_set('epoch duration', time_per_epoch)

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

    # Restrict the data to a class for training models to predict a reduced class set
    # 0 -> 3 | Binary classifier, reduce to a single class for classification (used for GAN)
    # 5      | Multi-class classifier for all classes
    # 6 -> 8 | Binary classifier, Reduce to a split, so classes 0 and 1 (6), 1 and 2 (7), 2 and 3 (8)
    # 9 -> 12| Multi-class classifier but drop one class, class 0 (9), class 1 (10), class 2 (11), class 3 (12)
    restrict_class = 5 
    model_type = 1
    model_names = {1:'residual', 2:'rnn', 3:'cnn'}
    model_name = model_names[model_type]

    # What headers should the model load? order matters for printing!
    # ['Depth mm', 'Depth Delta', 'Depth Accel', 'Force lbf', 'Force Delta', 'Force Accel']
    device = 'cuda:3'
    headers = ['Depth mm']
    headers = ['Force lbf']
    headers = ['Depth mm', 'Force lbf']

    # [64, 32, 26, 22, 13, 16] Max values by feature, same order as headers
    max_values = {'Depth mm':64,
                  'Depth Delta':32,
                  'Depth Accel':26,
                  'Force lbf':22,
                  'Force Delta':13,
                  'Force Accel':16}

    # Sequence length [294, 110, 400, 170, 32, 32, 32, 32] -> restrict class [0, 1, 2, 3, 5, 6, 7, 8]
    sequence_lengths = {0:294, 1:110, 2:400, 3:170, 5:32, 6:32, 7:32, 8:32}

    std = []
    for header in headers:
        std.append(max_values[header])

    mean = [0]*len(std)

    synthetic_paths = {'Depth mm': '../Data/generated/2024-02-12_10-25-00/Log Files Accepted',
                       'Force lbf': '../Data/generated/2024-02-12_09-25-24/Log Files Accepted',
                       'Both':'../Data/generated/2024-02-30_00-00-00/Log Files Accepted'}

    synthetic_path = synthetic_paths[headers[0]] if len(headers) < 2 else synthetic_paths['Both']

    dictionary = {# [64, 32, 26, 22, 13, 16] Max values by feature, same order as headers
                  'std': std,               # The variance to z-normalize with
                  'mean': mean,             # The mean to z-normalize with
                  'epochs': 2048,           # The max number of training epochs
                  'data type': 0,           # See dataloader for more information
                  'batch size': 128,        # The size of one training batch
                  'sequence length': sequence_lengths[restrict_class], 
                  'restrict class': restrict_class, 
                  'device': device,         # The device to train and test on
                  'lr': 0.0001,             # Learning rate
                  'patience':  50,          # Patience for early stopping
                  # Where is the data located?
                  'data path':'../Data/Labeled/processed-low-high/',
                  'synthetic path':synthetic_path,
                  'valid': True,            # Include Validaiton?
                  'train limit': 10000,     # Limit training files
                  'valid limit': 1000,      # Limit validation files
                  'tests limit': 1000,      # Limit testing files
                  'synth limit': 200000,    # Synthetic sequences to include, must be pregenerated
                  # Where should the model be saved?
                  'model path': f'../Models/{model_name}/{restrict_class}/{time_modifier}/',
                  'headers': headers,    # What headers should the model load?
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
        weights = [1, 2, 1, 2]
    else:
        weights = None

    outputs = 4 if dictionary['restrict class'] == 5 else 2 
    inputs = len(dictionary['headers'])
    
    if model_type == 1:    
        model = residual_ann(inputs=inputs, outputs=outputs, weights=weights)
    elif model_type == 2:
        model = rnn_ann(inputs=inputs, outputs=outputs, weights=weights)
    elif model_type == 3:
        model = cnn_ann(inputs=inputs, outputs=outputs, weights=weights)
    
    model = train_model(dictionary, model)
    test_model(dictionary, model)
        
