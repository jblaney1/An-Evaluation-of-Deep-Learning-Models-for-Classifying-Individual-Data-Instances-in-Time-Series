"""
Project: 
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
Used for torch model creation. Can be used to create many torch models quickly.
See ann_tester.py for examples of creating classification and regression models.
See gan_tester.py for examples of creating generative adversarial networks.

Functions:
 - build_cnn(dictionary)
 - build_linear(dictionary)
 - build_rnn(dictionary)
 - create_model()
 - expand_activations(activations, length)
 - expand_to_list(value, length)
 - get_activations(names=False)
 - get_pooling(names=False)
 - process_cnn_inputs()
 - process_global_inputs()
 - process_rnn_inputs()
 - set_activation(name)
 - set_pooling(name)
 
 DNN()
 - __init__(name, ann_type=0)
 - attribute_get(name)
 - attribute_set(name, value)
 - save_all(file_name='../Models/default/')
 - save_model(file_name, method='all', args={})
 - save_report(file_name)
  
 extract_tensor(torch.nn.Module)
 - forward(x)
 
 GAN()
 - __init__(name, discriminator_type=0, generator_type=0, device='cpu')
 - attribute_get(purpose, name)
 - attribute_set(name, value)
 - discriminator_get()
 - discriminator_set(model, dictionary)
 - generator_get()
 - generator_set(model, dictionary)
 - save_all(file_name_generator='../Models/gan/generator', file_name_discriminator='../Models/gan/discriminator')
 - save_model(purpose, file_name, method='all', args={}
 - save_report(file_name)

Included with: 
 - ann.py (current file)
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
 - trainer.py
 - transitioner.py
 
Notes:
For more information about the project contact 
 - Dr. Suresh Muknahallipatna -> sureshm@uwyo.edu
 - Josh Blaney -> jblaney1@uwyo.edu
"""

# Outisde Dependencies
import torch
from pytorch_model_summary import summary


# In House Dependencies
import common      
        
        
"""
    build_cnn(dictionary)
    A function to build a cnn model based on the infomration passed in through
    the dictionary. The entire pytorch model is returned. For information on 
    formatting the dictionary see the function create_model()
    
inputs:
    - dictionary (dict): A dictionary specifying the model structure
outputs:
    - model (pytorch model): The complete pytorch ML model
"""
def build_cnn(dictionary):
    common.Print('[INFO] Starting Build CNN Process')
    try:
        # Instantiate an empty sequential object
        model = torch.nn.Sequential()
        outputs = dictionary['inputs']

        # Identify if this is a 1d cnn or 2d cnn
        if dictionary['cnn type'] == '1d':
            conv = torch.nn.Conv1d
            dictionary['cnn batch norm function'] = torch.nn.BatchNorm1d
            dummy = torch.ones((1, dictionary['inputs'], dictionary['sequence length']))
           
        elif dictionary['cnn type'] == '2d':
            conv = torch.nn.Conv2d
            dictionary['cnn batch norm function'] = torch.nn.BatchNorm2d
            dummy = torch.ones((1, outputs, dictionary['image width'], dictionary['image height']))
        
        # Iterate through the dictionary and add the requested CNN layers
        for i in range(0, dictionary['cnn layers']):

            inputs = outputs
            outputs = dictionary['channels'][i]

            model.add_module(f'conv {i}', 
                             conv(in_channels=inputs,
                                  out_channels=outputs,
                                  kernel_size=dictionary['kernels'][i],
                                  stride=dictionary['strides'][i],
                                  padding=dictionary['paddings'][i]))

            if dictionary['activations'][i] is not None:
                model.add_module(f'activation {i}', dictionary['activations'][i]()) 

            outputs = model(dummy).shape[1]

            if dictionary['pooling'] is not None:
                model.add_module(f'pool {i}', dictionary['pooling'](kernel_size=(dictionary['pooling kernel'])))

            if dictionary['cnn batch normalization']:
                model.add_module(f'batch normalization {i}', dictionary['cnn batch norm function'](num_features=outputs))

            if dictionary['cnn dropout'] is not None:
                if dictionary['cnn dropout'][i] > 0:
                    model.add_module(f'dropout {i}', torch.nn.Dropout(dictionary['cnn dropout'][i]))

            outputs = model(dummy).shape[1]  

            common.Print_Status('Add CNN', i, dictionary['cnn layers'])
        
        # Add the requisite flatten layer to prep for linear layers
        i += 1
        model.add_module('flatten', torch.nn.Flatten())
        outputs = model(dummy).shape[1]
        
        # Iterate through the dictionary and add the requested linear layers
        j = 0 
        for j in range(dictionary['linear layers']):

            inputs = outputs
            outputs = dictionary['neurons'][j]
            model.add_module(f'linear {j+i}', torch.nn.Linear(inputs, outputs))
            
            if dictionary['activations'][j+i] is not None:
                model.add_module(f'activation {j+i}', dictionary['activations'][j+i]())   

            if dictionary['linear batch normalization'] and not dictionary['cnn batch normalization']:
                model.add_module(f'batch normalization {j+i}', torch.nn.BatchNorm1d(num_features=outputs))

            if dictionary['linear dropout'] is not None:
                if dictionary['linear dropout'][j] > 0:
                    model.add_module(f'dropout {j+i}', torch.nn.Dropout(dictionary['linear dropout'][j]))

            common.Print_Status('Add Linear Layers', j, dictionary['linear layers'])
        
        # Add the requisite output layer
        inputs = outputs
        outputs = dictionary['outputs']
        model.add_module(f'linear {j+i+1}', torch.nn.Linear(inputs, outputs))

        if dictionary['activations'][-1] is not None:
            model.add_module(f'activation {j+i+1}', dictionary['activations'][-1]())
 
        return model 
    
    except Exception as e:
        common.Print_Error('Builder -> build cnn', e)
        return
    

"""
    build_linear(dictionary)
    A function to build a linear model based on the infomration passed in through
    the dictionary. The entire pytorch model is returned. For information on 
    formatting the dictionary see the function create_model()
    
inputs:
    - dictionary (dict): A dictionary specifying the model structure
outputs:
    - model (pytorch model): The complete pytorch ML model 
"""
def build_linear(dictionary):
    
    common.Print('[INFO] Starting Build Linear Process')
    try:
        # Instantiate an empty pytorch model
        model = torch.nn.Sequential()
        outputs = dictionary['inputs']
        
        # Iterate through the dictionary and add the requested linear layers
        for i in range(dictionary['linear layers']):
            inputs = outputs
            outputs = dictionary['neurons'][i]
            model.add_module(f'linear {i}', torch.nn.Linear(inputs, outputs))

            if dictionary['activations'] is not None:
                model.add_module(f'activation {i}', dictionary['activations'][i]())  

            if dictionary['linear batch normalization']:
                model.add_module(f'batch normalization {i}', torch.nn.BatchNorm1d(num_features=outputs))

            if dictionary['linear dropout'] is not None:
                if dictionary['linear dropout'][i] > 0:
                    model.add_module(f'dropout {i}', torch.nn.Dropout(dictionary['linear dropout'][i]))

            common.Print_Status('Add Linear Layers', i, dictionary['linear layers'])

        # Add the requisite output layer
        inputs = outputs
        outputs = dictionary['outputs']
        model.add_module(f'linear {i+1}', torch.nn.Linear(inputs, outputs))
        
        if dictionary['activations'][-1] is not None:
            model.add_module(f'activation {i+1}', dictionary['activations'][-1]())

        return model
    
    except Exception as e:
        common.Print_Error('Builder -> build linear', e)
        return


"""
    build_rnn()
    A function to build a rnn model based on the infomration passed in through
    the dictionary. The entire pytorch model is returned. For information on 
    formatting the dictionary see the function create_model()
    
inputs:
    - dictionary (dict): A dictionary specifying the model structure
outputs:
    - model (pytorch model): The complete pytorch ML model
"""
def build_rnn(dictionary):
    common.Print('[INFO] Starting Build RNN Process')
    try:
        # Instantiate an empty pytorch model
        model = torch.nn.Sequential()
        dummy = torch.ones((1,1,dictionary['inputs']))
        
        # Use the pytorch functionality to setup multi-layer rnn
        if dictionary['rnn type'] == 'rnn':
            model.add_module('rnn', torch.nn.RNN(input_size=dictionary['inputs'],
                                           hidden_size=dictionary['hidden size'],
                                           num_layers=dictionary['rnn layers'],
                                           nonlinearity=dictionary['non-linearity'],
                                           bias=dictionary['bias'],
                                           batch_first=dictionary['batch first'],
                                           dropout=dictionary['rnn dropout'],
                                           bidirectional=dictionary['bidirectional']))
            
        elif dictionary['rnn type'] == 'lstm':                
            model.add_module('lstm', torch.nn.LSTM(input_size=dictionary['inputs'],
                                             hidden_size=dictionary['hidden size'],
                                             num_layers=dictionary['rnn layers'],
                                             bias=dictionary['bias'],
                                             batch_first=dictionary['batch first'],
                                             dropout=dictionary['rnn dropout'],
                                             bidirectional=dictionary['bidirectional'],
                                             proj_size=dictionary['projection size']))
            
        elif dictionary['rnn type'] == 'gru':
            model.add_module('gru', torch.nn.GRU(input_size=dictionary['inputs'],
                                           hidden_size=dictionary['hidden size'],
                                           num_layers=dictionary['rnn layers'],
                                           bias=dictionary['bias'],
                                           batch_first=dictionary['batch first'],
                                           dropout=dictionary['rnn dropout'],
                                           bidirectional=dictionary['bidirectional']))
        """
            Use the extract_tensor() class, similar to a flatten layer in cnn,
            to grab the specific output from the final rnn layer necessary to 
            append linear layers to this model. For more information on why 
            this process is necessary in pytorch, see the pytorch docs on 
            multi-layer recurrent networks [1] and the solution [2].
            [1] https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
            [2] https://stackoverflow.com/questions/72667646/how-to-connect-a-lstm-layer-to-a-linear-layer-in-pytorch
        """
        model.add_module('extractor', extract_tensor())
        outputs = model(dummy).shape[1]
        
        if dictionary['recurrent batch normalization']:
            model.add_module('recurrent batch normalization', torch.nn.BatchNorm1d(num_features=outputs))
 
        # Iterate through the dicitonary and append the requested linear layers
        for i in range(0, dictionary['linear layers']):
            inputs = outputs
            outputs = dictionary['neurons'][i]
            model.add_module(f'linear {i-1}', torch.nn.Linear(inputs, outputs))

            if dictionary['activations'][i] is not None:
                model.add_module(f'activation {i-1}', dictionary['activations'][i]())   
            
            if dictionary['linear batch normalization']:
                model.add_module(f'batch normalization {i}', torch.nn.BatchNorm1d(num_features=outputs))
            
            if dictionary['linear dropout'] is not None:
                if dictionary['linear dropout'][i] > 0:
                    model.add_module(f'dropout {i}', torch.nn.Dropout(dictionary['linear dropout'][i]))
            
            common.Print_Status('Add Linear Layers', i, dictionary['linear layers'])
        
        i = dictionary['linear layers']
			
        # Add the requisite output layer
        inputs = outputs
        outputs = dictionary['outputs']
        model.add_module(f'linear {i+1}', torch.nn.Linear(inputs, outputs))

        if dictionary['activations'][-1] is not None:
            model.add_module(f'activation {i+1}', dictionary['activations'][-1]())

        return model
    
    except Exception as e:
        common.Print_Error('Builder -> build rnn', e)
        return

    
"""
    create_model(model_type, inputs, outputs, neurons, activations, linear_batch_normalization=None, linear_dropout=None,
                 cnn_type=None, channels=None, image_width=None, image_height=None, kernels=None, strides=None, paddings=None, pooling=None, cnn_batch_normalization=None, cnn_dropout=None,
                 rnn_type=None, hidden_size=None, num_layers=None, bias=None, batch_first=None, rnn_dropout=None, bidirectional=None, proj_size=None)
    A function to automate the building of different network architectures.
    It can build linear, cnn, and rnn models using any of PyTorch's activation
    and pooling layers. The most flexible models are the rnn as they have the
    simplest setup.
    
inputs:
    global
    - purpose (string): The pupose of the model (generator, discriminator)
    - model_type (string): The model to build [linear, cnn, rnn]
    - inputs (int): The number of input neurons (linear/rnn) or channels (cnn)
    - outputs (int): The number of output neurons
    - neurons (list): A list of ints denoting the number of hidden neurons in each layer
    - activations (list): A list of strings denoting the names of the activation functions
    - linear_batch_normalization (bool): Should the linear layers be batch normalized?
    - linear_dropout (float): What percent of linear neurons be dropped?
    
    cnn
    - cnn_type (string): The cnn model to build [None, 1d, 2d]
    - channels (list): A list of ints denoting the number of output channels in each layer
    - image_width (int): The number of columns in the input array
    - image_height (int): The number of rows in the input array
    - kernels (list/tuple): A list of tuples or a tuple denoting the kernel size
    - strides (list/int): A list of ints or an int denoting the stride size
    - paddings (list/int): A list of ints or an int denoting the padding size
    - cnn_batch_normalization (bool): Should the cnn layers be batch normalized?
    - pooling (string): The name of the pooling function to use
    
    rnn
    - rnn_type (string): The rnn model to build [None, rnn, lstm, gru]
    - hidden_size (int): The number of hidden units each layer will have
    - num_layers (int): The number of rnn layers to create
    - bias (bool): Should the network use bias weights?
    - batch_first (bool): Is the batch the first dimension of the data?
    - dropout (float): The percent of neurons to dropout
    - bidirectional (bool): Is the network bidirectional?
    - proj_size (int): Only applies to LSTM projections
    - recurrent_batch_normalization (bool): Batch normalization after recurrent section?
outputs:
    - model (pytorch model): The complete pytorch model
    - dictionary (dict): The dictionary used for the creation of the model
"""
def create_model(model_type, inputs, outputs, neurons, activations, linear_batch_normalization=None, linear_dropout=None,
                 cnn_type=None, channels=None, image_width=None, image_height=None, kernels=None, strides=None, paddings=None, pooling=None, pooling_kernel=None, cnn_batch_normalization=None, cnn_dropout=None, cnn_sequence_length=None,
                 rnn_type=None, hidden_size=None, num_layers=None, bias=None, batch_first=None, rnn_dropout=None, bidirectional=None, proj_size=None, recurrent_batch_normalization=None):
    try:
        model_types = ['linear', 'cnn', 'rnn']
        
        if model_type not in model_types:
            common.Print_Input_Warning('Model -> create model', model_types)
            return
        
        dictionary = {}
        
        if not process_global_inputs(inputs, outputs, neurons, activations, linear_batch_normalization, linear_dropout):
            return
        
        elif model_type == 'linear':
            layers = len(neurons)
            dictionary['inputs'] = inputs
            dictionary['outputs'] = outputs
            dictionary['neurons'] = neurons
            dictionary['linear layers'] = layers
            dictionary['linear batch normalization'] = False if linear_batch_normalization is None else linear_batch_normalization
            dictionary['linear dropout'] = expand_to_list(0, layers) if linear_dropout is None else expand_to_list(linear_dropout, layers)
            dictionary['activations'] = expand_activations(activations, [layers,1])
            model = build_linear(dictionary)
            
            dummy = torch.ones(1, int(dictionary['inputs']))
                
        elif model_type == 'cnn':
            if process_cnn_inputs(cnn_type, channels, image_width, image_height, kernels, strides, paddings, pooling, pooling_kernel, cnn_batch_normalization, cnn_dropout, cnn_sequence_length):
                layers = len(neurons)
                dictionary['inputs'] = inputs
                dictionary['outputs'] = outputs
                dictionary['neurons'] = neurons
                dictionary['linear layers'] = layers
                dictionary['linear batch normalization'] = False if linear_batch_normalization is None else linear_batch_normalization
                dictionary['linear dropout'] = expand_to_list(0,layers) if linear_dropout is None else expand_to_list(linear_dropout, layers)
                
                layers = len(channels)
                dictionary['cnn type'] = cnn_type
                dictionary['channels'] = channels
                dictionary['cnn layers'] = layers
                dictionary['image width'] = image_width
                dictionary['image height'] = image_height
                dictionary['kernels'] = expand_to_list(kernels, layers) if type(kernels) is tuple else kernels
                dictionary['strides'] = expand_to_list(1, layers) if strides is None else expand_to_list(strides, layers)
                dictionary['paddings'] = expand_to_list(0, layers) if paddings is None else expand_to_list(paddings, layers)
                dictionary['pooling'] = pooling if pooling is None else set_pooling(pooling)
                dictionary['pooling kernel'] = 3 if pooling_kernel is None and pooling is not None else pooling_kernel
                dictionary['cnn batch normalization'] = False if cnn_batch_normalization is None else cnn_batch_normalization
                dictionary['cnn dropout'] = expand_to_list(0, layers) if cnn_dropout is None else expand_to_list(cnn_dropout, layers)
                dictionary['activations'] = expand_activations(activations, [dictionary['cnn layers'], dictionary['linear layers'], 1])
                dictionary['sequence length'] = cnn_sequence_length
                model = build_cnn(dictionary)
                
                if dictionary['cnn type'] == '1d':
                    dummy = torch.ones(1, int(dictionary['inputs']), int(dictionary['sequence length']))
                elif dictionary['cnn type'] == '2b':
                    dummy = torch.ones(int(dictionary['inputs']), int(dictionary['image width']), int(dictionary['image height']))

            else:
                return
            
        
        elif model_type == 'rnn': 
            if process_rnn_inputs(rnn_type, hidden_size, num_layers, bias, batch_first, rnn_dropout, bidirectional, proj_size, recurrent_batch_normalization):
                layers = len(neurons)
                dictionary['inputs'] = inputs
                dictionary['outputs'] = outputs
                dictionary['neurons'] = neurons
                dictionary['linear layers'] = layers
                dictionary['linear batch normalization'] = False if linear_batch_normalization is None else linear_batch_normalization
                dictionary['linear dropout'] = expand_to_list(0, layers) if linear_dropout is None else expand_to_list(linear_dropout, layers)
                dictionary['rnn type'] = rnn_type
                dictionary['hidden size'] = hidden_size
                dictionary['rnn layers'] = num_layers
                dictionary['bias'] = True if bias is None else bias
                dictionary['batch first'] = False if batch_first is None else batch_first
                dictionary['rnn dropout'] = 0 if rnn_dropout is None else rnn_dropout
                dictionary['bidirectional'] = False if bidirectional is None else bidirectional
                dictionary['projection size'] = 0 if proj_size is None else proj_size
                dictionary['recurrent batch normalization'] = False if recurrent_batch_normalization is None else recurrent_batch_normalization
                dictionary['activations'] = expand_activations(activations, [dictionary['linear layers'], 1])
                model = build_rnn(dictionary)
                
                dummy = torch.ones(1,1,dictionary['inputs'])

            else:
                return

        print(summary(model, dummy, show_input=True))

        return model, dictionary

    except Exception as e:
        common.Print_Error('GAN -> create model', e)  


"""
    expand_activations(activations, length)
    A function to expand a list of 2 or 3 activation functions to equal the
    number of hidden layers specified in length. It is assumed that the order
    of activations is linear, output (2) or cnn, linear, output (3). The 
    resulting list of activation functions is stored in the report dictionary.
    
inputs:
    - activations (list): A list of strings denoting the names of the activation functions
    - length (list): A list of ints specifing the number of layers using said activations
outputs:
    - 
"""
def expand_activations(activations, length):
    try:
        acts = []
        if len(activations) < sum(length):
            for i, value in enumerate(length):
                for j in range(value):
                    acts.append(set_activation(activations[i]))
            
        else:
            for activation in activations:
                acts.append(set_activation(activation))
            
        return acts
    
    except Exception as e:
        common.Print_Error('GAN -> expand activations', e)
    

"""
    expand_to_list(value, length)
    A function to expand a value into a list of that value.
    
inputs:
    - value (): The value to copy throughout the list
    - length (int): The length of the output list
outputs:
    - output_list (list): The list of expanded values
"""
def expand_to_list(value, length):
    try:
        output_list = []
        for i in range(length):
            output_list.append(value)
            
        return output_list
    
    except Exception as e:
        common.Print_Error('GAN -> expand to list', e)
			
			
"""
    get_activations(names=False)
    A function which returns either (1) the list of available activation functions
    by name, (2) the list of available activation functions as a dictionary of 
    torch functions
    
inputs:
 - names (bool): Should the names be returned?
outputs:
 - activations (list or dict): The selected object of activations
"""
def get_activations(names=False):
    try:
        activations = {'elu':torch.nn.ELU,
                       'hardshrink':torch.nn.Hardshrink,
                       'hardsigmoid':torch.nn.Hardsigmoid,
                       'hardtanh':torch.nn.Hardtanh,
                       'hardswish':torch.nn.Hardswish,
                       'leakyrelu':torch.nn.LeakyReLU,
                       'logsigmoid':torch.nn.LogSigmoid,
                       'multiheadattention':torch.nn.MultiheadAttention,
                       'prelu':torch.nn.PReLU,
                       'relu':torch.nn.ReLU,
                       'relu6':torch.nn.ReLU6,
                       'rrelu':torch.torch.nn.RReLU,
                       'selu':torch.nn.SELU,
                       'celu':torch.nn.CELU,
                       'gelu':torch.nn.GELU,
                       'sigmoid':torch.nn.Sigmoid,
                       'silu':torch.nn.SiLU,
                       'mish':torch.nn.Mish,
                       'softplus':torch.nn.Softplus,
                       'softshrink':torch.nn.Softshrink,
                       'softsign':torch.nn.Softsign,
                       'tanh':torch.nn.Tanh,
                       'tanhshrink':torch.nn.Tanhshrink,
                       'threshold':torch.nn.Threshold,
                       'glu':torch.nn.GLU,
                       'softmin':torch.nn.Softmin,
                       'softmax':torch.nn.Softmax,
                       'softmax2d':torch.nn.Softmax2d,
                       'logsoftmax':torch.nn.LogSoftmax,
                       'adaptivelogsoftmaxwithloss':torch.nn.AdaptiveLogSoftmaxWithLoss}
        
        activations = list(activations.keys()) if names else activations
        return activations
    
    except Exception as e:
        common.Print_Error('Trainer -> get activations', e)
        
    
"""
    get_pooling(name)
    A function to get the list of available pooling functions.
    
inputs:
    - names (bool): Return names? (or functions?)
outputs:
    - poolers (list or dict): The selected object of poolers
"""
def get_pooling(names=False):
    try:
        poolers = {'maxpool1d':torch.nn.MaxPool1d,
                   'maxpool2d':torch.nn.MaxPool2d,
                   'maxpool3d':torch.nn.MaxPool3d,
                   'avgpool1d':torch.nn.AvgPool1d,
                   'avgpool2d':torch.nn.AvgPool2d,
                   'avgpool3d':torch.nn.AvgPool3d,
                   'fractionalmaxpool2d':torch.nn.FractionalMaxPool2d,
                   'fractionalmaxpool3d':torch.nn.FractionalMaxPool3d,
                   'lppool1d':torch.nn.LPPool1d,
                   'lppool2d':torch.nn.LPPool2d,
                   'adaptivemaxpool1d':torch.nn.AdaptiveMaxPool1d,
                   'adaptivemaxpool2d':torch.nn.AdaptiveMaxPool2d,
                   'adaptivemaxpool3d':torch.nn.AdaptiveMaxPool3d,
                   'adaptiveavgpool1d':torch.nn.AdaptiveAvgPool1d,
                   'adaptiveavgpool2d':torch.nn.AdaptiveAvgPool2d,
                   'adaptiveavgpool3d':torch.nn.AdaptiveAvgPool3d}
        
        poolers = list(poolers.keys()) if names else poolers
        return poolers
        
    except Exception as e:
        common.Print_Error('GAN -> set pooling', e)
        
		
"""
    process_cnn_inputs(cnn_type, channels, image_width, image_height, kernels, strides, paddings, cnn_batch_normalization, pooling)
    A function to validate the inputs to build a convolution neural network. 
    If any of the inputs are invalid a boolean value of False is returned 
    and an appropriate error message is printed. See function create_model()
    for more information about what each variable is used for.
    
inputs:
    - See create_model() for more information on about the inputs
outputs:
    - success (bool): Was the validation successful?
"""
def process_cnn_inputs(cnn_type, channels, image_width, image_height, kernels, strides, paddings, pooling, pooling_kernel, cnn_batch_normalization, cnn_dropout, cnn_sequence_length):
    
    e = ''
    success = True
    cnn_types = [None, '1d', '2d']
    
    if cnn_type not in cnn_types:
        e += f'cnn type not recognized\nRecieved {cnn_type}'
        common.Print_Input_Warning('Model -> process cnn inputs', cnn_types)
        success = False
    
    elif type(cnn_type) is not str or cnn_type is None:
        e += f'cnn type must be of type <str>\nBut Recieved {cnn_type}'
        success = False
        
    elif (type(channels) is not list and type(channels) is not int) or channels is None:
        e += f'channels must be of type <list> or <int>\nBut Recieved {channels}'
        success = False
    
    elif type(image_width) is not int and image_width is not None:
        e += f'image width must be of type <int>\nBut Recieved {image_width}'
        success = False
    
    elif type(image_height) is not int and image_height is not None:
        e += f'image height must be of type <int>\nBut Recieved {image_height}'
        success = False
    
    elif type(kernels) is not list and type(kernels) is not tuple and type(kernels) is not int and kernels is not None:
        e += f'kernels must be of type <list> or <tuple>\nBut Recieved {kernels}'
        success = False
    
    elif type(strides) is not list and type(strides) is not int and strides is not None:
        e += f'strides must be of type <list> or <int>\nBut Recieved {strides}'
        success = False
    
    elif type(paddings) is not list and type(paddings) is not int and paddings is not None:
        e += f'paddings must be of type <list> or <int>\nBut Recieved {paddings}'
        success = False
        
    elif type(pooling) is not str and pooling is not None:
        e += 'pooling must be of type <str>\nBut Recieved {pooling}'
        success = False
        
    elif (type(pooling_kernel) is not int or type(pooling_kernel) is not tuple) and pooling_kernel is not None:
        e += 'pooling kernels must be of type <int> or <tuple> of ints\nBut Recieved {pooling_kernel}'
        success = False
        
    elif type(cnn_batch_normalization) is not bool and cnn_batch_normalization is not None:
        e += f'cnn batch normalization must be of type <float>\nBut Recieved {cnn_batch_normalization}'
        success = False
        
    elif type(cnn_dropout) is not float and cnn_dropout is not None:
        e += f'cnn dropout must be of type <float>\nBut Recieved {cnn_dropout}'
        success = False
    
    elif type(cnn_sequence_length) is not int and cnn_sequence_length is not None:
        e += f'cnn sequence length must be of type <int>\nBut Recieved {cnn_sequence_length}'
        success = False
 
    if not success:
        common.Print_Error('GAN -> process cnn inputs', e)
    
    return success
    

"""
    process_global_inputs(inputs, outputs, neurons, activations, batch_normalization, dropout)
    A function to validate the required inputs to build any network. If any
    of the inptus are invalid a boolean of False is returned and an 
    appropriate error message is printed. See function create_model() for
    more information about what each variable is used for.
    
inputs:
    - See create_model() for more information on about the inputs
outputs:
    - success (bool): Was the validation successful?
"""
def process_global_inputs(inputs, outputs, neurons, activations, batch_normalization, dropout):
    
    e = ''
    success = True
    
    if type(inputs) is not int or inputs < 1:
        e += f'inputs must be of type <int> and > 0\nBut recieved {inputs}'
        success = False
    
    elif type(outputs) is not int or outputs < 1:
        e += f'outputs must be of type <int> and > 0\nBut recieved {outputs}'
        success = False
        
    elif type(neurons) is not list:
        e += f'neurons must be of type <list>\nBut recieved {neurons}'
        success = False
    
    elif type(activations) is not list and type(activations) is not str:
        e += f'activations must be of type <list> or <str>\nBut recieved {activations}'
        success = False
    
    elif type(batch_normalization) is not bool and batch_normalization is not None:
        e += f'batch normalizations must be of type <float>\nBut recieved {batch_normalization}'
        success = False
    
    elif type(dropout) is not float and dropout is not None:
        e += f'dropout must be of type <list> or <float>\nBut recieved {dropout}'
        success = False
        
    if not success:
        common.Print_Error('GAN -> process global inputs', e)
    
    return success
    
    
"""
    process_rnn_inputs(rnn_type, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size)
    A function to validate the inputs to build an rnn neural network. If any
    of the inputs are invalid a boolean value of False is returned and an 
    appropriate error message is printed. See function create_model() for 
    more information about what each variable is used for.

inputs:
    - See create_model() for more information on about the inputs
outputs:
    - success (bool): Was the validation successful?
"""
def process_rnn_inputs(rnn_type, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size, recurrent_batch_normalization):
    
    e = ''
    success = True
    rnn_types = [None, 'rnn', 'lstm', 'gru']
    
    if rnn_type not in rnn_types:
        e += f'rnn type not recognized\nRecieved {rnn_type}'
        common.Print_Input_Warning('Model -> process rnn inputs', rnn_types)
        success = False
        
    elif type(rnn_type) is not str or rnn_type is None:
        e += f'rnn type must be of type <str>\nBut Recieved {rnn_type}'
        success = False
        
    elif type(hidden_size) is not int or hidden_size is None:
        e += f'hidden size must be of type <int>\nBut Recieved {hidden_size}'
        success = False
        
    elif type(num_layers) is not int or num_layers is None:
        e += f'num layers must be of type <int>\nBut Recieved {num_layers}'
        success = False
        
    elif type(bias) is not bool and bias is not None:
        e += f'bias must be of type <bool>\nBut Recieved {bias}'
        success = False
        
    elif type(batch_first) is not bool and batch_first is not None:
        e += f'batch first must be of type <bool>\nBut Recieved {batch_first}'
        success = False
        
    elif type(dropout) is not float and dropout is not None:
        e += f'dropout must be of type <float>\nBut Recieved {dropout}'
        success = False
        
    elif type(bidirectional) is not bool and bidirectional is not None:
        e += f'bidirectional must be of type <bool>\nBut Recieved {bidirectional}'
        success = False
        
    elif type(proj_size) is not int and proj_size is not None:
        e += f'proj size must be of type <int>\nBut Recieved {proj_size}'
        success = False
    
    elif type(recurrent_batch_normalization) is not bool and recurrent_batch_normalization is not None:
        e += f'recurrent batch normalization must be of type <bool>\nBut Recieved {recurrent_batch_normalization}'
        success = False

    if not success:
        common.Print_Error('GAN -> process inputs rnn', e)
    
    return success

		
"""
    set_activation(name)
    A function to set the activation type in the report dictionary. The 
    input name must be of type <str> and the defualt value is ReLU
    
inputs:
    - name (string): The name of the function to use
outputs:
    - activation (reference): A reference to the activation function
"""
def set_activation(name):
    try:
        if name is None:
            return None
        
        name = name.lower()
        activations = get_activations()
        activation = torch.nn.ReLU
        
        for key in activations.keys():
            if key == name:
                activation = activations[key]
                break
                
        return activation
    
    except Exception as e:
        common.Print_Error('GAN -> set activation', e)    


"""
    set_pooling(name)
    A function to set the pooling function in the report dictionary. A 
    pooling layer is not required so name can be None, in which case the 
    majority of the function is skipped.
    
inputs:
    - name (string): The name of the pooling function to use
outputs:
    -
"""
def set_pooling(name):
    try:
        if name is None:
            None
        else:
            name = name.lower()
            poolers = get_pooling()
            pooler = None
            
            for key in poolers.keys():
                if key == name:
                    pooler = poolers[key]
                    break
            
            return pooler
        
    except Exception as e:
        common.Print_Error('GAN -> set pooling', e)
            
            
"""
    DNN
    A class to house deep neural networks for classificaiton and regression.
"""
class DNN:

    def __init__(self, name, ann_type=0):
        self.report = {} 
        self.history = {}
        self.model = None
        
        self.attribute_set('name', name)
        self.attribute_set('ann type', ann_type)
    
    
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
        save_all(file_name='../Models/default/')
        A function to save the model and report
    """
    def save_all(self, file_name='../Models/default/'):
        file_name_folder = common.Split_Folder(file_name)
        common.Validate_Dir(file_name_folder)
        self.save_model(file_name=file_name, method='state dict')
        self.save_report(file_name=f'{file_name_folder}report')
        
    
    """
        save_model(file_name, method='all', args={})
        A function to save the model
    """
    def save_model(self, file_name, method='all', args={}):
        try:
            method = method.lower()
                
            if method == 'state dict':
                torch.save(self.model.state_dict(), file_name)
            elif method == 'all':
                torch.save(self.model, file_name)
            elif method == 'script':
                script = torch.jit.script(self.model)
                script.save(file_name)
            elif method == 'custom':
                torch.save(args, file_name)
                
        except Exception as e:
            common.Print_Error('DNN -> save model', e)
            
    
    """
        save_report(file_name)
        A function to save the report
    """
    def save_report(self, file_name):
        common.Save_Report(file_name + '.txt', self.report)
        

"""
    extract_tensor(torch.nn.Module)
    A neural network class which is used to grab the actual output from 
    recurrent layers so that it can be processed by linear layers. Pytorch
    recurrent layers return all time steps, but the only time step of interest
    is the last one. For batched inputs, the expected shape has 3 entries with 
    the ordering [batch size, sequence length, projection size]. We only 
    want the last entry in in the sequence, but we want all entries in the batch
    and the projection, so the access is [:,-1,:] and the new shape has 2 
    entries, a projection for each batch.
"""
class extract_tensor(torch.nn.Module):
    def forward(self, x):
        x, _ = x
        return x[:, -1, :]
    
    
"""
    GAN
    A class which stores two models, a generator and a discriminator, and a 
    report dictionary which describes each in turn. Used by gan_tester.py to pass
    generator objects between training and testing functions and by trainer.py
    to access the two models during training and testing.
"""
class GAN():
    def __init__(self, name, discriminator_type=0, generator_type=0, device='cpu'):
        self.train = []
        self.report = {} 
        self.generators = []
        self.disciminator = None
 
        self.attribute_set('name', name)
        self.attribute_set('device', device)
        self.attribute_set('discriminator type', discriminator_type)
        self.attribute_set('generator type', generator_type)
        
        
    """
        attribute_get(purpose, name)
        A function to retrieve values stored in the report
        
    inputs:
        - name (string): The key to retrieve the information from
    outputs:
        -
    """
    def attribute_get(self, name):
        try:
            return self.report[name]
        except Exception as e:
            common.Print_Error('GAN -> attribute get', e)
    
    
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
            common.Print_Error('GAN -> attribute set', e)


    """
        discriminator_get()
        A function to get the report information about the generator and 
        print it to the console.
        
    inputs:
        -
    outputs:
        -
    """
    def discriminator_get(self):
        try:
            message = ""
            for key in self.report.keys():
                if 'discriminator' in key:
                    message += f'{key} <- {self.report[key]} \n'
            
            common.Print(message)
            return self.generator
        
        except Exception as e:
            common.Print_Error('GAN -> dicsriminator get', e)
              
    
    """
        discriminator_set(model, dictionary)
        A function to set the generator informaiton in the report dictionary.
        
    inputs:
        - model (pytorch model): The model to use as the generator
        - dictionary (dictionary): The report information for the model
    outputs:
        -
    """
    def discriminator_set(self, model, dictionary):
        try:
            for key in dictionary.keys():
                self.report[f'discriminator {key}'] = str(dictionary[key])
            self.discriminator = model
        except Exception as e:
            common.Print_Error('GAN -> dsicriminator set', e)    

    
    """
        generator_get()
        A function to get the report information about the generator and 
        print it to the console.
        
    inputs:
        -
    outputs:
        -
    """
    def generator_get(self):
        try:
            message = ""
            for key in self.report.keys():
                if 'generator' in key:
                    message += f'{key} <- {self.report[key]} \n'
            
            common.Print(message)
            return self.generator
        
        except Exception as e:
            common.Print_Error('GAN -> generator get', e)


    """
        generator_set(model, dictionary)
        A function to set the generator informaiton in the report dictionary.
        
    inputs:
        - model (pytorch model): The model to use as the generator
        - dictionary (dictionary): The report information for the model
    outputs:
        -
    """
    def generator_set(self, model):
        try:
            self.train.append(True)
            self.generators.append(model)
        except Exception as e:
            common.Print_Error('GAN -> generator set', e)


    """
        save_all()
        A function to handle saving everything stored in the current gan object.
        
    inputs:
     - file_name_generator (string): The name to save the generator under
     - file_name_discriminator (string): The name of save the discriminator under
    outputs:
     - 
    """
    def save_all(self, file_name_generator='../Models/gan/generator', file_name_discriminator='../Models/gan/discriminator'):
        file_name_generator_folder = common.Split_Folder(file_name_generator)
        file_name_discriminator_folder = common.Split_Folder(file_name_discriminator)
        common.Validate_Dir(file_name_generator_folder)
        common.Validate_Dir(file_name_discriminator_folder)

        for model in self.generators:
            self.generator = model
            self.save_model(purpose='generator', file_name=file_name_generator, method='all')

        self.save_model(purpose='discriminator', file_name=file_name_discriminator, method='all')
        self.save_report(file_name=f'{file_name_generator_folder}report')
        
    
    """
        save_model(purpose, file_name, method='all', args={})
        A function to handle pytorch model saving for both the discriminator
        and the generator
        
    inputs:
     - purpose (string): generator or discriminator?
     - file_name (string): The name to save the model under
     - method (string): The pytorch save method
     - args (dict): Teh pytorch save method arguments
    outputs:
     - 
    """
    def save_model(self, purpose, file_name, method='all', args={}):
        try:
            purpose = purpose.lower()
            method = method.lower()
            
            if purpose.lower() == 'generator':
                model = self.generator
            elif purpose.lower() == 'discriminator':
                model = self.discriminator
                
            if method == 'state dict':
                torch.save(model.state_dict(), file_name)
            elif method == 'all':
                torch.save(model, file_name)
            elif method == 'script':
                script = torch.jit.script(model)
                script.save(file_name)
            elif method == 'custom':
                torch.save(args, file_name)
                
        except Exception as e:
            common.Print_Error('GAN -> save model', e)
            
    
    """
        save_report(file_name)
        A wrapper function for the common.py Save_Report function
        
    inputs:
     - file_name (string): The name of the file to save the report to
    """
    def save_report(self, file_name):
        common.Save_Report(file_name + '.txt', self.report)


