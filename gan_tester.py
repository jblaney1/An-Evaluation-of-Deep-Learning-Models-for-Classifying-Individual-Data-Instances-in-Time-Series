"""
Project: 
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
Uses the ann.py file to create GAN objects which can then be trained using the 
trainer.py file. 

Functions:
 - create_discriminator(dictionary)
 - create_cnn_generator(dictionary)
 - create_linear_generator(dictionary)
 - create_rnn_generator(dictionary)
 - find_synthetic_similarity(ref_file, syn_path, functions, headers, results_path, verbose=0)
 - generate_data(load_path, input_shape, concat_dim)
 - populate_gan_dictionary(dictionary, append)
 - save_synthetic_data(data, save_path, feature_header, restrict_class, concat_dim, label_header='Label', write='w')
 - start_testing(gan)
 - start_training(gan)

Included with: 
 - ann.py 
 - ann_tester.py
 - common.py 
 - data_generator.py
 - dataloader.py
 - gan_tester.py (current file)
 - labeler.py
 - main.py
 - messages.py
 - plotter.py
 - preprocessor.py 
 - stats.py
 - stats_class.py
 - stats_class_tester.py
 - trainer.py
 - transitioner.py
 - transitioner_tester.py
 
Notes:
For more information about the project contact 
 - Dr. Suresh Muknahallipatna -> sureshm@uwyo.edu
 - Josh Blaney -> jblaney1@uwyo.edu
"""

# Outside Dependencies
import random
import numpy as np
import pandas as pd
import datetime
import torch


# In House Dependencies
import ann
import common
import metrics
import trainer
import dataloader as dl


"""
    create_discriminator(dictionary)
    A function create a discriminator model. To change the model parameters, 
    change the variables in this function and the results of your changes will 
    propagate.
    
inputs:
 - dictionary (dict): The dictionary for the gan object
outputs:
 - model (torch model): A torch model from ann.py to be saved as the discriminator
 - (dict): The combined dictionary made from the input dict and the model dict
"""
def create_discriminator(dictionary):
    try:   
        multiclass = dictionary['classifier']
        restrict_class = dictionary['restrict class']
        
        # Setup Discriminator Parameters
        outputs = 5 if multiclass else 2
        output_activation = 'softmax' if multiclass else 'sigmoid' 
        loss = 'crossentropyloss' if multiclass else 'bcewithlogitsloss'

        dictionary['discriminator lr'] = 0.00001 if multiclass else 0.00001
        dictionary['discriminator optimizer'] = 'adam'

        # Populate discriminator object in the gan class
        if dictionary['discriminator type'] == 2: # RNN
            if restrict_class == 1:
                neurons = [32, 16]
                linear_dropout = 0.5
                hidden_size = 64
                num_layers = 2
                rnn_dropout = 0.5
               
            elif restrict_class == 2 or restrict_class == 3:
                neurons = [32, 16]
                linear_dropout = 0.25
                hidden_size = 128
                num_layers = 2
                rnn_dropout = 0.25

            else:
                neurons = [32]
                linear_dropout = 0.001
                hidden_size = 128
                num_layers = 1
                rnn_dropout = None


            model, temp_dict = ann.create_model(model_type='rnn', 
                                                inputs=len(dictionary['headers']),  
                                                outputs=outputs, 
                                                neurons=neurons, 
                                                activations=['leakyrelu', output_activation], 
                                                linear_batch_normalization=False, 
                                                linear_dropout=linear_dropout,
                                                rnn_type='gru', 
                                                hidden_size=hidden_size,  
                                                num_layers=num_layers, 
                                                batch_first=True, 
                                                rnn_dropout=rnn_dropout,
                                                bidirectional=False,
                                                proj_size=None,
                                                recurrent_batch_normalization=True)

        elif dictionary['discriminator type'] == 3: # CNN
            if restrict_class == 0 or restrict_class == 1:
                neurons = [32]
                linear_dropout = 0.5
                channels = [64]
                cnn_dropout = 0.5
            else:
                neurons = [64, 32]
                linear_dropout = 0.25
                channels = [128, 128]
                cnn_dropout = 0.25

            model, temp_dict = ann.create_model(model_type='cnn',
                                                inputs=len(dictionary['headers']),
                                                outputs=outputs,
                                                neurons=neurons,
                                                activations=['leakyrelu', 'leakyrelu', output_activation],
                                                linear_batch_normalization=True,
                                                linear_dropout=linear_dropout,
                                                cnn_type='1d',
                                                channels=channels,
                                                kernels=(3,),
                                                strides=None,
                                                paddings=None,
                                                pooling=None,
                                                pooling_kernel=None,
                                                cnn_batch_normalization=True,
                                                cnn_dropout=cnn_dropout,
                                                cnn_sequence_length=dictionary['sequence length'])

        else: # DNN
            model, temp_dict = ann.create_model(model_type='linear', 
                                                inputs=dictionary['sequence length'], 
                                                outputs=outputs, 
                                                neurons=[64], 
                                                activations=['leakyrelu', output_activation], 
                                                linear_batch_normalization=True, 
                                                linear_dropout=0.1) 
        
        temp_dict = populate_gan_dictionary(temp_dict, 'discriminator')
        temp_dict['loss'] = loss

        return model, {**dictionary, **temp_dict}

    except Exception as e:
        common.Print_Error('gan tester -> create discriminator', e)


"""
    create_cnn_generator(dictionary)
    A function to populate the gans generator object within the gan class. This
    function is specifically for populating the generator with a cnn. To change 
    the model parameters, change the variables in this function and the results
    of your changes will propagate.
    
inputs:
 - dictionary (dict): The dictionary for the gan object
outputs:
 - model (torch model): A torch model from ann.py to be saved as the generator
 - (dict): The combined dictionary made from the input dict and the model dict
"""
def create_cnn_generator(dictionary):
    try:  
        inputs = 1
        outputs = dictionary['outputs']
        seq_length = dictionary['sequence length']
        
        # Setup Generator Parameters
        dictionary['generator lr'] = 0.0001
        dictionary['generator optimizer'] = 'adam'
 
        model, temp_dict = ann.create_model(model_type='cnn', 
                                            inputs=inputs, 
                                            outputs=outputs, 
                                            neurons=[64, 32, seq_length], 
                                            activations=['leakyrelu', 'leakyrelu', 'leakyrelu'],#'sigmoid'], 
                                            linear_batch_normalization=True, 
                                            linear_dropout=0.10,
                                            cnn_type='1d', 
                                            channels=[2, 8, 32, 64, 128, 64],  
                                            kernels=(3,), 
                                            strides=None, 
                                            paddings=None, 
                                            pooling=None, 
                                            pooling_kernel=None, 
                                            cnn_batch_normalization=True, 
                                            cnn_dropout=0.10,
                                            cnn_sequence_length=seq_length)

        temp_dict = populate_gan_dictionary(temp_dict, 'generator')
        
        return model, {**dictionary, **temp_dict}
    
    except Exception as e:
        common.Print_Error('gan tester - > create cnn gan', e)


"""
    create_linear_generator(dictionary)
    A function to populate the gans generator object within the gan class. This
    function is specifically for populating the generator with a dnn. To change 
    the model parameters, change the variables in this function and the results
    of your changes will propagate.
    
inputs:
 - dictionary (dict): The dictionary for the gan object
outputs:
 - model (torch model): A torch model from ann.py to be saved as the generator
 - (dict): The combined dictionary made from the input dict and the model dict
"""
def create_linear_generator(dictionary):
    try:
        inputs = 1 
        outputs = dictionary['outputs']
        seq_length = dictionary['sequence length']

        # Setup Generator Parameters
        dictionary['generator optimizer'] = 'adam'
        dictionary['generator lr'] = 0.0001
        
        model, temp_dict = ann.create_model(model_type='linear', 
                                            inputs=inputs,
                                            outputs=outputs, 
                                            neurons=[2, 4, 8, 16, 32, 64, 128, 256, seq_length], 
                                            activations=['leakyrelu', 'leakyrelu'], 
                                            linear_batch_normalization=True,
                                            linear_dropout=0.5)

        temp_dict = populate_gan_dictionary(temp_dict, 'generator')

        return model, {**dictionary, **temp_dict}
    
    except Exception as e:
        common.Print_Error('gan tester - > create linear generator', e)


"""
    create_rnn_generator(dictionary)
    A function to populate the gans generator object within the gan class. This
    function is specifically for populating the generator with a rnn. To change 
    the model parameters, change the variables in this function and the results
    of your changes will propagate.
    
inputs:
 - dictionary (dict): The dictionary for the gan object
outputs:
 - model (torch model): A torch model from ann.py to be saved as the generator
 - (dict): The combined dictionary made from the input dict and the model dict
"""
def create_rnn_generator(dictionary):
    try:
        inputs = 1
        outputs = dictionary['outputs']
        seq_length = dictionary['sequence length']
        
        # Setup Generator Parameters
        dictionary['generator lr'] = 0.00001
        dictionary['generator optimizer'] = 'adam'
        
        model, temp_dict = ann.create_model(model_type='rnn', 
                                            inputs=inputs,
                                            outputs=outputs, 
                                            neurons=[128, 256, seq_length], 
                                            activations=['leakyrelu', 'leakyrelu'],
                                            linear_batch_normalization=True, 
                                            linear_dropout=0.5,
                                            rnn_type='gru', 
                                            hidden_size=128,
                                            num_layers=4, 
                                            batch_first=True,
                                            rnn_dropout=0.5)
        
        temp_dict = populate_gan_dictionary(temp_dict, 'generator')
        
        return model, {**dictionary, **temp_dict}
        
    except Exception as e:
        common.Print_Error('gan tester - > create rnn generator', e)


"""
    find_synthetic_similarity(ref_file, syn_path, functions, headers, results_path, thresholds=None, move_path=None, verbose=0)
    A function to automate the process of running similarity functions over
    a dataset comparing each synthetic sequence to the same reference sequence.
    Information about the similarity measures of each file are saved in a csv
    file, the statistics for each similarity measure are computed over each
    header and saved to a text file, and (optionally) files which meet specified
    thresholds are saved into a new directory.
    
inputs:
    - ref_file (string): The complete path to the reference data 
    - syn_path (string): The file path to the synthetic data to test (unglobbed)
    - functions (list): References to the distance functions to use for metrics
    - headers (list): The headers analyze when running comparisons
    - restrict_class (int): The label of the class to keep
    - results_path (string): The location to store the two results files
    - thresholds (dict): The min/max thresholds to evaluate synthetic data against
    - move_path (string): The location to move the acceptable synthetic data
    - verbose (int): Should a status bar be displayed? > 0 Should the best results print? > 1
outputs:
    -
"""
def find_synthetic_similarity(ref_file, syn_path, functions, headers, restrict_class, results_path, thresholds=None, move_path=None, verbose=0):

    try:
        save = thresholds is not None and move_path is not None
        files_to_move = [] if save else None
        
        report = {'Reference File':ref_file,
                  'Synthetic Path':syn_path,
                  'Functions List':functions,
                  'Headers List':headers}
        
        columns = []
        ref_data = pd.read_csv(ref_file)
        ref_data = ref_data[ref_data['Label']==restrict_class]
        ref_data = ref_data[headers]
        
        syn_data = dl.Preload_Data(f'{syn_path}/*.csv', headers)
        similarity_data = np.zeros((len(syn_data), len(headers)*len(functions)))
        
        syn_count = len(syn_data)
        function_count = len(functions)
        
        for k in range(len(syn_data)):
            move_file = True if save else False
            
            for j, header in enumerate(headers):
                for i, function in enumerate(functions):
                    similarity = metrics.custom_metrics(function, syn_data[k][header].values, ref_data[header].values)
                    similarity_data[k,(j*function_count)+i] = similarity
                    
                    if save and move_file:
                        if similarity > thresholds[header][function]['max'] or similarity < thresholds[header][function]['min']:
                            move_file = False
            
            if move_file:
                files_to_move.append(k)
                

            if verbose > 0:
                common.Print_Status('Find Synthetic Similarity', k, syn_count)
        
        report['Structure'] = 'Header - Function: Epoch - Min Value | Epoch - Max Value | Mean Value | STD Value'
        
        for j, header in enumerate(headers):
            for i, function in enumerate(functions):
                col = (j*function_count) + i
                max_index = np.argmax(similarity_data[:,col])
                max_value = similarity_data[max_index,col]
                
                min_index = np.argmin(similarity_data[:,col])
                min_value = similarity_data[min_index,col]
                
                mean_value = np.mean(similarity_data[:,col])
                std_value = np.std(similarity_data[:,col])
                
                columns.append(f'{header} - {function}')
                report[f'{header} - {function:<5}'] = f'{min_index:<5} - {min_value:<6} | {max_index:<5} - {max_value:<7} | {mean_value:<7} | {std_value:<7}'
                
            report[f'{header} - {function:<5}'] += '\n'
        
        if verbose > 1:
            common.Print_Message('Synthetic Similarity Results', report)
            
        similarity_df = pd.DataFrame(similarity_data, columns=columns)
        similarity_df.to_csv(f'{results_path}/similarity_results.csv', index=False)
        common.Save_Report(f'{results_path}/similarity_results.txt', report)
        
        if save:
            for index in files_to_move:
                file = f'{move_path}/{index}.csv'
                df = pd.DataFrame(syn_data[index], columns=headers)
                df.to_csv(file, index=False)
    
    except Exception as e:
        print(function)
        common.Print_Error('Gan Tester -> Find Synthetic Similarity', e)
    

"""
    generate_data(load_path, input_shape, concat_dim)
    A function to use the generator portion of a GAN to generate synthetic
    data. This can be used to generate data from multiple gan. After generation,
    the data are then concatenated on the specified dimension.
    
inputs:
    - load_path (string/list): The location of the generator model(s)
    - input_shpae (list): The number of datapoints to generate
    - concat_dim (int): The dimension which to concatenate data on
outputs:
    - data (torch tensor): The resulting generated data
"""
def generate_data(load_path, input_shape, concat_dim):

    model_count = len(load_path)

    for index, path in enumerate(load_path):
        generator = torch.jit.load(path)
        generator.eval()

        inputs = torch.normal(mean=0.0, std=1.0, size=input_shape)

        if index == 0:
            data = generator(inputs).detach()
            output_shape = (list(data.shape) + [1])
        else:
            data = torch.reshape(data, output_shape)
            temp = torch.reshape(generator(inputs).detach(), data.shape)
            data = torch.cat((data, temp), dim=concat_dim)

        common.Print_Status('Generate Synthetic Data', index, model_count)

    return data


"""
    populate_gan_dictionary(dictionary, append)
    A function to add the string in append to the keys in a dictionary. This
    function does not perform the operation in place.
    
    inputs:
     - dictionary (dict): The dictionary to change the keys of
     - append (string): The string to add to the front of the dict keys
    outputs:
     - temp_dict (dict): A new dictionary with the altered keys
"""
def populate_gan_dictionary(dictionary, append):
    try:
        temp_dict = {}
        for key in dictionary.keys():
            temp_dict[f'{append} {key}'] = dictionary[key]
            
        return temp_dict
    
    except Exception as e:
        common.Print_Error('GAN Tester -> populate gan dictionary', e)


"""
    save_synthetic_data(data, save_path, feature_header, restrict_class, label_header='Label', write='w')
    A function to save pytorch model outputs. This is especially useful for
    saving the output of the generator portion of GAN networks for later use
    in datasets. 

    inputs:
     - data (array): The output from the model, most likely as a tensor
     - save_path (string): The folder to save the data in
     - feature_header (string): The header of the generated data
     - restrict_class (int): Specifies which class was generated
     - concat_dim (int):
     - label_header (string): The name of the header to use for
     - write (char): The write type 'w' write, 'a' append, 'o' overwrite
    outputs:
     - 
"""
def save_synthetic_data(data, save_path, feature_header, restrict_class, concat_dim, label_header='Label', write='w'):

    sample_count = data.shape[0]
    folder_existed = not common.Validate_Dir(save_path)

    if not folder_existed and write == 'w':
        for index, sample in enumerate(data):
            df = pd.DataFrame(sample, columns=[feature_header])
            df[label_header] = restrict_class
            df.to_csv(f'{save_path}/{index}.csv', index=False)

            common.Print_Status('Save Synthetic Data (Write)', index, sample_count)

    elif write == 'o':
        for index, sample in enumerate(data):
            df = pd.DataFrame(sample, columns=[feature_header])
            df[label_header] = restrict_class
            df.to_csv(f'{save_path}/{index}.csv', index=False)

            common.Print_Status('Save Synthetic Data (Overwrite)', index, sample_count)


"""
    start_testing(dictionary, model)
    A function to interface with the gan training function. It also can provide
    some nifty console messages about training if the verbose is set.
"""
def start_testing(gan):
    try: 
        name = gan.report['name']
        data_path = '../Data/Labeled/processed-low-high/testing' if gan.report['data path tests'] is None else gan.report['data path tests']
        model_path = f'../Models/gan/{name}/' if gan.report['model path'] is None else gan.report['model path']
        
        desired_num_classes = 5 if gan.report['classifier'] else 2

        test_dl = dl.Preload_Dataset(data_path + '/*.csv',
                                     ann_type=gan.report['discriminator type'],
                                     data_type=gan.report['discriminator type'],
                                     batch_size=gan.report['batch size'],
                                     mean=gan.report['mean'],
                                     std=gan.report['std'],
                                     sequence_length=gan.report['sequence length'],
                                     limit=gan.report['tests limit'],
                                     headers=gan.report['headers'],
                                     restrict_class=gan.report['restrict class'],
                                     classifier=gan.report['classifier'],
                                     one_hot=int(gan.report['discriminator outputs'])>1,
                                     label_smooth=None,
                                     desired_num_classes=desired_num_classes)
        
        gan.attribute_set('device', gan.report['device']) 
        gan.attribute_set('batch size', gan.report['batch size'])

        gan_trainer = trainer.GAN(gan)        

        time_start = datetime.datetime.now()
        acc, confusion_matrix = gan_trainer.test(test=test_dl, 
                                             threshold=gan.report['threshold'],
                                             classifier=gan.report['classifier'],
                                             load_gan=gan.report['load gan'])

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

        common.Save_Results(f'{model_path}results.txt', gan.report, confusion_matrix)

    except Exception as e:
        common.Print_Error('gan test - > start testing', e)


"""
    start_training(gan)
    A function to interface with the gan training function. It also can provide
    some nifty console messages about training if the verbose is set.
"""
def start_training(gan):
    try:
        name = gan.report['name']
        save_path = f'../Data/generated/{name}/Log Files/' if gan.report['save path'] is None else gan.report['save path'] 
        data_path = '../Data/Labeled/processed-low-high/training/' if gan.report['data path train'] is None else gan.report['data path train']
        model_path = f'../Models/gan/{name}/' if gan.report['model path'] is None else gan.report['model path']

        desired_num_classes = 5 if gan.report['classifier'] else 2

        train_dl = dl.Preload_Dataset(data_path + '/*.csv',
                                      ann_type=gan.report['discriminator type'],
                                      data_type=gan.report['discriminator type'],
                                      batch_size=gan.report['batch size'],
                                      mean=gan.report['mean'],
                                      std=gan.report['std'],
                                      sequence_length=gan.report['sequence length'],
                                      limit=gan.report['train limit'],
                                      headers=gan.report['headers'],
                                      restrict_class=gan.report['restrict class'],
                                      shuffle=gan.report['load shuffle'],
                                      classifier=gan.report['classifier'],
                                      one_hot=int(gan.report['discriminator outputs'])>1,
                                      label_smooth=gan.report['label smooth'],
                                      desired_num_classes=desired_num_classes)

        gan.attribute_set('device', gan.report['device']) 
        gan.attribute_set('batch size', gan.report['batch size'])
        
        time_start = datetime.datetime.now()
        message = f'[INFO] Training started at: {time_start}'
        common.Print(message)

        gan_trainer = trainer.GAN(gan)
        history = gan_trainer.train(num_epochs=gan.report['epochs'], 
                                    train=train_dl,
                                    generator_class=gan.report['generator class'],
                                    classifier=gan.report['classifier'],
                                    headers=gan.report['headers'],
                                    save=True, 
                                    save_path=save_path,
                                    overwrite=gan.report['overwrite'],
                                    shuffle=gan.report['train shuffle'],
                                    verbose=gan.report['verbose'])

        time_stop = datetime.datetime.now()
        time_elapsed = time_stop - time_start
        time_per_epoch = time_elapsed / gan.report['epochs']
        message += f'\n\tTraining completed at: {time_stop}\n'
        message += f'\tElapsed time: {time_elapsed}\n'
        message += f'\tTime per epoch: {time_per_epoch}'
        common.Print(message)

        gan.attribute_set('Training start time', time_start)
        gan.attribute_set('Training end time', time_stop)
        gan.attribute_set('Training elapsed time', time_elapsed)
        gan.attribute_set('Training time per epoch', time_per_epoch)

        gan.save_all(f'{model_path}generator.pt', f'{model_path}discriminator.pt')
        common.Save_History(model_path + 'history.txt', history)
        common.Save_Report(f'{model_path}training.txt', gan.report)
        trainer.plot_history(history, save=True, save_path=model_path)

        return gan, history
        
    except Exception as e:
        common.Print_Error('gan tester - > start training', e)

    
if __name__=='__main__':
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    process = 2

    # Train and Test GAN
    if process == 0:
        model_names = {1:'linear', 2:'rnn', 3:'cnn'}

        restrict_class = 5 
        device = 'cuda:1'
        classifier_string = 'multiclass' if restrict_class == 5 else 'binary'

        discriminator_type = 3 
        discriminator_string = model_names[discriminator_type]

        generator_type = 3#int(device.split(':')[-1])
        generator_name = model_names[generator_type]

        time_modifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sequence_lengths = {0:294, 1:110, 2:400, 3:170, 5:32}

        dictionary = {#[64, 31.9, 25.59, 21.99, 12.84, 25.29] Feature max values, same order as headers
                      'std': [1,1],                                         # Z-Norm standard deviation values
                      'mean': [0,0],                                         # Z-Norm mean values
                      'label smooth': None,                                # Standard deviation for label smoothing
                      'device': device,                                    # Device to run on
                      'epochs': 16384,                                      # Number of epochs to train for
                      'classifier': classifier_string=='multiclass',       # Multi-class or  binary classification
                      'discriminator type':discriminator_type,             # 1: linear, 2:rnn, 3:cnn
                      'batch size': 128,                                     # The batch size for the dataloader
                      # Sequence Lengths [294, 110, 400, 170, 32] -> Restrict Classes [0, 1, 2, 3, 5]
                      'sequence length': sequence_lengths[restrict_class], # The expected length of input sequences
                      # Restrict the data to a class for training models to predict a reduced class set
                      # 0 -> 3 | Binary classifier, reduce to a single class for classification (used for GAN)
                      # 5      | Multi-class classifier for all classes
                      # 6 -> 8 | Binary classifier, Reduce to a split, so classes 0 and 1 (6), 1 and 2 (7), 2 and 3 (8)
                      'restrict class': restrict_class,
                      'generator class': 1,                                # Restrict generator to specific class?
                      'threshold': 0.8,                                    # Binary classifier threshold
                      # Which headers should be included?
                      # ['Depth mm', 'Depth Delta', 'Depth Accel', 'Force lbf', 'Force Delta', 'Force Accel']
                      'headers':['Depth mm', 'Force lbf'], 
                      'train limit':10000,                                 # Limit the number of training files? 
                      'tests limit':1000,                                  # Limit the number of testing files?
                      # Where will the data be loaded from?
                      'data path train':'../Data/Labeled/processed-low-high/training',
                      'data path tests':'../Data/Labeled/processed-low-high/testing',
                      # Where will synthetic data be saved?
                      'save path':f'../Data/generated/{time_modifier}/Log Files/',
                      # Where will the model and results be saved?
                      'model path':f'../Models/gan/{time_modifier}/',
                      'time modifier':time_modifier,                       # A modifier to distinguish models
                      'overwrite':True,                                    # Overwrite existing files?
                      'load shuffle':True,                                 # Shuffle data on load? See dataloader.py for more
                      'train shuffle':True,                                # Shuffle data during training? See trainer.py for more
                      'load gan':None,#'../Models/gan/2023-11-07_15-24-24/generator.pt',
                      'prop gan':False,
                      'verbose':1}                                         # Print verbosity 0: None, 1: Minimal, 2:All
 
        gan = ann.GAN(name=generator_name, 
                      generator_type=generator_type, 
                      discriminator_type=discriminator_type) 

        seq_length = dictionary['sequence length']
        header_count = len(dictionary['headers'])

        dictionary['outputs'] = seq_length if header_count < 3 else header_count*seq_length

        if generator_type == 1:
            generator, dictionary = create_linear_generator(dictionary)
        elif generator_type == 2:
            generator, dictionary = create_rnn_generator(dictionary)
        elif generator_type == 3:
            generator, dictionary = create_cnn_generator(dictionary)

        for index in range(len(dictionary['headers'])):
            gan.generator_set(generator) 

        gan.discriminator, dictionary = create_discriminator(dictionary)

        gan.report = {**gan.report, **dictionary}

        history = start_training(gan)
        start_testing(gan)

    # Use a generator to generate data
    elif process == 1:

        headers = ['Depth mm', 
                   'Force lbf']

        model_locations = ['2023-11-07_15-24-24',
                           '2023-11-07_17-32-04']

        load_paths = [f'../Models/gan/{model_locations[0]}/generator.pt',
                      f'../Models/gan/{model_locations[1]}/generator.pt']

        save_path = f'../Data/generated/{model_locations[0]}_new/'

        restrict_class = 1
        sequence_lengths = {0:294, 1:110, 2:400, 3:170, 5:32}

        generator_type = 3
        num_samples = 10000

        if generator_type == 1:
            input_shape = (num_samples, 1)
            concat_dim = 1

        elif generator_type == 2:
            input_shape = (num_samples, sequence_lengths[restrict_class], 1)
            concat_dim = 2

        elif generator_type == 3:
            input_shape = (num_samples, 1, sequence_lengths[restrict_class])
            concat_dim = 2

        data = generate_data(load_paths, input_shape, concat_dim=concat_dim)
        save_synthetic_data(data, save_path, headers, restrict_class, concat_dim)

    # Find the similarity between synthetic data and a reference
    elif process == 2:
        
        verbose = 1
        restrict_class = 1
        time_modifier = '2023-11-07_15-24-24_post'
        base_file_path = '../Data/generated'
        results_file_path = f'{base_file_path}/{time_modifier}/'
        reference_sequence_path = f'{base_file_path}/{time_modifier}/Log Files/618.csv'
        synthetic_sequence_path = f'{base_file_path}/{time_modifier}/Log Files'
        
        thresholds = None
        move_path = None
        
        comparison_headers = ['Depth mm', 'Force lbf']
        comparison_metrics = ['ed','fc','dtw']
        
        find_synthetic_similarity(reference_sequence_path, 
                                  synthetic_sequence_path, 
                                  comparison_metrics, 
                                  comparison_headers, 
                                  restrict_class,
                                  results_file_path, 
                                  thresholds,
                                  move_path,
                                  verbose)
