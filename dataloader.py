"""
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
Provides all of the necessary data loading utilities for the machine learning
portion of this project. The top level function for loading data during training
and testing is Preload_Dataset. See the function comments for more information.

Functions:
 - Augment_Data(features, labels, mean, std, randomize=False)
 - Batch_Dataset(data, targets, batch_size, data_type)
 - Load_Data(file, headers=['Depth mm'], rescale=True, mean=[0.0], std=[0.0], restrict_class=5)
 - Make_One_Hot(y, outputs=4)
 - Preload_Data(files)
 - Preload_Dataset()
 - Rescale(data, mean, std)
 - Window_Dataset(data, targets, sequence_length)

Included with: 
 - ann.py 
 - ann_tester.py
 - common.py 
 - data_generator.py
 - dataloader.py (current file)
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
import glob
import torch
import random
import numpy as np
import pandas as pd

# In House Dependencies
import common
import stats


"""
    Augment_Data(features, labels, mean, std, randomize=False)
    A function to automate data augmentation through adding gaussian noise to specific classes.
    The class can either be randomly chosen or static by changing class_index.
    
inputs:
    - features (numpy array): The input features to be augmented
    - labels (numpy array): The corresponding labels for features
    - mean (numpy array): The mean value for each class
    - std (numpy array): The standard deviation for each class
    - randomize (bool): Should the class be randomly chosen?
outputs:
    - features (numpy array): The inplace augmented feature vector
"""
def Augment_Data(features, labels, mean, std, randomize=False):
    try:
        if randomize:
            class_index = random.randrange(0, 3, 1)
        else:
            class_index = 1
            
        percent = random.randrange(0, 50, 1) / 100
        data_count = len(features)
        once = False
        
        min_index = 0
        max_index = data_count
        
        for index in range(data_count):
            if not once and labels[index] == class_index:
                min_index = index
                once = True
            elif once and labels[index] != class_index:
                max_index = index
                break
        
        augment_count = int(percent * (max_index - min_index))
        mu = mean[class_index]
        sigma = std[class_index]
        
        for i in range(augment_count):
            index = random.randrange(min_index, max_index, 1)
            features[index] += random.gauss(mu=mu, sigma=sigma) / 100
            
        return features
    
    except Exception as e:
        common.Print_Error('Dataloader -> Augment Data', e)


"""
    Batch_Dataset(data, targets, batch_size, data_type)
    A function to batch input data
    
    inputs:
     - data (np nd-array): The dataset of features in shape ()
     - targets (np nd-array): The dataset of labels in shape ()
     - batch_size (int): The number of inputs to include in each batch
     - data_type (int): Used to change the expected shape of the data
    outputs:
     - dataset (np nd-array): 
     - labels (np nd-array):
"""
def Batch_Dataset(data, targets, batch_size, data_type):
    try:
        inputs = data.shape[-1] if len(data.shape) > 2 else 1
        one_hot = True if len(targets.shape) > 1 else False
        batch_size = batch_size if batch_size is not None and batch_size < data.shape[0] else data.shape[0]
        num_batches = int(data.shape[0]/batch_size)
        sequence_length = data.shape[1] if len(data.shape) > 1 else data.shape[0]
        labels = np.empty((num_batches, batch_size, targets.shape[-1])) if one_hot else np.empty((num_batches, batch_size))        

        if data_type == 1:
            data = data.squeeze()
            dataset = np.empty((num_batches, batch_size, sequence_length))
            
            for i in range(num_batches):
                j = i * batch_size
                dataset[i,::] = data[j:j+batch_size,::]
                labels[i,::] = targets[j:j+batch_size,::] if one_hot else targets[j:j+batch_size]
                common.Print_Status('Batching', i, num_batches)
            
        elif data_type == 0 or data_type == 2 or data_type == 3:
            dataset = np.empty((num_batches, batch_size, sequence_length, inputs)) if data_type != 3 else np.empty((num_batches, batch_size, inputs, sequence_length))

            for i in range(num_batches):
                j = i * batch_size
                dataset[i,::] = np.reshape(data[j:j+batch_size,::], (dataset.shape[1:]))
                labels[i,::] = targets[j:j+batch_size,::] if one_hot else targets[j:j+batch_size]
                common.Print_Status('Batching', i, num_batches)
   
        return dataset, labels
    
    except Exception as e:
        common.Print_Error('Dataloader -> Batch Dataset', e)
    
    
"""
    Load_Data(file, headers=['Depth mm'], label_header="Label", rescale=True, mean=[0.0], std=[0.0], restrict_class=5, sequence_length=0)
    A function used to load individual files and preprocess for ML.
    
inputs:
    - file (string): The location of the csv file to process
    - headers (list): The name of the columns to draw data from.
    - label_header (string): The name of the header ot use for labels
    - rescale (bool): Should the features be normalized?
    - mean (numeric): If rescaling mean should be provided
    - std (numeric): If rescaling std should be provided
    - restrict_class (int): Restrict the data to one class? [0:3] 5 denotes all
    - sequence_length (int): used to adjust the indexing on the low index data 
    - synthetic_path (string): The location of a synthetic instance to use in place of real data
outputs:
    - features (numpy array): The selected ML features
    - labels (numpy array): The ML labels
"""
def Load_Data(file, headers=['Depth mm'], label_header="Label", rescale=True, mean=[0.0], std=[1.0], restrict_class=5, sequence_length=0, synthetic_path=None):
    try:        
        df = pd.read_csv(file)[headers + [label_header]]
        
        if restrict_class < 5 and restrict_class >= 0:
            df = df[df[label_header] == restrict_class]
            df[label_header] = df[label_header] - restrict_class

        elif restrict_class > 5 and restrict_class <= 8:
            classes = df[label_header].unique()
            low_class = restrict_class - 6
            high_class = low_class + 1

            if low_class in classes and high_class in classes:
                low_data = df[df[label_header] == low_class]
                high_data = df[df[label_header] == high_class]
                high_index = len(df)-1 if high_class == 3 else high_data.index.values[-1]
                low_index = 0 if low_class == 0 else (low_data.index.values[0] - sequence_length) 
                df = df.loc[low_index:high_index,:]
                df[label_header] = df[label_header] - low_class
            else:
                print(file)
                return None, None

        labels = df[label_header].values        
        raw = df[headers].values
        
        if rescale:
            features = Rescale(raw, mean, std)
        else:
            features = raw
                  
        return features, labels
    
    except Exception as e:
        common.Print_Error('Dataloader -> Load Data', f'File: {file}\n{e}')
        return None, None


"""
    Make_One_Hot(y, outputs=4)
    A function to expand an integer label into a one hot vector label with the 
    specified number of outptus.
    
    inputs:
     - y (np array): The array of integer labels
     - outputs (int): The number of desired labels
    outputs:
     - one_hot_y (np array): The array of one hot vectors
"""
def Make_One_Hot(y, outputs=4):
    try:
        one_hot_y = np.zeros((y.shape[0], outputs))
        
        for i, entry in enumerate(y):
            one_hot_y[i, int(entry)] = 1
            
        return one_hot_y
    except Exception as e:
        common.Print_Error('Dataloader -> Make One Hot', e)

    
"""
    Preload_Data(file_path, headers=None, columns_to_remove=None)
    Loads a dataset from a file list into a dictionary for fast access by the calling function

inputs: 
    - file_path (string): The location of the data to load
    - headers (list): The headers to include in the dataframe
    - columns_to_remove(list): The headers to remove from the dataframe
outputs:
    - data(dictionary): A dictionary of input file feature data without labels.
"""
def Preload_Data(file_path, headers=None, columns_to_remove=None):
    try:
        files = glob.glob(file_path) if type(file_path) == str else file_path
        file_count = len(files)
        data = {}
        
        for index in range(file_count):
            data[index] = pd.read_csv(files[index])
            
            if headers is not None:     
                data[index] = data[index][headers]
                
            common.Print_Status("Preload", index, file_count)
        
        return data
    
    except Exception as e:
        common.Print_Error('Dataloader -> Preload', e)
        

"""
    Preload_Dataset(file_path, data_type, batch_size, mean, std, 
                    sequence_length=None, limit=None, augment=None, headers=None, 
                    restrict_class=5, shuffle=True, classifier=False, one_hot=False, label_smooth=1.0)
    A function to automate csv data loading and preprocessing for training data of timeseries ML. 
    Each file is loaded, passed to the windowing function, appended to the dataset, and then
    the dataset is passed to the batching function.
    
inputs:
    - file_path (string): A glob ready location of the training data
    - data_type (int): What is the structure of the data to be loaded? (linear - 1, rnn - 2)
    - batch_size (int): Controls the amount of data seen before making a network update
    - mean (numeric): Mean to center the data around, see Rescale() for more
    - std (numeric): STD to normalize the data to, see Rescale() for more
    - sequence_length (int): Controls the number of timesteps in each window
    - limit (int): A limit on the number of files to load
    - augment (int): The number of augmented samples to generate
    - headers (string/list): The name of the header of the feature vector
    - restrict_class (int): Restrict the data to one class? [0:3] 5 denotes all
    - shuffle (bool): Should the data within a window be shuffled?
    - classifier (bool): Multi-class or binary classifier?
    - one_hot (bool): One hot or integer multi-class?
    - label_smooth (float): STD of the normal distribution to use for label smoothing
    - smooth_percent (float): The percentage of data to smooth the labels of
outputs:
    - dataset (list): A list of timeseries datasets ready for individual fit by a ML model
"""
def Preload_Dataset(file_path, ann_type, data_type, batch_size, mean, std, 
                    sequence_length=None, limit=None, augment=None, headers=None, 
                    restrict_class=5, shuffle=True, classifier=False, one_hot=False, 
                    label_smooth=None, smooth_percent=0.0, desired_num_classes=None,
                    synth_limit=0, synthetic_path=None):
    try:
        if type(headers) == list:
            assert len(headers) == len(mean) and len(headers) == len(std), "Length of header, Mean, and STD lists must match"
            
        process = 'Loading' if data_type == 1 else 'Windowing'
        file_list = glob.glob(file_path)
        file_count = len(file_list)
        dataset = None
        labels = None
    
        if limit is not None:
            if limit <= file_count:
                early_stop_index = int(limit)
            else:
                early_stop_index = file_count
        else:
            early_stop_index = file_count
            
        if augment is not None:
            augment_count = augment
            data_stats = stats.mean_std_class(file_list)
            if type(headers) == list:
                for header in headers:
                    class_mean = data_stats[header + " - mean"].values
                    class_std = data_stats[header + " - std"].values
            else:
                class_mean = data_stats[headers + " - mean"].values
                class_std = data_stats[headers + " - std"].values
        else:
            augment_count = 0

        length = 4096
        sequence_length = length if sequence_length is None else sequence_length
        
        print(f'[INFO] Prelaoding {early_stop_index} / {file_count} files from: {file_path}')
        
        stop_index = early_stop_index + augment_count
        smooth_count = int(smooth_percent * early_stop_index)

        for index in range(stop_index):
            access_index = random.randrange(0, file_count)

            x, y = Load_Data(file=file_list[access_index],
                             headers=headers,
                             mean=mean,
                             std=std,
                             restrict_class=restrict_class,
                             sequence_length=sequence_length,
                             synthetic_path=synthetic_path)

            if x is not None and y is not None and len(y) > 0:
                if index > early_stop_index:
                    if index < early_stop_index + augment_count:
                        x = Augment_Data(x, y, class_mean, class_std)
                        common.save_csv(f'../Data/Augmented/{index-early_stop_index}.csv', x, y, headers)
                        
                if one_hot:
                    if desired_num_classes is None:
                        y = Make_One_Hot(y, outputs=5) if classifier else Make_One_Hot(y) 
                    else:
                        y = Make_One_Hot(y, outputs=desired_num_classes)

                if data_type == 0: # CLASSIFIER
                    y = y[sequence_length:,::] if one_hot else y[sequence_length:]
                    data = Window_Dataset(data=x, 
                                          targets=y, 
                                          sequence_length=sequence_length)
            
                else: # GAN
                    if sequence_length < len(x):
                        data = Window_Dataset(data=x,
                                              targets=y[sequence_length:],
                                              sequence_length=sequence_length)

                        y = y[sequence_length:, ::] if one_hot else np.ones((len(data),1))

                    else:
                        data = np.zeros((1,int(sequence_length),x.shape[-1]))
                        data[:,:x.shape[0]] = np.reshape(x.T, (1,x.shape[0],x.shape[-1]))
                        y = np.reshape(y[-1,::],(1,y[-1,::].shape[-1])) if one_hot else np.ones((1,1))

                if label_smooth is not None and index < smooth_count:
                    y = y - np.random.normal(scale=label_smooth, size=y.shape)

                labels = y if labels is None else np.append(labels, y, axis=0)
                dataset = data if dataset is None else np.append(dataset, data, axis=0)

            common.Print_Status(process, index, stop_index)
   
        if synth_limit > 0:
            synth_dataset = np.zeros((synth_limit, sequence_length, len(headers)))
            synth_file_list = glob.glob(f'{synthetic_path}/*.csv')
            synth_file_count = len(synth_file_list)

            print(f'[INFO] Loading {synth_limit} / {synth_file_count} Synthetic Sequences from: {synthetic_path}')

            for i in range(synth_limit):
                synth_df = pd.read_csv(synth_file_list[i%synth_file_count])
                synth_dataset[i,:,:] = synth_df[headers].values

                common.Print_Status('Loading Synth', i, synth_limit)

            synth_labels = np.zeros((synth_limit, labels.shape[-1]))
            synth_labels[:,synth_df["Label"][0]] = 1

            labels = np.append(labels, synth_labels, axis=0)
            dataset = np.append(dataset, synth_dataset, axis=0)

        if shuffle:
            dataset, labels = common.Shuffle(dataset, labels)

        dataset, labels = Batch_Dataset(dataset, labels, batch_size, ann_type)
        
        return torch.from_numpy(dataset), torch.from_numpy(labels)
    
    except Exception as e:
        common.Print_Error('Dataloader -> Preload Dataset', e) 
        return 


"""
    Rescale(data, mean, std)
    Used to normalize an array of data. The data will be centered around mean and normalized
    based on std. If mean is zero but std is not the data will be normalized based on std but 
    may not be centered around zero. If both mean and std are zero, precomputed values will be used.
    
inputs:
    - data (numpy array): The input array to be processed
    - mean (numeric): The point to center the data around, 0.0 does nothing
    - std (numeric): The value to normalize the data to, 0.0 does nothing
outputs:
    - data (numpy array): A normalized array
"""
def Rescale(data, mean, std):
    try:
        if len(data.shape) > 1:
            assert type(mean) is list and type(std) is list, "[ERROR] Mean and STD must be of type <list> for multivariate inputs"
            assert data.shape[1] == len(mean) and data.shape[1] == len(std), "[ERROR] Mean and STD must be the same length as the number of inputs"
            
            for index in range(data.shape[1]):
                data[:,index] = (data[:,index] - mean[index]) / std[index]
            
        else:
            assert type(mean) is float and type(std) is float, "[ERROR] Mean and STD must be of type <float> for univariate inputs"
            
            data -= mean
            data /= std
        
        return data
    
    except Exception as e:
        common.Print_Error('Dataloader -> Rescale', e)   


"""
    Window_Dataset(data, targets, sequence_length, shuffle)
    A function to window the frovided data and targets with the option to shuffle
    
    inputs:
     - data (np nd-array): The dataset of features in shape ()
     - targets (np nd-array): The dataset of labels in shape ()
     - sequence_length (int): The number of inputs to include in a sequence
    outputs:
     - dataset (np nd-array): The windowed dataset
"""
def Window_Dataset(data, targets, sequence_length):
    try:
        inputs = data.shape[-1] if len(data.shape) > 1 else 1
        sequence_count = len(targets)
        dataset = np.empty(shape=(sequence_count, sequence_length, inputs))
        
        for i in range(sequence_count):
            dataset[i,::] = data[i:i+sequence_length,:] if inputs > 1 else data[i:i+sequence_length]
        
        return dataset
    
    except Exception as e:
        common.Print_Error('Dataloader -> Window Dataset', e)
