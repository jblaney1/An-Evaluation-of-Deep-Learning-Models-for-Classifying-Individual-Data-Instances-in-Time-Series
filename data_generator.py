"""
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
Implements the statistical synthetic data generation algorithm described in the 
report "Perturbation Analysis for Creating Synthetic Data.pdf" as well as insertion
of gaussian noise into any provided sequence.

Functions:
 - augment_data(features, labels, mean, std, randomize=False)
 - generate_data(statistics, header='Depth mm', window_width=5, severity=0.0, std_normalizer=5)
 - generate_perturbances(statistics, header, severity, std_normalizer, section)
 - generate_section(mu, std, length)
 - perturb(statistics, header, std_normalizer, length, index_list)
 - window_average(data, window_width)

Included with: 
 - ann.py 
 - ann_tester.py
 - common.py 
 - data_generator.py (current file)
 - dataloader.py
 - gan_tester.py
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
import numpy as np
import pandas as pd
    
# In House Dependencies



"""
    augment_data(features, labels, mean, std)
    A function to automate data augmentation through adding gaussian noise to specific classes.
    The class can either be randomly chosen or static by changing class_index.
    
inputs:
    - features (numpy array): The input features to be augmented
    - labels (numpy array): The corresponding labels for features
    - mean (numpy array): The mean value for each class
    - std (numpy array): The standard deviation for each class
outputs:
    - features (numpy array): The inplace augmented feature vector
"""
def augment_data(features, labels, mean, std, randomize=False):
    
    if randomize:
        class_index = np.random.randint(low=0,high=3)
    else:
        class_index = 1
        
    percent = np.random.randint(low=0, high=50) / 100
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
        index = np.random.randint(low=min_index, high=max_index)
        features[index] += np.random.normal(mu, sigma) / 100
        
    return features


"""
    generate_data(statistics, header='Depth mm', window_width=5, severity=0.0, std_normalizer=5)
    A function to generate one sequence of synthetic data from start to finish with the header
    specifying which columns to generate
    
inputs:
    - statistics (dict): A dictionary created by stats_class which has the statistics from all available files
    - header (string): The name of the column to generate the data for. 
    - window_width (int): The width of the averaging window used by window_average()
    - severity (float): The percent of class 1 points to perturb
    - std_normalizer (real): A normalizer used to prevent zero length class sections when std is large
outputs:
    - data (DataFrame): A sequence of synthetic data ready to be used by an RNN.
"""
def generate_data(statistics, header='Depth mm', window_width=5, severity=0.0, std_normalizer=5):
    labels = []
    for index in range(len(statistics['class length mean'])):
        length = int(np.random.normal(statistics['class length mean'][index], statistics['class length std'][index]/std_normalizer))
        section = generate_section(statistics[header + ' velo mean'][index], statistics[header + ' velo std'][index]/std_normalizer, length)
        if index == 0:
            depth = section
        else:
            max_depth = max(depth)
            if index == 1:
                section = generate_perturbances(statistics, header, severity, std_normalizer, section)
            for j, entry in enumerate(section):
                section[j] = entry + max_depth
            depth += section
        labels += (np.ones(len(section)).astype('int64') * index).tolist()
        
    depth = window_average(depth, window_width)
    force = np.ones(len(depth))
    data = {'Depth mm':depth, 'Force lbf':force, 'Label':labels[int(window_width/2):-window_width*2]}
    data = pd.DataFrame(data)
    return data


"""
    generate_perturbances(statistics, header, severity, std_normalizer, section)
    A function to automatically generate sequences of perturbances according to the 
    specified severity.
    
inputs:
    - statistics (dict): A dictionary created by stats_class which has the statistics from all available files
    - header (string): The name of the column to generate the data for. 
    - severity (int): The percent of class 1 points to perturb
    - std_normalizer (real): A normalizer used to prevent zero length class sections when std is large
    - section (list): The input list of values to add perturbations to
outputs:
    - section (list): The in place altered list of values with perturbed values
"""
def generate_perturbances(statistics, header, severity, std_normalizer, section):
    section_length = len(section)
    perturbances = int(severity*4) if severity >= 0 else 0
    perturbance_start = int(np.random.uniform(0.0, 1.0 - severity) * section_length) if severity < 1.0 else 0
    stop_index = perturbance_start
    
    for perturbance in range(0,perturbances):
        perturbance_length = int(np.random.normal(50,25))
        start_index = stop_index
        stop_index = int(start_index + perturbance_length)
        index_list = [0,1] if start_index/section_length <= 0.5 else [2,1]
        perturbed_section = perturb(statistics, header, std_normalizer, perturbance_length, index_list)

        for index in range(perturbance_length):
            perturbed_section[index] += section[start_index]
            
        if stop_index < section_length:
            section[start_index:stop_index] = perturbed_section
        elif start_index < section_length:
            length = len(section) - start_index
            section[start_index:] = perturbed_section[:length]
            section += perturbed_section[length:]
            stop_index -= 1
        else:
            section += perturbed_section
            stop_index -= 1
        
    return section


"""
    generate_section(mu, std, length)
    A function to generate a straight line with length points and a randomly selected slope 
    centered on mu with standard deviation std.
    
inputs:
    - mu (float): The mean to center the distribution of the slope
    - std (float): The standard deviation of the slope from mu
    - length (int): The number of evenly spaced points to generate
outputs:
    - y (list): The list of generated y points
"""
def generate_section(mu, std, length):
    y = []
    m = abs(np.random.normal(mu, std))
    
    for x in range(length):
        y.append(m*x)
        
    return y


"""
    perturb(statistics, header, std_normalizer, length, index_list)
    A function used to add perturbances to the second class (class 1) of the synthetic data.
    These perturbances are designed to mimic examples drawn from the training dataset which 
    exhibit perturbances in both the depth and force graphs.
    
inputs:
    - statistics (dict): A dictionary created by stats_class which has the statistics from all available files
    - header (string): The name of the column to generate the data for. 
    - std_normalizer (real): A normalizer used to prevent zero length class sections when std is large
    - length (int): The length of the perturbance sequence
    - index_list (list): A list of the class statistics to use for each section
outputs:
    - section (list): The in place altered list of values with perturbed values
"""
def perturb(statistics, header, std_normalizer, length, index_list):
    split = np.random.uniform(0.25,0.75)
    section1_length = int(split * length)
    section2_length = int(length - section1_length)        
    section1 = generate_section(statistics[header + ' velo mean'][index_list[0]], statistics[header + ' velo std'][index_list[0]]/std_normalizer, length=section1_length)
    section2 = generate_section(statistics[header + ' velo mean'][index_list[1]], statistics[header + ' velo std'][index_list[1]]/std_normalizer, length=section2_length)
    
    for index, entry in enumerate(section2):
        section2[index] = entry + max(section1)
        
    return section1 + section2


"""
    window_average(data, window_width)
    A function to perform windowed averaging over an array like structure. Windowed averaging can
    also be thought of as simple smoothing with larger windows smoothing more than small windows.
    
inputs:
    - data (1xn array): The input data which will be averaged. Can be an array, list, or similar struct
    - window_width (int): The width of the window to use for averaging. Larger windows smooth more.
outputs:
    - averaged_data (): The averaged data restricted to points which were smoothed.
"""
def window_average(data, window_width):
    averaged_data = data
    windows = range(window_width-1, len(data)-window_width-1)
    window_width_2 = int(window_width / 2)
    
    for window in windows:
        i = window - window_width_2
        j = window + window_width_2
        averaged_data[i] = sum(data[i:j]) / window_width
        
    return averaged_data[window_width_2:-window_width*2]