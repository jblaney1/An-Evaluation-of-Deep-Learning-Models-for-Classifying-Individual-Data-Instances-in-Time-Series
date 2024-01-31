""" 
Project: 
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
Provides a single file to find any custom metric implementations.

Functions:
 - custom_metrics(name, pred, targets)
 - mean_absolute_error(preds, targets)
 - mean_absolute_percentage_error(preds, targets)
 - mean_squared_error(preds, targets)
 - root_mean_squared_error(preds, targets)

Included with: 
 - ann.py 
 - ann_tester.py
 - common.py
 - data_generator.py
 - dataloader.py
 - gan_tester.py
 - labeler.py
 - metrics.py (current file)
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
import numpy as np

# In House Dependencies
import common


__name__ = 'metrics'


"""
    check_for_batch(preds, targets)
    A function to check if the input predictions and targets are batched.
    If they are not, then the arrays are put into a batch of size 1 and 
    returned.
    
    inputs:
     - preds (array): The models predictions
     - targets (array): The expected values or labels
    outputs:
     - preds (array): The batched or unaltered prediction array
     - targets (array): The batched or unaltered prediciton array
"""
def check_for_batch(preds, targets):
    batched = len(preds.squeeze().shape) > 1
    preds = preds if batched else np.reshape(preds, (1, preds.squeeze().shape[0]))
    targets = targets if batched else np.reshape(targets, (1, targets.squeeze().shape[0]))
    return preds, targets
    

"""
    custom_metrics(name, pred, targets)
    A function to allow calling custom metrics during training. If you want to
    make a new metric available during training, this is the place to do it. 
    The list of currently available metrics is below. For the exact implementation 
    see the metric's function.
   
    inputs:
     - name (string): The name of the metric to use
     - pred (array): The models predictions
     - targets (array): The labels or expected values
    outputs:
     - metric (float): The resultant metric, None if error occured
"""
def custom_metrics(name, preds, targets):
    try:
        preds, targets = check_for_batch(preds, targets)
        available_metrics = ['Mean Absolute Error (mae)', 
                             'Mean Squeare Error (mse)',
                             'Root Mean Square Error (rmse)',
                             'Mean Absolute Percentage Error (mape)',
                             'Dynamic Time Warping (dtw)',
                             'Euclidean Distance (ed)']

        if name == 'mae':
            metric = mean_absolute_error(preds, targets)

        elif name == 'mse':
            metric = mean_squared_error(preds, targets)

        elif name == 'rmse':
            metric = root_mean_squared_error(preds, targets)

        elif name == 'mape':
            metric = mean_absolute_percentage_error(preds, targets)
            
        elif name == 'dtw':
            metric = dynamic_time_warping(preds, targets)
        
        elif name == 'ed':
            metric = euclidean_distance(preds, targets)
        
        elif name == 'fc':
            metric = fourier_coefficients(preds, targets)
        
        else:
            metric = None
            e = f'Undefined metric name recieved <{name}>\n'
            e += f'The abailable metrics are:\n{available_metrics}'
            common.Print_Error('Trainer -> custom metrics',f'Undefined metric name recieved <{name}>\n')

        return metric

    except Exception as e:
        common.Print_Error(f'Trainer -> custom metrics <{name}>', e) 
        
        
"""
    dynamic_time_warping(preds, targets)
    A function to implement dynamic time warping (DTW), a standard
    similarity measure used within the field of time series
    analysis and often the baseline when combined with 
    K-nearest neighbor algorithms. If the inputs are batched, it 
    is assumed that the sequence lengths as constant for the 
    predictions and the targets, but it is not assumed that the 
    sequence lengths are equal for predictions and targets. 
    The definition of DTW can be found at the link below.
    
    https://en.wikipedia.org/wiki/Dynamic_time_warping
    
    inputs:
     - preds (array): The models predictions
     - targets (array): The expected values or labels
     - window_size (int): The size of the window to consider when warping
    outputs:
     - (float): The resulting DTW score
"""
def dynamic_time_warping(preds, targets, window_size=4):

    loss = np.ones((preds.shape[-1], targets.shape[-1]))*1e9
    losses = np.ones((preds.shape[0],)) if len(preds.shape) > 1 else np.zeros((1,))
    window_size = None if window_size is None else max(window_size, abs(preds.shape[-1]-targets.shape[-1]))

    for i in range(losses.shape[0]):
        loss[0,0] = 0.0
        pred = preds[i]
        target = targets[i]
        pred_count = len(pred)
        targ_count = len(target)
        normalizer = pred_count * targ_count
        
        for j in range(1, pred_count):
            top_index = targ_count if window_size is None else min(targ_count, j+window_size)
            bottom_index = 1 if window_size is None else max(1, j-window_size)
            
            for k in range(bottom_index, top_index):
                index_diff = (j-k)*(j-k)/normalizer
                point_diff = (pred[j]-target[k])*(pred[j]-target[k])/4096
                distance = np.sqrt(index_diff + point_diff)
                loss[j,k] = distance + min(loss[j-1, k], loss[j, k-1], loss[j-1, k-1])
        
        losses[i] = loss[-1,-1]
        
    return losses
    

"""
    euclidean_distance()
    A function to compute the euclidean distance between two sequences as is
    commonly defined in time series classification, not actual euclidean
    distance between all points. For more information on the algorithm see 
    reference [1].
    
    [1] Serra J and Acros J, An Empirical Evaluation of Similarity Measures for 
        Time Series Classification, Knowledge-Based Systems Vol 67 2014
    
    inputs:
     - preds (array): The models predictions
     - targets (array): The expected values or labels
    ouputs:
     - (float): The euclidean distance between the predictions and the targets
"""
def euclidean_distance(preds, targets, p=2):
    difference = preds - targets
    power = np.emath.power(difference, p)
    summation = np.sum(power)
    return np.emath.power(summation, 1/p)


"""
    fourier_coefficients(preds, targets)
    A function to compute the distance between fourier coefficients as is 
    commonly done in time series classification. Can be spedup by including
    filtering functionality, but not implemented here. When performed without 
    filtering approximates the euclidean distance. For more information on the 
    algorithm see reference [1].
    
    [1] Serra J and Acros J, An Empirical Evaluation of Similarity Measures for 
        Time Series Classification, Knowledge-Based Systems Vol 67 2014
    
    inputs:
     - preds (array): The models predictions
     - targets (array): The expected values or labels
    ouputs:
     - (float): The euclidean distance between the predictions and the targets
"""
def fourier_coefficients(preds, targets):
    preds_dft = np.fft.fft(preds)
    targets_dft = np.fft.fft(targets)
    return euclidean_distance(preds_dft, targets_dft, 2)
    

"""
    mean_absolute_error(preds, targets)
    A function to implement the mae function found at the link below.

    https://en.wikipedia.org/wiki/Mean_absolute_error

    inputs:
     - preds (array): The models predictions
     - targets (array): The expected values or labels
    ouputs:
     - (float): The mae between the predictions and the targets
"""
def mean_absolute_error(preds, targets):
    return sum(abs(preds - targets)) / len(preds)


"""
    mean_absolute_percentage_error(preds, targets)
    A function to implement the mape function found at the link below.

    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    inputs:
     - preds (array): The models predictions
     - targets (array): The expected values or labels
    outputs:
     - (float): The mape between the predictions and the targets
"""
def mean_absolute_percentage_error(preds, targets):
    mape = 0
    num_preds = len(preds)
    difference = preds - targets

    for index in range(num_preds):
        mape += abs(difference(index)/targets(index))

    return mape / num_preds


"""
    mean_squared_error(preds, targets)
    Implements the MSE function found at the link below.

    https://en.wikipedia.org/wiki/Mean_squared_error

    inputs:
     - preds (array): The models predictions
     - targets (array): The expected values or labels
    outputs:
     - (float): The mse between the predictions and the targets
"""
def mean_squared_error(preds, targets):
    difference = preds - targets
    return np.dot(difference, difference) / len(preds)


"""
    root_mean_squared_error(preds, targets)
    A function to implement the rmse function found at the link below.

    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    inputs:
     - preds (array): The models predictions
     - targets (array): The expected values or labels
    outputs:
     - (float): The rmse between the predictions and the targets
"""
def root_mean_squared_error(preds, targets):
    mse = mean_squared_error(preds, targets)
    return np.sqrt(mse)