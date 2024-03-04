"""
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
A file to provide statistical analysis functionality for processing data from
time series spread accross multiple/many files.

Functions:
 - correlation(file_list, plot=False, save=False, file_path=None)
 - histogram(file_list, max_dict=None, min_dict=None, buckets=100, headers=None, plot=False, save=False, save_path=None)
 - length(file_list, compute_stats=False, save=False, save_path=None)
 - maximum_minimum(file_list, headers=None, output_type=list, save=False, save_path=None)
 - mean_std(file_list, headers=None, average=False, save=False, save_path="../Stats/")
 - point_count(file_list, save=False, save_path="../Stats/")
 - print_stats(file_path)

 stats()
 - stats(name='default', file_list=None, headers=None, overwrite=False, save=False, save_path=None, label_header=None)
 - attribute_get(purpose, name)
 - attribute_set(name, value)
 - correlation(plot=False)
 - histogram(labels=[0,1,2,3], max_dict=None, min_dict=None, buckets=100, plot=False, use_matplotlib=False)
 - length_class(compute_stats=False)
 - maximum_minimum(output_type)
 - mean_std_class(average=False)
 - point_count()
 - validate_inputs_class(save_path, label_header)

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
 - stats.py (current file)
 - trainer.py
 - transitioner.py
 
Notes:
For more information about the project contact 
 - Dr. Suresh Muknahallipatna -> sureshm@uwyo.edu
 - Josh Blaney -> jblaney1@uwyo.edu
"""

# Outside Dependencies
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# In House Dependencies
import common
import dataloader as dl
    

"""
    correlation(file_list, plot=False, save=False, file_path=None)
    A function to compute the average correlation matrix for the files included
    in file list. No attempts are made to ensure the correlation matrices are
    the same shape. It is up to you to ensure that all of your data is in the 
    same input feature space!
    
    inputs:
     - file_list (list): A list of files to process correlation matrices for
     - plot (bool): Should a plot be generated?
     - save (bool): Should the plot be saved?
     - file_path (string): Where should the plot be saved?
    outputs:
     - correlation_matrix (np array): The average covariance matrix 
"""
def correlation(file_list, headers=None, plot=False, save=False, file_path=None):
    assert type(file_list) is list, 'file_list must be of type <list>'
    
    scaler = 4
    file_count = len(file_list)
    df = pd.read_csv(file_list[0])
    
    if headers is None:
        df = df.drop(columns='index') if 'index' in df.keys() else df
        df = df.drop(columns='Unnamed: 0') if 'index' in df.keys() else df
        headers = df.keys()
        
    correlation_matrix = df[headers].corr()*0.0
    
    for index, file in enumerate(file_list):
        try:
            df = pd.read_csv(file)[headers].astype(float)
            correlation = df.corr(numeric_only=True)
            correlation_matrix += 0.0 if correlation.isnull().values.any() else correlation
            common.Print_Status('Correlation', index, file_count)
            
        except Exception as e:
            common.Print_Error('Stats -> correlation', f'File: {file}\n{e}')
        
    correlation_matrix /= file_count
    
    if save:
        common.Validate_Dir(file_path)
        
    if plot:
        fig, ax = plt.subplots(figsize=(4*scaler,3*scaler))
        sns.heatmap(correlation_matrix, 
                    vmin=-1, 
                    vmax=1, 
                    cmap=sns.color_palette("mako", as_cmap=True), 
                    annot=True, 
                    linecolor='w', 
                    linewidths=1)
        
        if save:
            file = f'{file_path}/correlation.png'
            plt.savefig(file)
    
    return correlation_matrix
    
    
"""
    histogram(file_list, max_dict=None, min_dict=None, buckets=100, headers=None, plot=False, save=False, save_path=None)
    A function which iterates through the files in file_list and creates histograms
    for each header in the files.
    
    inputs:
     - file_list (list): The list of file names to process
     - max_dict (dict): A dictionary with expected headers for keys and floats for entries
     - min_dict (dict): A dictionary with expected headers for keys and floats for entries
     - buckets (int): The number of buckets to use
     - headers (list): A list of headers to compute histograms for
     - plot (bool): Should plots be generated?
     - save (bool): Should the plots be saved?
     - file_path (string): Where should the plots be saved?
    outputs:
     - histograms (dict): A dictionary of histograms with headers for keys
"""
def histogram(file_list, max_dict=None, min_dict=None, buckets=100, headers=None, plot=False, save=False, save_path=None, use_matplotlib=True):
    assert type(buckets) == int, 'buckets must be of type <int>'
    
    file_count = len(file_list)
    
    if headers is not None:
        assert type(headers) == list, 'headers must be of type <list>'
    else:
        file = file_list[0]
        df = pd.read_csv(file)
        df = df.drop(columns='index') if 'index' in df.keys() else df
        df = df.drop(columns='Unnamed: 0') if 'Unnamed: 0' in df.keys() else df
        
        headers = list(df.keys())
        
    if use_matplotlib:
        histograms = {}
        bucket_size = {}
        
        compute_min_max = True if max_dict == None and min_dict == None else False
        
        if compute_min_max:
            max_dict, min_dict = maximum_minimum(file_list, headers=headers, output_type=dict, save=save, save_path=save_path)
            
        for header in headers:
            histograms[header] = {}
            bucket_size[header] = (max_dict[header] - min_dict[header])/buckets
            
            for i in range(buckets):
                histograms[header][i] = 0
    
        for i, file in enumerate(file_list):
            try:
                df = pd.read_csv(file)[headers]
                
                for header in headers:
                    column = df[header]
                    column = (column - min_dict[header]) / bucket_size[header]
                    column = column.astype(int)
                
                    for entry in column:
                        if entry in histograms[header].keys():
                            histograms[header][entry] += 1
                    
                common.Print_Status('Histogram', i, file_count)    
                
            except Exception as e:
                common.Print_Error('Stats -> Histogram', f'File: {file}\n{e}')
                return None
        
        if save:
            common.Validate_Dir(save_path)
        
        if plot:
            for header in histograms.keys():
                fig, ax = plt.subplots()
                height = []
                x = list(histograms[header].keys())
                
                for i, entry in enumerate(x):
                    height.append(histograms[header][entry])
                    x[i] = entry * bucket_size[header] + min_dict[header]
                    
                ax.bar(x, height)
                ax.title.set_text(header)
                
                if save:
                    try:
                        header = header.replace('/','-') if '/' in header else header
                        file = f'{save_path}/mat-{header}.png'
                        plt.savefig(fname=file)
                        
                    except Exception as e:
                        common.Print_Error('Stats -> Histogram', f'File: {file}\n{e}')
                        
        return histograms
    
    else:
        stat_data = pd.read_csv(file_list[0])[headers]
        
        for index in range(1,file_count):
            df = pd.read_csv(file_list[index])[headers]
            if not df.isnull().values.any():
                stat_data = pd.concat([stat_data, df])
                common.Print_Status("Histogram -> Stack Data", index, file_count)
        
        if plot:
            for header in headers:
                fig, ax = plt.subplots()
                sns.histplot(data=stat_data, 
                             x=header, 
                             bins=buckets, 
                             kde=True, 
                             ax=ax)
                
                if save:
                    try:
                        header = header.replace('/','-') if '/' in header else header
                        file = f'{save_path}/sns-{header}.png'
                        plt.savefig(fname=file)
                        
                    except Exception as e:
                        common.Print_Error('Stats -> Histogram', f'File: {file}\n{e}')
        
    
"""
    length(file_list, compute_stats=False, save=False, save_path=None)
    Finds the longest file in the provided filelist and returns the length or 
    finds the longest and shortest files, and computes the mean and std of the 
    file lengths for the dataset
    
inputs: 
    - file_list (list): A list of files to process, this list should not be globbed.
    - compute_stats (bool): Should min, mean, and std be computed also?
    - save (bool): Should the results be saved?
    - save_path (string): Where should the results be saved?
outputs:
    - stats (list): The requested statistics
"""
def length(file_list, compute_stats=False, save=False, save_path=None):
    try:        
        file_count = len(file_list)
        
        stats = []
        for index in range(file_count):
            df = pd.read_csv(file_list[index])
            stats.append(df.shape[0])
            common.Print_Status("Length", index, file_count)
        
        if compute_stats:
            
            stats = {'max length': [max(stats)], 
                     'min length': [min(stats)],
                     'mean length': [np.mean(stats)],
                     'std length': [np.std(stats)]}
            
            file_name = 'length-stats.csv'
        else:
            stats = {'max length': [max(stats)]}
            file_name = 'length.csv'
        
        if save:
            df = pd.DataFrame(data=stats)
            df.to_csv(f'{save_path}{file_name}', index=False)
            
        return stats
    
    except Exception as e:
        common.Print_Error('Stats -> length', e)
        

"""
    maximum_minimum(file_list, headers=None, output_type=list, save=False, save_path=None)
    Finds the maximum and minimum along columns of a dataset stored in various 
    csv files. For brevity max and min will be refered to as stats. This function 
    finds the stats of each file, compares them, and returns the most distant 
    outliers in the dataset, for each column, over the entire list of files.
    
inputs:
    - file_list (list): A list of files to process, this list should not be globbed.
    - headers (list): A list of strings denoting the columns to compute stats over
    - output_type (python type): The data type to return, see outputs for more
    - save (bool): Should the intermediate dataframe be saved?
    - save_path (string): The location to save the dataframe to 

outputs:
    - stats (dict): A dict of lists with shape [max, min]
    or
    - max_dict (dict): A dict of maximum floating point values
    - min_dict (dict): A dict of minimum floating point values
    
"""
def maximum_minimum(file_list, headers=None, output_type=list, save=False, save_path=None):
    try:
        file_count = len(file_list)
        
        if headers is not None:
            assert type(headers) == list, 'headers must be of type <list>'
        else:
            file = file_list[0]
            df = pd.read_csv(file)
            df = df.drop(columns='index') if 'index' in df.keys() else df
            df = df.drop(columns='Unnamed: 0') if 'Unnamed: 0' in df.keys() else df
            
            headers = list(df.keys())
        
        max_dict = {}
        min_dict = {}
        for header in headers:
            max_dict[header] = -1e9
            min_dict[header] = 1e9
        
        for i, file in enumerate(file_list):
            try:
                df = pd.read_csv(file)[headers]
                
                for header in headers:
                    column = df[header].values
                    column_min = np.min(column)
                    column_max = np.max(column)
                    max_dict[header] = max_dict[header] if max_dict[header] > column_max else column_max
                    min_dict[header] = min_dict[header] if min_dict[header] < column_min else column_min
            
                common.Print_Status('Compute Max - Min', i, file_count)

            except Exception as e:
                common.Print_Error('Stats -> Histogram', f'File: {file}\n{e}')         
        
        stats = {}
        message = 'Max - Min Results\n'
        
        for header in headers:
            stats[header] = [max_dict[header], min_dict[header]]
            message += f'{header:<15} Max: {max_dict[header]:<8} | Min: {min_dict[header]:<8} \n'
        common.Print(message)
        
        if save:
            common.Validate_Dir(save_path)
            df = pd.DataFrame(stats, index=['max', 'min'], columns=headers)
            df.to_csv(f'{save_path}/max-min.csv')
            
        if output_type == list:
            return stats
        
        elif output_type == dict:
            return max_dict, min_dict
    
    except Exception as e:
        common.Print_Error('Stats -> maximum', e)


"""
    mean_std(file_list, headers=None, average=False, save=False, save_path="../Stats/")
    Finds the mean and standard deviation along columns of a dataset stored in 
    a list of csv files.
    
inputs:
    - file_list (list): A list of files to process, this list should not be globbed.
    - headers (list): A list of strings denoting the columns to compute stats for
    - average (bool): Compute the average mean and std or the global?
    - save (bool): Save the resulting dataframe as a csv?
    - save_path (string): The location to save the dataframe to
outputs:
    - mean(list): A list of the mean values for each column across all input files.
    - std(list): A list of the standard deviation values for each column across all input files.
"""
def mean_std(file_list=None, headers=None, average=False, save=False, save_path="../Stats/"):
    try:         
        
        df = pd.read_csv(file_list[0])
        
        if headers is None:
            headers = list(df[0].keys())
            
        file_count = len(file_list)
        mean = np.zeros((len(headers),))
        std = np.zeros((len(headers),))
        
        if average:
            for index in range(file_count):
                df = pd.read_csv(file_list[index])[headers].astype(float)
                df_mean = df.mean(axis=0)
                df_std = df.std(axis=0)
                
                if not df_mean.isnull().values.any() and not df_std.isnull().values.any():
                    mean += df_mean
                    std += df_std
                    
                common.Print_Status("Mean/STD -> Average", index, file_count)
        
            mean = (mean / file_count).values
            std = (std / file_count).values
            stats = np.array([mean, std])
            file_name = 'mean-std-ave.csv'
            
        else:
            
            stats = np.zeros((2,len(headers)))
            stat_data = pd.read_csv(file_list[0])[headers]
            
            for index in range(1,file_count):
                df = pd.read_csv(file_list[index])[headers]
                if not df.isnull().values.any():
                    stat_data = np.append(stat_data, df, axis=0)
                common.Print_Status("Mean/STD -> Stack Data", index, file_count)
            
            for index in range(len(headers)):
                stats[0,index] = stat_data[:,index].mean()
                stats[1,index] = stat_data[:,index].std()
                common.Print_Status("Mean/STD", index, len(headers))
            
            file_name = 'mean-std.csv'   
            
        if save:
            df = pd.DataFrame(stats, index=["mean", "std"], columns=headers)
            df.to_csv(f'{save_path}{file_name}', index=False)
        
        return mean, std
    
    except Exception as e:
        common.Print_Error('Stats -> mean std', e)


"""
    point_count(file_list, save=False, save_path="../Stats/")
    A function to compute point based statistics over a dataset contained in 
    the files of file_list. The stats computed are total points (in all files), 
    max points (longest file), min points (shortest file), mean points 
    (average file length),
"""
def point_count(file_list, save=False, save_path="../Stats/"):
    try:            
        file_count = len(file_list)
        
        stats = {}
        points = []
        
        for index in range(file_count):
            df = pd.read_csv(file_list[index])
            points.append(df.shape[0])
            
            common.Print_Status('Point Count', index, file_count)

        stats['total points'] = [sum(points)]
        stats['max points'] = [max(points)]
        stats['min points'] = [min(points)]
        stats['mean points'] = [np.mean(points)]
        stats['std points'] = [np.std(points)]
                
        file_name = 'point-stats.csv'
            
        if save: 
            df = pd.DataFrame(stats)
            df.to_csv(f'{save_path}{file_name}', index=False)
        
        return stats
    
    except Exception as e:
        common.Print_Error('Stats Class -> point count', e)
        
        
"""
    print_stats(file_path)
    A function to streamline the printing process for the stats stored in a csv
    at the file path
    
    inputs:
     - file_path (string): Location of the stats csv file
"""
def print_stats(file_path):
    try:
        stat = {}
        file_list = glob.glob(file_path)
        file_count = len(file_list)
        stat['max'] = maximum_minimum(file_list)
        stat['mean/std'] = mean_std(file_list)
        output = ""
        
        entry_count = 0
        
        for index, file in enumerate(file_list):
            entry_count += len(pd.read_csv(file).values)
            common.Print_Status('Counting', index, file_count)
        
        stat['count'] = entry_count
        
        for key in stat.keys:
            output += f'[INFO] {key} : {stat[key]}'
            
    except Exception as e:
        common.Print_Error('Stats -> print stats', e)
    
  
"""
    stats(name='default', file_list=None, headers=None, overwrite=False, save=False, save_path=None, label_header=None)
    A class used to compute and store useful statistics about the dataset. 
    Automatically looks for an existing statistics csv file which may be loaded 
    instead of recomputing the statistics. Unlike the function above, all 
    functionality in this class is focused on processing labeled data and 
    outputing information about the data for each label.
    
inputs:
    - name (string): Used to differentiate between stats objects
    - file_list (list): The list of files to process
    - headers (list): The list of headers to process
    - overwrite (bool): Should the stats files be overwritten if present?
    - save (bool): Should the stats dataframe be saved after it is created?
    - save_path (string): Where should the stats file be saved to?
    - label_header (string): The column to use for differentiating between labels
outputs
    -
"""
class stats:
    
    def __init__(self, name='default', file_list=None, headers=None, overwrite=True, save=True, save_path=None, label_header=None):
        
        headers = ['Depth mm', 'Force lbf'] if headers is None else headers
        
        if file_list is not None:
            self.file_list = file_list if type(file_list) is list else glob.glob(file_list)
            self.file_count = len(self.file_list)
            self.data = dl.Preload_Data(file_list, headers=headers+[label_header])
        else:
            self.file_list = None
            self.file_count = 0
            self.data = None
        
        self.report = {}
        self.attribute_set('name', name)
        self.attribute_set('file list', file_list)
        self.attribute_set('file count', self.file_count)
        self.attribute_set('headers', headers)
        self.attribute_set('overwrite', overwrite)
        self.attribute_set('save', save)
        self.attribute_set('save path', save_path)
        self.attribute_set('label header', label_header)
        self.stats = {}
    
    
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
        correlation(plot=False)
        A function to compute the average correlation matrix for the files included
        in file list. No attempts are made to ensure the correlation matrices are
        the same shape. It is up to you to ensure that all of your data is in the 
        same input feature space!
        
        inputs:
         - plot (bool): Should a plot be generated?
        outputs:
         - correlation_matrix (np array): The average covariance matrix 
    """
    def correlation(self, plot=False):
        
        if self.file_count > 0:
            save = self.attribute_get('save')
            save_path = self.attribute_get('save path')
            overwrite = self.attribute_get('overwrite')
            
            headers = self.attribute_get('headers')
            label_header = self.attribute_get('label header')
            save_path, label_header = self.validate_inputs_class(save_path, label_header)
            
            if common.Validate_File(save_path) and not overwrite:
                print("[INFO] Found an existing stats-class.csv file at: " + save_path)
                stats = pd.read_csv(save_path)
                print(stats)
                self.stats = stats
                return stats
        else:
            return None
        
        scaler = 4
        headers_to_keep = headers + [label_header]
        file_count = len(self.file_list)
        df = pd.read_csv(self.file_list[0])[headers_to_keep]
        correlation_matrix = df.corr()*0.0
        
        for index, file in enumerate(self.file_list):
            try:
                df = pd.read_csv(file)[headers_to_keep]                
                correlation = df.corr(numeric_only=True).fillna(value=0.0)
                correlation_matrix += correlation
                    
                common.Print_Status('Correlation', index, file_count)
                
            except Exception as e:
                common.Print_Error('Stats -> correlation', f'File: {file}\n{e}')
        
        correlation_matrix /= file_count
        
        if save:
            common.Validate_Dir(save_path)
            
        if plot:
            fig, ax = plt.subplots(figsize=(4*scaler,3*scaler))
            sns.heatmap(correlation_matrix, 
                        vmin=-1, 
                        vmax=1, 
                        cmap=sns.color_palette("mako", as_cmap=True), 
                        annot=True, 
                        linecolor='w', 
                        linewidths=1)
            if save:
                file = f'{save_path}/correlation.png'
                plt.savefig(file)
        
        return correlation_matrix
        
        
    """
        histogram(labels=[0,1,2,3], max_dict=None, min_dict=None, buckets=100, plot=False, use_matplotlib=False)
        A function which iterates through the files in file_list and creates histograms
        for each header in the files.
        
        inputs:
         - labels (list): The number of expected classes to divide the histograms into
         - max_dict (dict): A dictionary with expected headers for keys and floats for entries
         - min_dict (dict): A dictionary with expected headers for keys and floats for entries
         - buckets (int): The number of buckets to use
         - plot (bool): Should plots be generated?
         - use_matplotlib (bool): Use matplotlib for ploting? or seaborn?
        outputs:
         - histograms (dict): A dictionary of histograms with headers for keys
    """
    def histogram(self, labels=[0,1,2,3], max_dict=None, min_dict=None, buckets=100, plot=False, use_matplotlib=False):
        
        if self.file_count > 0:
            save = self.attribute_get('save')
            save_path = self.attribute_get('save path')
            overwrite = self.attribute_get('overwrite')
            
            headers = self.attribute_get('headers')
            label_header = self.attribute_get('label header')
            save_path, label_header = self.validate_inputs_class(save_path, label_header)
            
            if common.Validate_File(save_path) and not overwrite:
                print("[INFO] Found an existing stats-class.csv file at: " + save_path)
                stats = pd.read_csv(save_path)
                print(stats)
                self.stats = stats
                return stats
        else:
            return None
        
        assert type(buckets) == int, 'buckets must be of type <int>'
        
        file_count = len(self.file_list)
        
        if labels is None:
            df = self.data[0]
            labels = df[label_header].unique()
            
        if use_matplotlib:
            histograms = {}
            bucket_size = {}
            
            compute_min_max = True if max_dict == None and min_dict == None else False
            
            if compute_min_max:
                max_dict, min_dict = self.maximum_minimum(output_type=dict)
            
            for header in headers:
                histograms[header] = {}
                bucket_size[header] = {}
                for label in labels:
                    histograms[header][label] = {}
                    bucket_size[header][label] = (max_dict[header][label] - min_dict[header][label])/buckets
                
                    for i in range(buckets):
                        histograms[header][label][i] = 0
    
            for i, file in enumerate(self.file_list):
                try:
                    df = pd.read_csv(file).fillna(value=0.0)
                    
                    for header in headers:
                        for label in labels:
                            column = df[df[label_header] == label][header]
                            column = (column - min_dict[header][label]) / bucket_size[header][label]
                            column = column.astype(int)
                        
                            for entry in column:
                                if entry in histograms[header][label].keys():
                                    histograms[header][label][entry] += 1
                        
                    common.Print_Status('Histogram', i, file_count)    
                    
                except Exception as e:
                    common.Print_Error('Stats -> Histogram', f'File: {file}\n{e}')
                    return None
            
            if save:
                common.Validate_Dir(save_path)
            
            if plot:
                c = {0: 'red', 1:'blue', 2:'green', 3:'black'}
                
                for header in histograms.keys():
                    fig_header, ax_header = plt.subplots()
                    
                    for i, label in enumerate(labels):
                        fig_label, ax_label = plt.subplots()
                        height = []
                        x = list(histograms[header][label].keys())
                    
                        for j, entry in enumerate(x):
                            height.append(histograms[header][label][entry])                        
                            x[j] = entry * bucket_size[header][label] + min_dict[header][label]
                        
                        ax_label.bar(x, height, color=c[i], alpha=0.32)
                        ax_label.title.set_text(f'{header}-{label}')
                        
                        if save:
                            try:
                                file_header = header.replace('/','-') if '/' in header else header
                                file = f'{save_path}/mat-class-{file_header}-{label}.png'
                                fig_label.savefig(fname=file)
                            
                            except Exception as e:
                                common.Print_Error('Stats -> Histogram', f'File: {file}\n{e}')
                        
                        ax_header.bar(x, height, color=c[i], label=label, alpha=0.32)
                        
                    ax_header.title.set_text(f'{header}')
                    ax_header.legend()
                    
                    if save:
                        try:
                            file_header = header.replace('/','-') if '/' in header else header
                            file = f'{save_path}/mat-class-{file_header}.png'
                            fig_header.savefig(fname=file)
                        
                        except Exception as e:
                            common.Print_Error('Stats -> Histogram', f'File: {file}\n{e}')
                
            return histograms
        
        else:
            headers_to_keep = headers + [label_header]
            stat_data = self.data[0][headers_to_keep]
            
            for index in range(1,file_count):
                df = self.data[index][headers_to_keep]
                if not df.isnull().values.any():
                    stat_data = pd.concat([stat_data, df])
                    common.Print_Status("Histogram -> Stack Data", index, file_count)
            
            if plot:
                if min(stat_data[label_header]) == 0:
                    stat_data[label_header] += 1
                    
                for header in headers:
                    fig, ax = plt.subplots()
                    sns.histplot(data=stat_data, 
                                 x=header, 
                                 bins=buckets, 
                                 kde=True, 
                                 hue=label_header, 
                                 palette=sns.color_palette("muted", as_cmap=True)[0:len(labels)],
                                 ax=ax)
                    
                    if save:
                        try:
                            header = header.replace('/','-') if '/' in header else header
                            file = f'{save_path}/sns-class-{header}.png'
                            plt.savefig(fname=file)
                            
                        except Exception as e:
                            common.Print_Error('Stats -> Histogram', f'File: {file}\n{e}')


    """
        length_class(compute_stats=False)
        Finds useful stats about the number of points in each class
    
    inputs: 
        - compute_stats (bool): Should the min, mean, and std be computed? or just max?
    outputs:
        - stats (dict): The dictionary of computed stats
    """
    def length_class(self, compute_stats=False):
        try:
            if self.file_count > 0:
                save = self.attribute_get('save')
                save_path = self.attribute_get('save path')
                overwrite = self.attribute_get('overwrite')
                
                label_header = self.attribute_get('label header')
                save_path, label_header = self.validate_inputs_class(save_path, label_header)
                
                if common.Validate_File(save_path) and not overwrite:
                    print("[INFO] Found an existing stats-class.csv file at: " + save_path)
                    stats = pd.read_csv(save_path)
                    print(stats)
                    self.stats = stats
                    return stats
                
                class_data = {}
                stats = {}
                
                df = self.data[0]
                labels = df[label_header].unique()
                label_count = len(labels)
                
                headers = ['max length', 'min length', 'mean length', 'std length'] if compute_stats else ['max length']
                
                for header in headers:
                    stats[header] = [0]*label_count
                    
                for label in labels:
                    class_data[label] = []
                
                for i in range(self.file_count):
                    try:
                        df = self.data[i]
                        labels = df[label_header].unique()
                        
                        for label in labels:
                            length_val = df[df[label_header] == label].shape[0]
                            class_data[label].append(length_val)
                    
                    except Exception as e:
                        common.Print_Error("Length Class", f'File: {self.file_list[i]}\n{e}')
                    
                    common.Print_Status("Length Class", i, self.file_count)
                
                if compute_stats:
                    for index, label in enumerate(labels):
                        stats[headers[0]][index] = max(class_data[label])
                        stats[headers[1]][index] = min(class_data[label])
                        stats[headers[2]][index] = np.mean(class_data[label])
                        stats[headers[3]][index] = np.std(class_data[label])
                    
                    file_name = 'length-stats-class.csv'
                        
                else:
                    for index, label in enumerate(labels):
                        stats[headers[0]] = max(class_data[label])
                    
                    file_name = 'length-class.csv'
                
                for key in stats.keys():
                    self.stats[key] = stats[key]
                
                df = pd.DataFrame(self.stats)
                if save: df.to_csv(f'{save_path}{file_name}', index=False)
                
                return stats
            else:
                message = "File list is empty"
                common.Print_Error('length - class', message)
                return None
            
        except Exception as e:
            common.Print_Error('Stats Class -> length class', e)
    
    
    """
        maximum_minimum(output_type)
        Finds the maximum and minimum along columns of a dataset stored in various 
        csv files. For brevity max and min will be refered to as stats. This function 
        finds the stats of each file, compares them, and returns the most distant 
        outliers in the dataset, for each column, over the entire list of files.
        
    inputs:
        - output_type (python type): The data type to output, see outputs for more

    outputs:
        - stats (dict): A dict of lists with shape [max, min]
        or
        - max_dict (dict): A dict of maximum floating point values
        - min_dict (dict): A dict of minimum floating point values
        
    """
    def maximum_minimum(self, output_type=dict):
        try:
            if self.file_count > 0:
                save = self.attribute_get('save')
                save_path = self.attribute_get('save path')
                overwrite = self.attribute_get('overwrite')
                
                headers = self.attribute_get('headers')
                label_header = self.attribute_get('label header')
                save_path, label_header = self.validate_inputs_class(save_path, label_header)
                
                if common.Validate_File(save_path) and not overwrite:
                    print("[INFO] Found an existing stats-class.csv file at: " + save_path)
                    stats = pd.read_csv(save_path)
                    print(stats)
                    self.stats = stats
                    return stats
                
                df = self.data[0]
                labels = df[label_header].unique()
                label_count = len(labels)
                
                max_dict = {}
                min_dict = {}
                for header in headers:
                    max_dict[header] = {}
                    min_dict[header] = {}
                    
                    for label in labels:
                        max_dict[header][label] = -1e9
                        min_dict[header][label] = 1e9
                        
                for i, file in enumerate(self.file_list):
                    try:
                        df = pd.read_csv(file).fillna(value=0.0)
                        
                        for header in headers:
                            for label in labels:
                                column = df[df[label_header] == label][header].values
                                
                                if column.shape[0] > 0:
                                    column_min = np.min(column)
                                    column_max = np.max(column)
                                
                                    max_dict[header][label] = max_dict[header][label] if max_dict[header][label] > column_max else column_max
                                    min_dict[header][label] = min_dict[header][label] if min_dict[header][label] < column_min else column_min
                    
                        common.Print_Status('Compute Max - Min', i, self.file_count)

                    except Exception as e:
                        common.Print_Error('Stats Class -> maximum-minimum', f'File: {file}\n{e}')         
                
                stats = {}
                message = 'Max - Min Results\n'
                for header in headers:
                    message += f'{header}\n'
                    stats[f'{header} max'] = [0.0]*label_count
                    stats[f'{header} min'] = [0.0]*label_count
                    
                    for index, label in enumerate(labels):
                        stats[f'{header} max'][index] = max_dict[header][label]
                        stats[f'{header} min'][index] = min_dict[header][label]
                        message += f'{label:<2} Max: {max_dict[header][label]:<8} | Min: {min_dict[header][label]:<8} \n'
                        
                common.Print(message)
                
                if save:
                    df = pd.DataFrame(stats, index=labels)
                    df.to_csv(f'{save_path}/max-min-class.csv')
                
                if output_type == list:
                    return stats
                elif output_type == dict:
                    return max_dict, min_dict
            
            else:
                message = "File list is empty"
                common.Print_Error('Stats Class -> maximum-minimum', message)
                return None
            
        except Exception as e:
            common.Print_Error('Stats Class -> maximum-minimum', e)
    
    
    """
        mean_std_class(average=False)
        Finds the mean and standard deviation along columns of a dataset stored in 
        a list of csv files.
        
    inputs:
        - average (bool): Compute the average mean and std or the global?
    outputs:
        - stats (np array): An array of stats for each column separated by label.
    """
    def mean_std_class(self, average=False):
        try:
            if self.file_count > 0:
                save = self.attribute_get('save')
                save_path = self.attribute_get('save path')
                overwrite = self.attribute_get('overwrite')
                
                headers = self.attribute_get('headers')
                label_header = self.attribute_get('label header')
                save_path, label_header = self.validate_inputs_class(save_path, label_header)
                
                if common.Validate_File(save_path) and not overwrite:
                    print("[INFO] Found an existing stats-class.csv file at: " + save_path)
                    stats = pd.read_csv(save_path)
                    print(stats)
                    self.stats = stats
                    return stats
                
                headers_to_keep = headers + [label_header]
                
                column_names = []
                df = self.data[0]
                
                header_count = len(headers)
                label_count = len(df[label_header].unique())
                stats = np.empty((label_count, 2*header_count))
                
                for header in headers:
                    column_names.append(f'{header} mean')
                    column_names.append(f'{header} std')
                
                if average:                    
                    for j in range(label_count):
                        mean = np.zeros((header_count,))
                        std = np.zeros((header_count,))
                        
                        for i in range(self.file_count):
                            df = self.data[i][headers_to_keep]
                            label_data = df[df[label_header] == j][headers].astype(float)
                            label_mean = label_data.mean(axis=0)
                            label_std = label_data.std(axis=0)
                            
                            if not label_std.isnull().values.any() and not label_mean.isnull().values.any():
                                mean += label_mean
                                std += label_std
                            
                            common.Print_Status(f'Mean/STD/Class -> Average Class {j}', i, self.file_count)
                        
                        for i in range(header_count):
                            k = int(2 * i)
                            stats[j,k] = mean[i] / self.file_count
                            stats[j,k+1] = std[i] / self.file_count
                            
                    file_name = 'mean-std-class-ave.csv'
                    
                else:
                    data = self.data[0][headers_to_keep]
                    
                    for index in range(1, self.file_count):
                        df = self.data[index][headers_to_keep]
                        data = pd.concat([data, df], axis=0)
                        
                        common.Print_Status('Mean/STD/Class -> Stack Data', index, self.file_count)
                        
                    for i in range(label_count):
                        label_data = data[data[label_header] == i][headers]
                        mean = label_data.mean()
                        std = label_data.std()
                        
                        for j in range(header_count):
                            k = int(2 * j)
                            stats[i,k] = mean[j]
                            stats[i,k+1] = std[j]
                        
                        common.Print_Status('Mean/STD/Class -> Compute', i, label_count)
                        
                    file_name = 'mean-std-class.csv'
                    
                if save: 
                    df = pd.DataFrame(stats, columns=column_names)
                    df.to_csv(f'{save_path}{file_name}', index=False)
                
                return stats
            
            else:
                message = "File list is empty"
                common.Print_Error('mean - std - class', message)
                return None
        
        except Exception as e:
            common.Print_Error('Stats Class -> mean std class', e)
    
    
    """
        point_count()
        A function to compute statistics about the number of data points for 
        each label. Performs similar functionality to length_class(), but with
        the exception that this function is not intented to be used during
        dataset loading prior to training, instead it is designed to be used
        in an offline manner for saving data and presenting results.
        
    inputs:
        -
    outputs:
        - stats (dict): A dict of the computed stats
    """
    def point_count(self):
        try:
            if self.file_count > 0:
                save = self.attribute_get('save')
                save_path = self.attribute_get('save path')
                overwrite = self.attribute_get('overwrite')
                
                label_header = self.attribute_get('label header')
                save_path, label_header = self.validate_inputs_class(save_path, label_header)
                
                if common.Validate_File(save_path) and not overwrite:
                    print("[INFO] Found an existing stats-class.csv file at: " + save_path)
                    stats = pd.read_csv(save_path)
                    print(stats)
                    self.stats = stats
                    return stats
                
                df = self.data[0]
                labels = df[label_header].unique().astype(int)
                    
                label_count = len(labels)
                
                stats = {'total points':[0]*label_count, 
                         'max points'  :[0]*label_count, 
                         'min points'  :[0]*label_count, 
                         'mean points' :[0]*label_count,
                         'std points'  :[0]*label_count}
                
                points = np.zeros((self.file_count, label_count))
                
                for index in range(self.file_count):
                    df = self.data[index]
                    
                    for label in labels:
                        points[index,label] = df[df[label_header] == label].shape[0]
                    
                    common.Print_Status('Point Count', index, self.file_count)
                    
                for label in labels:
                    
                    stats['total points'][label] = sum(points[:,label])
                    stats['max points'][label] = max(points[:,label])
                    stats['min points'][label] = min(points[:,label])
                    stats['mean points'][label] = np.mean(points[:,label])
                    stats['std points'][label] = np.std(points[:,label])
                        
                file_name = 'point-stats-class.csv'
                    
                if save: 
                    df = pd.DataFrame(stats)
                    df.to_csv(f'{save_path}{file_name}', index=False)
                
                return stats
            
            else:
                message = "File list is empty"
                common.Print_Error('mean - std - class', message)
                return None
        
        except Exception as e:
            print(label)
            common.Print_Error('Stats Class -> point count', e)
        
        
    """
        validate_inputs_class(save_path, label_header)
        A function used to automate the validation of specific parameters
        including the save path and label header.
        
    inputs:
        - save_path (string): If the input is None a default value will be used instead
        - label_header (string): If the input is None a default value will be used instead
    outputs:
        - save_path (string): Either default value or the input value
        - label_header (string): Either default value or the input value
    """
    def validate_inputs_class(self, save_path, label_header):
        try:
            if save_path is None:
                save_path = "../stats-class.csv"
            
            if label_header is None:
                label_header = 'Label'
            
            return save_path, label_header
        except Exception as e:
            common.Print_Error('Stats Class -> valdiate inputs class', e)
            
            
# Functionality for testing the stats.py functions and generating the stats outputs
if __name__=="__main__":    
    
    sub_set = '_training'
    save_path = f"../Stats/processed-low-high/2022/{sub_set}/"
    #load_path = f"../Data/{sub_set}/*/Log Files/*.csv"
    #load_path = "../Data/temp/*/*.csv"
    load_path = f"../Data/Labeled/processed-low-high/*/*.csv"
    file_list = glob.glob(load_path)
    
    headers = ['Depth mm', 'Depth Delta', 'Depth Accel', 'Force lbf', 'Force Delta', 'Force Accel']
    
    correlation(file_list, headers=headers, plot=True, save=True, file_path=save_path)
    histogram(file_list, headers=headers, buckets=100, plot=True, save=True, save_path=save_path, use_matplotlib=False)
    length(file_list, compute_stats=True, save=True, save_path=save_path)
    maximum_minimum(file_list, headers=headers, output_type=list, save=True, save_path=save_path)
    mean_std(file_list, headers=headers, average=False, save=True, save_path=save_path)
    mean_std(file_list, headers=headers, average=True, save=True, save_path=save_path)
    point_count(file_list, save=True, save_path=save_path)
    
    stat = stats(file_list=load_path, save=True, save_path=save_path, label_header='Label', headers=headers)
    stat.correlation(plot=True)
    stat.histogram(buckets=100, plot=True, use_matplotlib=False)
    stat.length_class(compute_stats=True)
    stat.maximum_minimum()
    stat.mean_std_class(average=False)
    stat.mean_std_class(average=True)
    stat.point_count()