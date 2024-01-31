"""
Project: 
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
To label log files. There are three ways which this class can perform labeling.
The first is automatic labeling from provided files. Reference the 2023 
Log Files - Labels csv file for the expected label file structure. The second
way is to use a ML model to label log files in a semi-supervised scheme. 
The third way is to use manually labeled point files from the plotter routine 
to generate the labels. 

Functions:
 - __init__(path, year, load_path, save_path, manual_headers, label_headers, automatic_headers)
 - Build_File_List()
 - Label_Automatic_From_File(log_file='Log File - Labels.csv')
 - Label_Automatic_With_Model(model_path, sequence_length)
 - Label_Manual()
 - Process_Manual_Labels(file)
 - Transfer_Points_From_Log_Files(log_file='Log File - Labels.csv', header_label='Alarms', header_feature='Depth mm'):

Included with: 
 - ann.py 
 - ann_tester.py
 - common.py 
 - data_generator.py
 - dataloader.py
 - gan_tester.py
 - labeler.py (current file)
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
import numpy as np
import pandas as pd

import torch


# In House Dependencies
import common
import dataloader as dl


"""
    Labeler
    Constructed as a class to reduce the amount of data being passed between functions
"""
class Labeler:
    
    """
        __init__(self, path, year, load_path, save_path)
        Appends the year to the path and stores the resulting location locally
    
    inputs: 
        - path (string): The base path to the top level of the labeled data
        - year (string): May be a specific year to process or * to process all years
        - load_path (string): The folder of the location to load point csv data from
        - save_path (string): The folder of the location to save labeled csv data to
    outputs:
        - 
    """
    def __init__(self, path, year, load_path, save_path, manual_headers, label_headers, automatic_headers):
        self.data_path = path + '/' + load_path + '/' + year     # Tracks the path to the data
        self.file_list = ''                                      # Stores the list of files to process 
        self.load_path = load_path                               # Stores the folder to load data from
        self.save_path = save_path                               # Stores the folder to save data to
        self.df = ''                                             # Stores the current log files dataframe
        self.pf = ''                                             # Stores the current point file dataframe
        self.errors = []                                         # Tracks error which occur during processing
        self.manual_headers = manual_headers                     # Tracks the valid headers for manual labeling
        self.label_headers = label_headers                       # Tracks the headers used in the automatic label file
        self.automatic_headers = automatic_headers               # Tracks the valid headers for automatic labeling
                

    """
    
    """
    def Forward_Prop_Logs(self, log_folder='Log Files'):
        load_list = []
        save_path = self.data_path.replace(self.load_path, self.save_path) + f'/{log_folder}/*.csv'
        save_list = glob.glob(save_path)
        
        file_count = len(save_list)
        
        for entry in save_list:
            load_list.append(entry.replace(self.save_path, self.load_path))
    
        for index, file in enumerate(load_list):
            try:
                df = pd.read_csv(file)
                file = file.replace(self.load_path, self.save_path)
                df.to_csv(file, index=False)
                
                common.Print_Status('Forward Prop Logs', index, file_count)
            except Exception as e:
                common.Print_Error('Labeler -> Backprop Labels', f'File: {file}\n{e}')
        
        
    """
        Build_File_List()
        Build File List is used at the beginning of the program to create the path
        to the files which will be processed
    """
    def Build_File_List(self):
        path = self.data_path + '/Log Files/*.csv'
        self.file_list = glob.glob(path)
        print('Found ' + str(len(self.file_list)) + ' files in ' + str(path))    
        
        path = self.data_path + '/Log Files/'
        path = path.replace(self.load_path, self.save_path)
        common.Validate_Dir(path)
        
        
    """
        Label_Automatic()
        Automatically loads labels from the provided files and uses the proprietary
        labeling functions to generate labels without human intervention.
    """
    def Label_Automatic_From_File(self, log_file='Log File - Labels.csv'):
        
        label_file = self.data_path + '/' + log_file
        df = pd.read_csv(label_file)
        file_names = df['Name'].values
        
        save_path = f'{self.data_path.replace(self.load_path, self.save_path)}/Log Files/'
        common.Validate_Dir(save_path)
        
        columns = df.columns.values
        for header in self.label_headers:
            try:
                columns = columns[columns != header]
            except Exception as e:
                common.Print_Error('Labeler -> Label Automatic', f'Header: {header}\n{e}')
                
        labels_all = df.drop(columns=columns).values
        
        file_count = labels_all.shape[0]
        
        for index, file in enumerate(file_names):
            try:
                file = f'{self.data_path}/Log Files/{file}'
                log = pd.read_csv(file).reset_index()

                labels_file = labels_all[index,:]
                labels_list = np.zeros((log.shape[0],))
                
                for label in labels_file:
                    break_index = log[log['Depth mm'] > label]['index'].values[0]
                    labels_list[break_index:] += 1
                
                log.drop(columns='index')
                log.insert(1, column='Label', value=labels_list)    # Insert a column to store labels
                file = file.replace(self.load_path, self.save_path)
                log.to_csv(file, index=False)
                
                common.Print_Status('Label Automatic', index, file_count)
                
            except Exception as e:
                common.Print_Error('Labeler -> Label Automatic', f'File: {file}\n{e}')
            
    
    """
        Label_Automatic_With_Model(model_path, sequence_length)
        A function which uses a provided model to generate the labels for the data.
        Used in semi-supervised learning to augment the data by creating labels
        where none were before.
        
        inputs:
         - model_path (string): The path to the torch model
         - sequence_length (int): The length of training sequences, labels will be zero padded
        outputs:
         -
    """
    def Label_Automatic_With_Model(self, model_path, sequence_length):
        model = torch.load(model_path)
        file_count = len(self.file_list)    # Used by print status to display the percent of files processed
        
        for index, file in enumerate(self.file_list):
            try:
                df = pd.read_csv(file, index=None)
                row_count = df.shape[0]
                labels = np.zeros((row_count,))
                data = dl.Preload_Dataset(file_path=file,
                                          
                                          verbose=0)
                labels[sequence_length:] = torch.argmax(model(data), dim=1)
                df.insert(1, column='Label', value=labels)
                
            except Exception as e:
                common.Print_Error('Labeler -> Label Automatic With Model', f'File: {file}\n{e}')
        
            
    """
        Label_Manual()
        Auto label uses point files created by the plotter routine during manual 
        labeling to generate labeled files. It pulls in csv data and calls on 
        other functions to process the data.
    """
    def Label_Manual(self, automatic=None):
        file_count = len(self.file_list)    # Used by print status to display the percent of files processed
        
        # Iterate over all files and process each 
        for index, file in enumerate(self.file_list):
            try:
                self.df = pd.read_csv(file)     # Load the csv file into a dataframe
                
                #if len(self.df['Alarms'].unique()) == 4 and automatic is not None:
                #    self.Process_Automatic_Labels(file, automatic)
                #else:
                self.Process_Manual_Labels(file, automatic)       # Add the labels to the data and save the new csv
                    
                        
                common.Print_Status('Label Manual', index, file_count)
            
            except Exception as e:
                common.Print_Error('Labeler -> Label Manual', f'File: {file}\n{e}')
            index += 1  
    
    
    """
        Process_Labels(file)
        A function to add a label column and populate it based on the point file data.
        The final dataframe is then saved.
        
    inputs:
        - file (string): Location of a file to load for processing
    outputs:
        -
    """
    def Process_Manual_Labels(self, file, automatic_points):
        # Check that the number of columns is correct
        if len(self.df.columns) < 40:
            point_file = file.replace('Log', 'Point')
            self.pf = pd.read_csv(point_file)
            
            new_col = []
            
            j = 0
            # For all of the points in the point dataframe add labels in ascending order
            for i in range(len(self.df)):
                if j < len(self.pf):
                    if i > self.pf.values[j,1]:
                        j += 1
                    
                new_col.append(j)


            self.df.insert(1, column='Label', value=new_col)    # Insert a column to store labels
            
            self.df = self.df[self.manual_headers]                      # Remove the extra column
            label_path = file.replace(self.load_path, self.save_path)   # Change from load to save folder
            self.df.to_csv(label_path, index=False)                     # Save the labeled data 
        else:
            self.errors.append('[ERROR] The file does not have the correct number of headers | ' + file)
            
    
    """
        Transfer_Points_From_Log_Files(log_file, header_label, header_feature)
        A function to take a partially completed log file of labels and complete
        it by using the controller algorithms Alarms outputs as the ground truth.
        
        inputs:
         - log_file (string): The name of the log file of labels to complete
         - header_label(string): The header to use for labels in the drill file
         - header_feature(string): The header to use for label evaluation
        outputs:
         - 
    """
    def Transfer_Points_From_Log_Files(self, log_file='Log File - Labels.csv', header_label='Alarms', header_feature='Depth mm'):
        
        label_file = self.data_path + '/' + log_file
        df = pd.read_csv(label_file)
        file_names = df['Name'].values
        
        save_path = f'{self.data_path.replace(self.load_path, self.save_path)}/'
        common.Validate_Dir(save_path)
        save_path += 'Log File - Labels.csv'
        
        file_count = len(file_names)
        first_point_list = np.zeros((file_count,))
        second_point_list = np.zeros((file_count,))
        
        for index, file in enumerate(file_names):
            try:
                file = f'{self.data_path}/Log Files/{file}'
                log = pd.read_csv(file)
                
                break_point = log[log[header_label] == 1][header_feature].values
                if len(break_point) > 1:
                    first_point_list[index] = break_point[0]
                    
                break_point = log[log[header_label] == 2][header_feature].values
                if len(break_point) > 1:
                    second_point_list[index] = break_point[0]
                
                common.Print_Status('Transfer Points', index, file_count)
                
            except Exception as e:
                common.Print_Error('Labeler -> Transfer Points From Log Files', f'File: {file}\n{e}')
        
        df[self.label_headers[0]] = first_point_list
        df[self.label_headers[2]] = second_point_list
        
        label_file = label_file.replace(self.load_path, self.save_path)
        df.to_csv(label_file, index=False)


"""
Load csv: base_path + load_path + year + file_name.csv
Save csv: base_path + save_path + year + file_name.csv
"""
if __name__=='__main__':
    label_type = 0
    log_file = 'Log File - Labels.csv'
    base_path = '../Data/'
    #base_path = 'A:/School/Research/Drill/Data/Composite/Expanded-Features/'
    #base_path = 'F:/blaney/Drill/Data/Composite/Expanded-Features/'
    load_path = 'Pruned'
    save_path = 'temp'
    year = '2023'
    
    manual_headers = ['Depth mm', 'Force lbf', 'Label']
    label_headers = ['CT First', 'CT Endosteal', 'CT Second']
    automatic_headers = ['Depth mm',                    
                         'Depth Delta',
                         'Depth Accel',
                         'Force lbf',
                         'Force Delta',
                         'Force Accel',
                         'Force/Distance',
                         #'Alarms',
                         'Label']
    
    model_path = '../Models/rnn/2023-08-29_18-08/model.pt'
    
    label = Labeler(path=base_path, 
                    year=year, 
                    load_path=load_path, 
                    save_path=save_path, 
                    #manual_headers=manual_headers, 
                    manual_headers=automatic_headers,
                    label_headers=label_headers,
                    automatic_headers=automatic_headers)
    
    if label_type == 0:
        label.Label_Automatic_From_File(log_file=log_file)
        
    elif label_type == 1:
        label.Build_File_List()
        label.Label_Automatic_With_Model(model_path)
    
    elif label_type == 2:
        label.Build_File_List()
        label.Label_Manual()
        
    elif label_type == 3:
        label.Transfer_Points_From_Log_Files(log_file=log_file)
        
    elif label_type == 4:
        label.Forward_Prop_Logs()
    