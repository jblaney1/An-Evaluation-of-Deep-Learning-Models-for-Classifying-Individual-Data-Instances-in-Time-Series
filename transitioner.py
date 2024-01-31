"""
Project: 
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
Identifies where a model has predicted transitions, regardless of class 
correctness, and computes a transition accuracy based on if a transition was 
identified in the specified window. Specifically requested for this project 
to test the usability of models which may not have good accuracy on individual 
timesteps but may be able to identify when the transitions occur. 

Functions:
Transitioner
 - def __init__(name=None, window_size=10)
 - process_transitions(path)
 - process_transitions_helper(preds)

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
 - trainer.py
 - transitioner.py (current file)
 
Notes:
For more information about the project contact 
 - Dr. Suresh Muknahallipatna -> sureshm@uwyo.edu
 - Josh Blaney -> jblaney1@uwyo.edu
"""


# Outside Dependencies
import glob
import numpy as np
import pandas as pd


# In House Dependencies
import common


"""
    transitioner
    A class to house the transition functions and results.
"""
class transitioner():
    def __init__(self, name=None, window_size=10):
        self.name = name
        self.window_size = window_size
    
            
    """
        process_transitions(path)
        A function to process all of the files at path and identify where the 
        transitions occured in the predictions. The result is stored in a local
        variable as a floating point accuracy.
        
        inputs:
         - path (string): Location of the files to process
        outputs:
         - 
    """
    def process_transitions(self, path):
        file_list = glob.glob(path)
        file_count = len(file_list)
            
        if file_count < 1:
            e = "[ERROR] File list is empty"
            common.Print_Error('Transitioner - > load preds', e)
            return
        
        message = "[INFO] Starting the load predictions process\n"
        message += f"       Load Location: {path}\n"
        message += f"       File Count: {file_count}\n"
        common.Print(message)
        
        file = file_list[0]
        df = pd.read_csv(file)
        preds = df.values
        self.transition_accuracy = np.zeros((preds.shape[-1]))
        
        for i, file in enumerate(file_list):
            df = pd.read_csv(file)
            preds = df.values
            self.transition_accuracy += self.process_transitions_helper(preds)
            common.Print_Status('Process Transitions', i, file_count)
            
        self.transition_accuracy /= file_count
    
    
    """
        process_transitions_helper(preds)
        A function to assist in identifying transitions. This function takes an
        array of predictions and finds all of the points where class transitions
        occur regardless of proper class identification. The transitions are 
        represented as 1's in a vector of equal length to the predicition array.
        
        inputs:
         - preds (array): The predictions to identify transitions in
        outputs:
         - transitions (list): The list of where transitions occured
    """
    def process_transitions_helper(self, preds):
        transitions = []
        point_count = preds.shape[0]
        labels = np.unique(preds[:,2])
        labels_count = len(labels)
        half_window = int(self.window_size / 2)
        
        for i, label in enumerate(labels):
            if i == labels_count - 1:
                break
            
            transitions.append(0)
            label_index = np.where(preds[:,2]==label)[0][-1] - half_window
            label1 = preds[label_index,1]
            
            for j in range(self.window_size):
                if label_index + j < point_count: 
                    label2 = preds[label_index+j,1]
                    if label1 != label2:
                        transitions[i] = 1
                        break
                else:
                    break
        
        return transitions
    

# Functionality to test the transitioner functions
if __name__ == 'main':
    windows = 8
    window_scaler = 2
    transitions = []
    path_preds = "../Data/predictions/*.csv"
    trans = transitioner(name='tester')
    
    for i in range(windows):
        trans.window_size = window_scaler**i
        trans.process_transitions(path_preds)
        transitions.append(trans.transition_accuracy)
    
    for i in range(windows):
        print(f"Window Size : {window_scaler**i:>4} | Transition Accuracy : {transitions[i]}")