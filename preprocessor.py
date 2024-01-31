"""
Project: 
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
A class which can be used to preprocess the log files so that they are ready 
for machine learning. Specifically this file allows validation, truncation,
linearization, averaging, frequency analysis, and pruning. For more information
about the specific usage of a function refer to the comments in and around
that fuction.

Functions:
 - Average_Transform(window=5)
 - Build_File_Lists()
 - Change_Name(old='default', new='default')
 - Frequency_Transform(transform=1)
 - Headers_Correct()
 - Headers_Validate()
 - Load_Files(file_list, headers='infer')
 - Log_Event(message='default', file='default')
 - Save_Logs()
 - Split()
 - Truncate()

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
 - preprocessor.py (current file)
 - stats.py
 - trainer.py
 - transitioner.py
 
Notes:
For more information about the project contact 
 - Dr. Suresh Muknahallipatna -> sureshm@uwyo.edu
 - Josh Blaney -> jblaney1@uwyo.edu
"""

# Outside Dependencies
import os
import glob
import pandas as pd
import numpy as np


# In House Dependencies
import common


"""
The following functions have been grouped into a class to reduce the amount
of data passed between functions
"""
class Preprocessor:
    
    """
        __init__(self, name='default', base_path='', year='*', load='', save1='Original', save2='Linearized', save3='Averaged', save4='Velocity')
        The initializer is used to transfer the base file path and year to the objects local memory
        as well as establishing which headers need to be removed.
    """
    def __init__(self, name='default', base_path='', year='*', load='', save1='Original', save2='Linearized', save3='Averaged', save4='Velocity', minimum_length=100):
        self.name = name                    # A string used to track which processor you are using if making a list of processors
        self.year = year                    # A string used to track the year folder being processed
        self.load = load                    # A string used to track the path change made to load the original csv files
        self.save1 = save1                  # A string used to track the path change made to store truncated csv files 
        self.save2 = save2                  # A string used to track the path change made to store linearized csv files 
        self.save3 = save3                  # A string used to track the path change made to store averaged csv files 
        self.save4 = save4                  # A string used to track the path change made to store velocity csv files
        self.min_length = minimum_length    # A int used to track the minimum acceptable run length to keep
        
        self.invalid_files = []             # A list of the files which have failed validation
        self.invalid_file_count = 0         # A count of the files which have failed validation
        self.base_path = base_path          # A string used to track the default location of the data
        self.load_path = ''                 # A string used to track the load location of the data
        self.save_truncs_path = ''          # A string used to track the save location of truncated data
        self.save_linear_path = ''          # A string used to track the load location of linearized data
        self.save_averag_path = ''          # A string used to track the load location of averaged data 
        self.file_list = []                 # A list of the available files
        self.change_log = []                # A list of the changed which have been made to the data
        self.error_log = []                 # A list of the errors which have occured during processing
        self.valid_headers = ['Depth mm',   # A list of the headers to keep
                              'Depth Delta',
                              'Depth Accel',
                              'Force lbf',
                              'Force Delta',
                              'Force Accel',
                              'Force/Distance',
                              'Alarms']
        
        self.Build_File_Lists()
        
        
    """
        Average_Transform(window)
        Performs window averaging such that each point is replaced by the average of the window on each side of the point
    
    inputs:
        - window (int): The number of elements to average per window
    outputs:
        -
    """
    def Average_Transform(self, window=5):
        try:
            self.index = 0 # Used by prompt to track the percent of files processed
            file_list = glob.glob(self.save_linear_path)
            file_count = len(file_list) # Used as the total files to process
            
            # Notify the programmer which step is about to commense
            print('[INFO] Transforming the data of ' + str(file_count) + ' CSV files from ' + self.save_linear_path)
            
            # Iterate over all files and shorten their datasets so that they make 
            # sense for a time series. Look at a plot for more information
            for file in file_list:
                self.index += 1
                common.Print_Status('Average Transform', file_num=self.index, total_files=file_count) # Print the current percent of processed files
                df = pd.read_csv(file)      # Load the csv file with headers in the header
                index_count = df.shape[0]
                
                ave_force = []
                force_col = df[self.valid_headers[1]]
        
                if index_count > 100:
                    for index in range(window-1, index_count-window):
                        ave_force.append(sum(force_col[index-window:index+window]) / (2*window+1))
                        
                    df = df.iloc[window-1:index_count-window]
                    
                    df[self.valid_headers[1]] = pd.Series(ave_force)
                    
                    # If almost all of the data was removed delete the file
                    # Otherwise write a new file with the fixed and truncated data
                    file = file.replace('Linearized', 'Averaged')
                    file = file.replace('Log Files', 'Log Files ' + str(window))
                    df.to_csv(file, index=False)
                    
        except Exception as e:
            common.Print_Error('Preprocessor -> Average Transform', e)
    
    
    """
        Build_File_Lists()
        Build File List is used at the beginning of the program to create the path to the files which will be processed
    """
    def Build_File_Lists(self):
        try:
            self.load_path = f'{self.base_path}/{self.load}/{self.year}/Log Files/*.csv'
            
            save_truncs_base = f'{self.base_path}/{self.save1}/{self.year}/Log Files'
            save_linear_base = f'{self.base_path}/{self.save2}/{self.year}/Log Files'
            save_averag_base = f'{self.base_path}/{self.save3}/{self.year}/Log Files'
            
            self.save_truncs_path = f'{save_truncs_base}/*.csv'
            self.save_linear_path = f'{save_linear_base}/*.csv'
            self.save_averag_path = f'{save_averag_base}/*.csv'
            
            if '*' in self.year:
                year_path = glob.glob(f'{self.base_path}/{self.load}/{self.year}')
                
                for year in year_path:
                    year = year.split('\\')[-1]
                    temp_path = save_truncs_base.replace(self.year, year)
                    common.Validate_Dir(temp_path)
                    temp_path = save_linear_base.replace(self.year, year)
                    common.Validate_Dir(temp_path)
                    temp_path = save_averag_base.replace(self.year, year)
                    common.Validate_Dir(temp_path)  
                    
            else:
                common.Validate_Dir(save_truncs_base)
                common.Validate_Dir(save_linear_base)
                common.Validate_Dir(save_averag_base)
                    
            self.file_list = glob.glob(self.load_path)
        
        except Exception as e:
            common.Print_Error('Preprocessor -> Build File List', e)
        
    
    """
        Change_Name(old, new)
        A function to change the name of many csv files from old to new
        
    inputs:
        - old (string): Original/problematic name convention
        - new (string): new name convention
    outputs:
        -
    """
    def Change_Name(self, old='default', new='default'):
        try:
            self.index = 0                      # Used by prompt to track the percent of files processed
            file_count = len(self.file_list)    # Used by prompt to track the total file count
            
            # Iterate over all files and replace the specified string (old) to new
            for file in self.file_list:
                self.index += 1                                                 # Update the index of the files being processed
                df = pd.read_csv(file)                                          # Load the data from the original file
                new_file = file.replace(old, new)                               # Replace the old string with the new one
                df.to_csv(new_file, index=False)                                # Save the data with the new file name
                os.remove(file)                                                 # Remove the old file
                common.Print_Status('Change Name', self.index, file_count)      # Update the status bar
                
        except Exception as e:
            common.Print_Error('Preprocessor -> Change Name', e)
    
    
    
    """
        Frequency_Transform(transform)
        Performs a fast fourier transform based on the value of transform in either 1 or 2 dimensions.
        
    inputs:
        - transform (int): The dimensionality of the transform
    outputs:
        -
    """
    def Frequency_Transform(self, transform=1):
        file_list = glob.glob(self.save_linear_path)
        file_count = len(file_list)
        
        # Notify the programmer which step is about to commense
        print('[INFO] Transforming the data of ' + str(file_count) + ' CSV files from ' + self.save_linear_path)
        
        # Iterate over all files and shorten their datasets so that they make 
        # sense for a time series. Look at a plot for more information
        for index, file in enumerate(file_list):
            try:
                df = pd.read_csv(file)
                
                if df.shape[0] < self.min_length:
                    os.remove(file)
                else:
                    
                    if transform == 1:
                        df_temp = pd.DataFrame(np.absolute(np.fft.fft(df[self.valid_headers[1]])))
                        df[self.valid_headers[1]] = df_temp
                    elif transform == 2:
                        df_temp = pd.DataFrame(np.absolute(np.fft.fft2(df)))
                        df[self.valid_headers[0]] = df_temp[0]
                        df[self.valid_headers[1]] = df_temp[1]
                    
                    # If almost all of the data was removed delete the file
                    # Otherwise write a new file with the fixed and truncated data
                    if df.shape[0] >= self.min_length:
                        file = file.replace('Linearized', 'Transformed')
                        file = file.replace('Log Files', 'Log Files ' + str(transform))
                        df.to_csv(file, index=False)
                    
                common.Print_Status('Frequency Transform', index, file_count) # Print the current percent of processed files
                        
            except Exception as e:
                common.Print_Error('Preprocessor -> Frequency Transform', e)
        
    
    """
        Headers_Correct()
        Headers Correct should be used second. This function iterates over the files listed for editing
        and performs two steps, 1.) it checks the length of the header row to make sure that only 41
        headers are present, sometimes data will be written to the header row along with the headers 
        2.) it removes all of the headers listed as invalid and then writes a new csv file
    """
    def Headers_Correct(self):
        # Notify the programmer which step is about to commense
        print('Fixing ' + str(self.invalid_file_count) + ' files')
        
        # Iterate over all invalid files and attempt to fix the header
        for index, file in enumerate(self.invalid_files):                
            # Some files are too big for the C engine and the python engine
            # must be used instead. Load the csv with headers in row 0
            try:
                df = pd.read_csv(file, header=None, low_memory=False)
            except Exception as e:
                self.Log_Event(message='[ERROR] ' + str(e), file=file)
                df = pd.read_csv(file, header=None, engine='python', low_memory=False)
                
            header_count = len(df.iloc[0].values)               # Check the current number of headers

            # The MO files should have a max of 41 headers but data may be on the same line
            if header_count > 40:
                actual_header_count = int((header_count+1) / 2)
                split_header = df.iloc[0].values[actual_header_count-1].split('n') # Split the final header on its last character
                
                # If there was something split from the header data was incorrectly
                # appended to the header row. Separate the data from the header
                if len(split_header) > 1:
                    data = [float(split_header[1])]
                    df.iloc[0, actual_header_count-1] = split_header[0] + 'n'

                header_data = df.iloc[0].values[actual_header_count:]       # Store data on the header row
                headers = df.iloc[0].values[:actual_header_count]           # Store the headers themselves
                data = np.append(data, header_data)                         # Combine the initial data point and the data from the header row
                df = df.dropna(axis=1)                                      # Remove the N/A values from the rows with data as headers
                df  = df[1:]
                df = df.set_axis(headers, axis=1)                           # Set the header of the dataframe equal to the stored headers
                df = df.reset_index(drop=True)                              # Restart the indexing at 0
                df.to_csv(file, index=False)                                # Overwrite the corrected csv file
            
            else:
                print('Unable to edit file, an unknown issue is occuring ' + file)
            
            common.Print_Status('Headers Correct', self.index, self.invalid_file_count)   # Print the current percent of processed files
                
        x = ''
        
        # Check to see if another validation pass is requested
        while x != 'y' and x != 'n':
            print('[INFO] Correction has completed. Do you want to validate again? [y, n]')
            x = input()
        
        if x == 'y':
            print('[INFO] Validation will now begin with header_validate')
            self.Headers_Validate()
        else:
            print('[INFO] Ending Correction Routine')
            
            
    """
        Headers_Validate()
        Headers Validate is the first function which should be run. It iterates over all provided 
        files and checks to make sure that the headers stored in valid_headers are present in the
        dataframe. If any headers are missing the file is added to a list to be editted later. The
        files listed for editting and the reason are written to a log file when the function completes
        processing.
    """
    def Headers_Validate(self):
        file_count = len(self.file_list)    # Used by prompt to track the total file count
        
        # Only process files which were either invalid or have not been processed
        if self.invalid_file_count > 0:
            file_count = self.invalid_file_count
            self.file_list = self.invalid_files
            self.invalid_file_count = 0
            self.invalid_files = []
        
        # Notify the programmer which step is about to commense
        print('[INFO] Validating the headers of ' + str(file_count) + ' CSV files from ' + self.load_path)
        
        # Iterate over all files in the list and check their headers
        for index in range(file_count):
            file = self.file_list[index]
            df = pd.read_csv(file, header=None, low_memory=False)
            
            headers = df.iloc[0].values             # Store the headers in a separate list for processing
            header_count = len(headers)             # Get the number of headers to validate the count
            error_found = False                     # Track if an error has been found
            
            # The MO csv files have a maximum of 40 headers depending on format
            if header_count > 41:
                self.Log_Event(message='[INFO] The header is too long', file=file)
                error_found = True
            
            # Create a list to keep track of which headers have failed validation
            header_list = []
            
            # Iterate through the valid headers to make sure they are all present
            for header in self.valid_headers:
                if header not in headers:
                    header_list.append(header)
            
            # Generate a message which lists all of the missing headers for logging
            if len(header_list) > 0:
                error_found = True
                message = '[INFO] The header(s) | '
                for header in header_list:
                    message += header + ', '
                self.Log_Event(message=message + ' | is missing or incomplete', file=file)
            
            # Add the file to a list which indicates it needs to be fixed
            if error_found:
                self.invalid_files.append(file)
                self.invalid_file_count += 1
            else:
                file = file.replace(self.load, self.save1)
                df.to_csv(file, header=False, index=False)
            
            common.Print_Status('Headers Validate', index, file_count)    # Print the current percent of processed files
                
        # Check for invalid files, if there are none complete processing
        # If there are invalid files suggest that they be fixed
        if self.invalid_file_count > 0:
            x = ''
            
            # Prompt the user for a yes or no answer to fixing the invalid files
            while x != 'y' and x != 'n':
                print('[INFO] ' + str(self.invalid_file_count) + ' files failed validation')
                print('[INFO] Do you want to try correcting the files? [y, n]')
                x = input()
                
            if x == 'y':
                print('[INFO] Using header_correct to fix the broken files')
                self.Headers_Correct()
            else:
                print('[INFO] Ending validation routine')
        
        else:
            print('[INFO] Validation Complete')
        
    
    """
        Linearize()
        Linearize forces the dataset to progress monotonically over the specified column. For the 
        MO project one feature was being plotted with respect to another so the independent feature
        had to be normalized before the time series analysis could be performed.
    """           
    def Linearize(self):
        file_list = glob.glob(self.save_truncs_path)
        file_count = len(file_list)
        
        # Notify the programmer which step is about to commense
        print('[INFO] Linearizing the data of ' + str(file_count) + ' CSV files from ' + self.save_truncs_path)
        
        # Iterate over all files and shorten their datasets so that they make 
        # sense for a time series. Look at a plot for more information
        for index, file in enumerate(file_list):
            try:
                df = pd.read_csv(file)      # Load the csv file with headers in the header
                
                if df.shape[0] < self.min_length:
                    os.remove(file)
                else:
                    col_index_depth = df.columns.get_loc(self.valid_headers[0])     # Find the index of the Depth mm column
                    current_depth = df.min(axis=1,)[col_index_depth]                # Find the initial lowest depth that is recorded
                    duplicate_index_list = []
                    i = 0                                                           # Initialize the iterator to 1
                    
                    temp = df[self.valid_headers[0]]
                    
                    for i in range(temp.shape[0]):
                        if temp[i] > current_depth:
                            current_depth = temp[i]
                        else:
                            duplicate_index_list.append(i)
                            
                    df = df.drop(axis='index', index=duplicate_index_list)
                    
                    # If almost all of the data was removed delete the file
                    # Otherwise write a new file with the fixed and truncated data
                    if df.shape[0] >= self.min_length:
                        file = file.replace('Truncated', 'Linearized')
                        df.to_csv(file, index=False)
                        
                common.Print_Status('Linearize', index, file_count) # Print the current percent of processed files
            
            except Exception as e:
                common.Print_Error('Preprocessor -> Linearize', e)
    
    
    """
        Load_Files(file_list, headers)
        Load Files automatically loads all of the data present into an array of arrays of dataframes.
    """
    def Load_Files(self, file_list, headers='infer'):
        try:
            print('[INFO] Preloading ' + str(len(file_list)) + ' Files | From: ' + str(file_list[0]))
            temp = np.array([np.array(pd.read_csv(file, header=headers, low_memory=False)) for file in file_list])
            print('[INFO] Preload complete')
            return temp
        
        except Exception as e:
            common.Print_Error('Preprocessor -> Load Files', e)
    
    
    """
        Log_Event()
        Used to generate event messages and store them in a list to be written later
    """
    def Log_Event(self, message='default', file='default'):
        try:
            if '[ERROR]' in message:
                if file != 'default':
                    self.error_log.append(message + ' || An error occured in file: ' + file + '\n')
                else:
                    self.error_log.append(message + '\n')
            elif '[INFO]' in message:
                if file != 'default':
                    self.change_log.append(message + ' || ' + file + '\n')
                else:
                    self.change_log.append(message + '\n')
            else:
                print('[INFO] The event was not logged because it did not fit in a logging protocol')
                print(message + ' || ' + file)
                
        except Exception as e:
            common.Print_Error('Preprocessor -> Log Event', e)

        
    """
        Save_Logs()
        Used at the end of the program to write the log files if any events have occured
    """
    def Save_Logs(self):
        try:
            # Save the error log
            if len(self.error_log) > 0:
                print("Saving Error Logs")
                log = open(self.base_path + "/Error_Log.txt", "w+")
                log.writelines(self.error_log)
                log.close()
                
            # Save the change log
            if len(self.change_log) > 0:
                print("Saving Event Logs")
                log = open(self.base_path + "/Change_Log.txt", "w+")
                log.writelines(self.change_log)
                log.close()
                
        except Exception as e:
            common.Print_Error('Preprocessor -> Save Logs', e)
            
            
    """
        Split()
        Split should be used whenever a file contains multiple passes through a bone. The major 
        indicator that multiple runs were stored in one file is an oscillation of the depth 
        between a negative value and positive values. The split files will be stored in a split
        folder and named according to the index of the zero crossing.
    """
    def Split(self):
        file_count = len(self.file_list) # Used as the total files to process
        
        # Notify the programmer which step is about to commense
        print(f'[INFO] Splitting the data of {file_count} CSV files from {self.file_list[0]}\n\tTo: {self.file_list[0].replace("Log","Split")}')
        
        for index, file in enumerate(self.file_list):
            try:
                df = pd.read_csv(file)
                
                indexes = [0]
                depth_sign_changed = False
                depth_col = df[self.valid_headers[0]]
                
                for index in range(df.shape[0]-1):
                    if depth_col[index] * depth_col[index + 1] < 0:
                        depth_sign_changed = True
                        
                    if depth_sign_changed:
                        depth_sign_changed = False
                        indexes.append(index)
                
                for i in range(len(indexes)-1):
                    temp_df = df.iloc[indexes[i]:indexes[i+1]]
                    temp_file = file.replace("Log", "Split")
                    temp_file = temp_file.replace(".csv", "-00" + str(i))
                    temp_file += ".csv"
                    temp_df.to_csv(temp_file)
                
                common.Print_Status('Split', index, file_count) # Print the current percent of processed files
                    
            except Exception as e:
                common.Print_Error('Preprocessor -> Split', e)
                
    
    """
        Truncate()
        Truncate should be used after all headers have been validated. Truncate finds two points
        1.) the deepest depth that the drill goes to and 2.) the point where the depth is equal 
        to zero. The dataframe is then constrained to values between zero and the maximum depth. 
        The csv file is overwritten by this function.
    """
    def Truncate(self, truncate_header='Depth mm', low=True, high=True):
        assert truncate_header in self.valid_headers, 'The truncation header must be in valid_headers'
        
        file_list = glob.glob(self.save_truncs_path)
        file_count = len(file_list)
        
        # Notify the programmer which step is about to commense
        print('[INFO] Truncating the data of ' + str(file_count) + ' CSV files from ' + self.save_truncs_path)
        
        # Iterate over all files and shorten their datasets so that they make 
        # sense for a time series. Look at the plots in the Original folder 
        # for more information
        for index, file in enumerate(file_list):
            try:
                df = pd.read_csv(file)      # Load the csv file with headers in the header
                col_count = df.shape[1]     # Record the number of columns
                
                # If the dataframe has more than the valid columns drop the invalid ones
                if col_count > len(self.valid_headers):
                    columns = df.columns.values
                    
                    for header in self.valid_headers:
                        try:
                            columns = columns[columns != header]
                        except Exception as e: 
                            common.Print_Error('Preprocessor -> Truncate', f'Header: {header}\n{e}')
                        
                    df = df.drop(columns=columns)
                
                min_index = 0
                
                if low: # Find the index of the first point that crosses the 0.0 depth point
                    col = df[truncate_header]
                    max_index = df[df[truncate_header] > 0.0].idxmax(axis="rows")
                    max_index = max_index[truncate_header]
                    
                    for min_index in range(max_index, -1, -1):
                        if col[min_index] <= 0.0:
                            min_index += 1
                            break
                
                max_index = df.shape[0]
                
                if high: # Find the index of the highest value for the selected header
                    max_index = df[df[truncate_header] > 0.0].idxmax(axis="rows")
                    max_index = max_index[truncate_header]

                df = df.iloc[min_index:max_index] # Only keep values in the acceptable range
                df = df.reset_index(drop=True)    # Restart the indexing at 0

                if df.shape[0] > self.min_length:
                    df.to_csv(file, index=False)
                else:
                    os.remove(file)   
                    print(f'\nRemoved File: {file}')
                    
                common.Print_Status('Truncate', index, file_count) # Update the process status
                
            except Exception as e:
                common.Print_Error('Preprocessor -> Truncate', f'File: {file}\n{e}\n{df.head()}')
                os.remove(file)
                print(f'\nRemoved File: {file}')
    
                
"""
The year is appended to the base data path. Newly generated files are stored
in a folder specified by save_path
Load -> data_path / load_from / year / *.csv
Truncs <- data_path / save_truncs / year / *.csv
Linear <- data_path / save_linear / year / *.csv
Transf <- data_path / save_transf / year / *.csv
Averag <- data_path / save_averag / year / *.csv
Veloci <- data_path / save_veloci / year / *.csv
"""
if __name__ == '__main__':
    data_path = '../Data/'
    #data_path = 'F:/blaney/Drill/Data/Composite/Expanded-Features/'
    load_from = 'Original'
    save_truncs = 'Truncated'
    save_linear = 'Linearized'
    save_transf = 'Transformaed fed'
    save_averag = 'Averaged'
    save_veloci = 'Linearized/Velocity'
    year = '*'
    
    windows=64
    
    processor = Preprocessor(name='temp', base_path=data_path, year=year, load=load_from, save1=save_truncs, save2=save_linear, save3=save_averag, save4=save_veloci, minimum_length=100)
    #processor.Change_Name(old='02-09', new='08-29')
    #processor.Split()
    #processor.Headers_Validate()
    processor.Truncate(truncate_header='Depth mm', low=True, high=True)
    processor.Linearize()
    #processor.Frequency_Transform(transform=2)
    #processor.Average_Transform(window=5)
    #processor.Velocity(window=windows)
    processor.Save_Logs()

