"""
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 09/01/2023

Purpose:
Used to prune and label the MO dataset with labels. The two plotting modes
are interactive and regular. In interactive mode the mouse is used to select
points on the plot (left mouse button to add, right mouse button to remove).
The points are stored in a similarly named csv file and the console prompts
for a termination input before moving to the next plot. The regular plotting 
mode uses the console to input either a terminate input, delete input, or a 
continue input (y, d, enter, respectively). 

Functions:
    
Plotter()
 - Auto_Plot(save)
 - Auto_Plot_Interactive()
 - Build_File_List()
 - Create_Folder(file)
 - Interactive_Plot()
 - on_click(event)
 - Plot()
 - Process_Inputs(file)
 - Prompt(file)
 - Save_All(file)
 - Save_CSV(file)
 - Save_Plot()

Included with: 
 - ann.py
 - ann_tester.py
 - common.py
 - data_generator.py
 - dataloader.py
 - gan_tester.py
 - labeler.py
 - metrics.py
 - plotter.py (current file)
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
import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton

# In House Dependencies
import common


"""
The following functions have been collected into a class to reduce the amount 
of data passed between functions
"""


class Plotter:

    """
        __init__(path, year, headers, years=None, load_folder='Truncated_Linearized', save_folder='Pruned_Depth', sub_folder_logs='Log Files', sub_folder_plot='Plot Files', interactive=False, df=None, prompt_delay=10, start_index=0, plot_arrangment=None, c=None, overwrite=False)


    inputs:
        - path (string): Path to top level of the data
        - year (string): Year specifier
        - headers (list): List specifying x and y columns
        - years (list): List specifying subfolders to iterate through, see bottom of file for more
        - load_folder (string): Folder to load the csv data from
        - save_folder (string): Folder to save the plots to
        - sub_folder_logs (string): Folder holding the csv log files
        - sub_folder_plot (string): Folder holding the plot files
        - interactive (boolean): Are you going to label data?
        - df (dataframe): A dataframe can be specified if one has already been loaded
        - prompt_delay (int): How long to wait before asking if the user wants to quit
        - start_index (int): How many csv files to skip at the beginning
        - skip_interval (int): The number of files to skip between plots
        - plot_arrangment (tuple): Tuple of (rows, cols) for plot display
        - c (string): Specifies the header to draw the color data from
        - overwrite (boolean): Should new plots be generated over old ones?
    outputs:
        -
    """

    def __init__(self,
                 path,
                 year,
                 headers,
                 expected_points=3,
                 years=None,
                 load_folder=None,
                 save_folder=None,
                 move_folder=None,
                 sub_folder_logs=None,
                 sub_folder_plot=None,
                 interactive=False,
                 plot_legend=True,
                 low_from_data=False,
                 high_from_data=False,
                 df=None,
                 prompt_delay=10,
                 start_index=0,
                 skip_interval=0,
                 plot_arrangment=None,
                 c=None,
                 overwrite=False):

        self.data_path = path + '/' + load_folder + '/' + \
            year      # String to store the path to the data
        # String to store the individual years for folder validation
        self.years = years
        # String to store the path of the logs sub folder
        self.sub_folder_logs = sub_folder_logs
        # String to store the path of the plot sub folder
        self.sub_folder_plot = sub_folder_plot
        # String to store the path of the load from folder
        self.load_folder = load_folder
        # String to store the path of the save to folder
        self.save_folder = save_folder
        # String to store the path of the move to folder
        self.move_folder = move_folder
        # List of file locations
        self.file_list = []
        if type(headers[0]) == list:
            self.x = headers
            self.y = None
        else:
            # String which stores the saved x coordinate
            self.x = headers[0]
            # String which stores the saved y coordinate
            self.y = headers[1:] if len(headers) > 2 else headers[1]
        # String storing the header of the color column
        self.c = c
        self.df = df                                                # Blank dataframe
        # List of the x coordinates to save
        self.saved_x = []
        # List of the y coordinates marked, only used for plotting
        self.saved_y = []
        # Int to track the number of mouse clicks during interactive plotting
        self.clicks = 0
        # Count of the currently stored x coordinates, max of 4
        self.added_points = 0
        # Boolean to terminate the program
        self.done = False
        # Boolean to track if this is an interactive plot session
        self.interactive = interactive
        # Boolean to track if the low class mark is drawn from the data
        self.low_from_data = low_from_data
        # Boolean to track if the high class mark is drawn from the data
        self.high_from_data = high_from_data
        # Figure and axis objects used to keep one plot window open for all plots
        if plot_arrangment is None:
            self.fig, self.ax = plt.subplots()
            self.multiplot = False
        else:
            self.fig, self.ax = plt.subplots(
                plot_arrangment[0], plot_arrangment[1])
            self.multiplot = True

        scaler = 3
        # Statement to control the figure layout
        self.fig.tight_layout(pad=3.5)
        self.fig.set_size_inches(4*scaler, 2*scaler)
        # Int which identifies the number of labels to expect
        self.expected_points = expected_points
        # Int which identifies how often to prompt the labeler
        self.delay = prompt_delay
        # Int which identifies how many data examples to skip at the start
        self.start_index = start_index
        # Int which identifies the number of files to skip between saved plots
        self.skip_interval = skip_interval
        # Int which counts the number of files removed during pruning
        self.remove_count = 0

        # Dictionary of the color scheme to use when plotting
        self.colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}
        # Local file tracking variable used in debugging
        self.file = None
        # When generating plots overwrite existing files?
        self.overwrite = overwrite
        
        self.plot_legend = plot_legend
        self.legend = []

        for key in self.colors.keys():
            self.legend.append(f'Class {key+1}')


    """
        Auto_Plot(save)
        A function to automate many csv plotting for both dataset pruning and automatic plot generation. 
        Autmatic plot generation iterates over all files found and plots according to the parameters
        set at the init. Pruning is performed by plotting each file and waiting for the users input.
        
    inputs: 
        - save (boolean): Run the automated generation and save routine? or Prune?
    outputs:
        - 
    """
    def Auto_Plot(self, save=False):
        try:
            # Used by prompt to track the percent of files processed
            index = 0
            self.done = False                   # Used to terminate the program
            # Used to track the total files to process
            file_count = len(self.file_list)

#            if save:
#                mpl.use('Agg')

            plot_range = range(self.start_index, file_count, skip_interval)
            plot_count = len(plot_range)

            # Iterate over all files and either plot them and prompt the user to
            # delete or just plot and save without prompting the user
            for i in plot_range:

                if self.done:
                    break             # If the user wants to terminate the program done is True
                plotted = False
                self.file = self.file_list[i]

                # If the save boolean has been set an automated process is run which
                # iterates over all files, generates plots, and saves the plots
                if save:
                    index += 1
                    common.Print_Status('Auto Plot', index, plot_count)
                    plot_file = self.file.replace(
                        self.sub_folder_logs, self.sub_folder_plot)
                    plot_file = plot_file.replace('csv', 'png')

                    if self.overwrite:
                        # Load a csv file with the header in the header
                        self.df = pd.read_csv(self.file)
                        # Run the plotting routine
                        self.Plot(show=False)
                        self.Save_Plot(self.file)
                        plotted = True

                    elif not common.Validate_File(plot_file):
                        # Load a csv file with the header in the header
                        self.df = pd.read_csv(self.file)
                        # Run the plotting routine
                        self.Plot(show=False)
                        self.Save_Plot(self.file)
                        plotted = True

                else:
                    self.file = self.file_list[i]
                    # Load a csv file with the header in the header
                    self.df = pd.read_csv(self.file)
                    self.Plot()                          # Run the plotting routine
                    print(self.file + " | File Number: " + str(i+1) +
                          " | Files Pruned: " + str(self.remove_count))
                    self.done = self.Prompt(self.file)
                    plotted = True

                if plotted and not self.multiplot:
                    plt.cla()
                elif plotted and self.multiplot:
                    if type(self.c) is list or self.y is None:
                        for i in range(len(self.ax)):
                            # Clear the plot before adding new data
                            self.ax[i].cla()
                    else:
                        plt.cla()

            plt.close()     # Close the plotting window on program close

        except Exception as e:
            common.Print_Error('Plotter -> Auto Plot', e)


    """
        Auto_Plot_Interactive()
        A function to automate many csv plotting for point labeling. A plot is generated and the program waits
        for the user to click expected_points times on the plot, each time marking the x coordinate. The points
        are then saved in a csv file to be used by labeler.py
    """
    def Auto_Plot_Interactive(self):
        self.done = False                   # Used to terminate the program
        file_count = len(self.file_list)
        index = 1

        plot_range = range(self.start_index, file_count, skip_interval)
        plot_count = len(plot_range)

        # Iterate over all files and create interactive plots for each
        for i in plot_range:
            self.file = self.file_list[i]
            # Notify the programmer of the current file being processed
            print(self.file + " | File Number: " + str(i + 1))
            if self.done:
                # If the user wants to terminate the program done is True
                break
            # Read the csv file and store in a dataframe
            self.df = pd.read_csv(self.file)
            # Run the Interactive Plotting routine
            self.Interactive_Plot()
            # Process and save the inputs from the interactive plot
            self.Process_Inputs(self.file)
            
            if index % self.delay == 0:
                # Prompt the user to see if the program should terminate
                self.done = self.Prompt(self.file)
                
            # Clear the plot before adding new data
            if type(self.ax) is np.ndarray:
                for axes in self.ax:
                    axes.cla()
            else:
                plt.cla()
                
            index += 1
        # Close the plotting window on program close
        plt.close()


    """
        Build_File_List()
        A function used to combine the init path parameters and validate the list of files which are loaded can also be saved.
    """
    def Build_File_List(self):
        sub_path = self.data_path + '/' + self.sub_folder_logs
        file_path = sub_path + '/*.csv'
        self.file_list = glob.glob(file_path)
        print('Found ' + str(len(self.file_list)) + ' files in ' + file_path)

        if "*" not in self.data_path:
            common.Validate_Dir(sub_path)
            sub_path = sub_path.replace(self.load_folder, self.save_folder)
            common.Validate_Dir(sub_path)
            sub_path = sub_path.replace(
                self.sub_folder_logs, self.sub_folder_plot)
            common.Validate_Dir(sub_path)
        elif self.years is not None:
            sub_path = sub_path.replace("*", self.years[0])
            sub_path = sub_path.replace(self.load_folder, self.save_folder)
            sub_path = sub_path.replace(
                self.sub_folder_logs, self.sub_folder_plot)

            for index in range(1, len(self.years)):
                common.Validate_Dir(sub_path)
                sub_path = sub_path.replace(
                    self.years[index-1], self.years[index])


    """
        Create_Folder(file)
        A function which attempts a folder creation when writing a file fails
        
        inptus:
         - file (string): The location which failed a write
    """
    def Create_Folder(self, file):
        try:
            common.Print('Attempting Folder Creation')
            split_file = file.split('\\')
            split_file[-1] = ''
            split_file = '\\'.join(split_file)
            common.Validate_Dir(split_file)
            plt.savefig(fname=file)

        except Exception as e:
            common.Print_Error('Plotter -> Create Folder',
                               f'File: {file}\n{e}')


    """
        Interactive_Plot()
        Interactive Plot updates the plotting window and then halts the program repeatedly 
        for 0.1 seconds. This halting process will end when expected_points have been 
        recieved at the polotting window.
    """
    def Interactive_Plot(self):
        try:
            self.added_points = 0                           # No points have been chosen
            self.Plot()

            # Halt the program until 4 points have been chosen on the plot
            while self.clicks < self.expected_points + 1:
                plt.pause(0.1)

            self.clicks = 0

        except Exception as e:
            common.Print_Error('Interactive Plot', e)


    """
        on_click(event)
        A function to handle click events on the plotting window. The left mouse button adds 
        points, the right mouse button removes points, the plot is updated with each point 
        individually, and the process ends when 4 points are recorded in the list.
    
    inputs:
        - event (event): A mouse click event which needs to be processed.
    outputs:
        - 
    """
    def on_click(self, event):
        # Check that the mouse is within the plotting window
        if event.inaxes:
            # Check which mouse button was pressed
            if event.button is MouseButton.LEFT:
                if self.clicks < self.expected_points:
                    # Save the x position of the mouse when the click occurred.
                    self.saved_x.append(event.xdata)
                    # Save the y position of the mouse when the click occurred.
                    self.saved_y.append(event.ydata)
                self.clicks += 1                        # Increment the number of saved clicks
                
            elif event.button is MouseButton.RIGHT:
                # Check that there is at least one entry to remove
                if len(self.saved_x) > 0:
                    self.clicks -= 1                # Decrement the number of saved clicks
                    # Remove the latest entry from the list of x values
                    self.saved_x.pop(-1)
                    # Remove the latest entry from the list of y values
                    self.saved_y.pop(-1)

            # Update the current number of saved points
            self.added_points = len(self.saved_x)

            # Clear the plotting window before adding more data
            plt.cla()

            self.Plot()
            
            # Update the marked points
            if type(self.ax) is np.ndarray:
                self.ax[-1].scatter(self.saved_x, self.saved_y, c='m')
            else:
                self.ax.scatter(self.saved_x, self.saved_y, c='m')

            # Enable verticle line plotting for all saved x points to enhance accuracy
            for x in self.saved_x:
                plt.axvline(x=x, color='r')


    """
        Plot()
        A function to update the plotting window each time new data is processed.
    """
    def Plot(self, show=True):
        try:
            # Automatically change the x label from INDEX to TIME when applicable
            if self.x == 'index':
                x_label = 'Time'
            else:
                x_label = self.x
                
            if self.x is None:  # Plotting for all data in dataframe
                self.ax.grid(True)
                self.ax.plot(self.df)

            elif self.y is None:  # Multiplot for Pruning and Labeling
                data = self.df.reset_index()
                for i, header in enumerate(self.x):
                    for index, entry in enumerate(self.legend):
                        scatter_data = data[data[self.c] == index]
                        y = scatter_data[header[1:]]
                        x = scatter_data[header[0]]
                        self.ax[i].scatter(x, y, c=self.colors[index], label=entry)

                    self.ax[i].grid(True)
                    self.ax[i].set_ylabel(header[1:])
                    self.ax[i].set_xlabel(x_label)
                    self.ax[i].title.set_text(header)
                    
                    if self.plot_legend:
                        self.ax[i].legend(loc='upper left')

            else:  # Plotting for intermediate preprocessing results
                # If there is no colorscheme specified, plot normally
                if self.c is None:
                    data = self.df.reset_index()                        
                    self.ax.grid(True)
                    self.ax.plot(data[self.x], data[self.y])
                    self.ax.set_ylabel(self.y)
                    self.ax.set_xlabel(x_label)
                    
                else:
                    # If the colorscheme is a list, iterate through the colors
                    if type(self.c) is list:
                        for i in range(len(self.c)):
                            for index, entry in enumerate(self.legend):
                                data = self.df[self.df[self.c[i]] == index].reset_index()
                                self.ax[i].scatter(
                                    data['index'], data['feature'], c=self.colors[index], label=entry)

                            self.ax[i].grid(True)
                            self.ax[i].set_ylabel(self.y)
                            self.ax[i].set_xlabel(x_label)
                            self.ax[i].title.set_text(self.c[i])
                            if self.plot_legend:
                                self.ax[i].legend(loc='upper left')

                    else:
                        # If multiple dependent variables are specified, multiplot
                        if type(self.y) is list:
                            j = 0
                            for y in self.y:
                                for index, entry in enumerate(self.legend):
                                    data = self.df[self.df[self.c]
                                                   == index].reset_index()
                                    if j > 0:
                                        self.ax.scatter(
                                            data[self.x], data[y], c=self.colors[index])
                                    else:
                                        self.ax.scatter(
                                            data[self.x], data[y], c=self.colors[index], label=entry)
                                j += 1
                        else:
                            for index, entry in enumerate(self.legend):
                                data = self.df[self.df[self.c] == index].reset_index()
                                self.ax.scatter(data[self.x], data[self.y], c=self.colors[index], label=entry)

                        self.ax.grid(True)
                        self.ax.set_ylabel(self.y)
                        self.ax.set_xlabel(x_label)
                        
                        if self.plot_legend:
                            self.ax.legend(loc='upper left')

            if show:
                plt.show()

            if not self.interactive and show:
                plt.pause(0.5)

        except Exception as e:
            common.Print_Error('Plot\n' + self.file, e)


    """
        Process_Inputs(file)
        A function to save the marked x labels in a csv file and clear the marked points lists
        
    inputs:
        - file (string): file to save the labels to
    outputs:
        -
    """
    def Process_Inputs(self, file):

        try:
            # Change the save location to the Point folder
            file = file.replace("Log", "Point")
            file = file.replace(self.load_folder, self.save_folder)
            
            # Convert the x points list to a dataframe
            if self.low_from_data: # Use the labels from the low data in the csv
                data = self.df.reset_index()[self.df[self.c]==0]
                low_depth = data['index'].values[-1]
                self.saved_x = [low_depth] + self.saved_x
                
            if self.high_from_data: # Use the labels from the high data in the csv
                data = self.df.reset_index()[self.df[self.c]==2]
                high_depth = data['index'].values[0]
                self.saved_x.append(high_depth)
                
            temp = pd.DataFrame(self.saved_x)
            
            # Save the dataframe to a new file
            temp.to_csv(file, index=False)
            self.saved_x.clear()                # Clear the list of x points
            self.saved_y.clear()                # Clear the list of y points

        except Exception as e:
            common.Print_Error('Plotter -> Process Inputs',
                               f'File: {file}\n{e}')

            # Attempt folder creation and then try saving the csv again
            self.Create_Folder(file)
            temp.to_csv(file, index=False)
            self.saved_x.clear()                # Clear the list of x points
            self.saved_y.clear()                # Clear the list of y points
            
            
    """
        Prompt(file)
        A function which displays a status bar during manual processing. If the user is ready to quit 
        they have the option to and can write down the last file processed to start there again later.
        Files which should be pruned can be removed from the dataset.
        
    inputs:
        - file (string): The file being processed
    outputs:
        - (boolean): should the process stop?
    """
    def Prompt(self, file):
        x = ''

        if self.interactive:
            print('Finished? [y] \t\t Continue? [enter]')
        else:
            print('Finished? [y] \t\t No Save? [n] \t\t Continue? [enter]')

        x = input()

        if 'n' not in x and not self.interactive:
            self.Save_CSV(file)
        elif self.save_folder == self.load_folder:
            os.remove(file)
            self.remove_count += 1
        else:
            file = file.replace(self.load_folder, self.move_folder)
            self.Save_CSV(file)
            self.remove_count += 1

        done = True if 'y' in x else False
        return done
    

    """
        Save_All(file)
        A function to save both the csv file and the plot file.
        
    inputs:
        - file (string): Input file which was processed
    outputs:
        - 
    """
    def Save_All(self, file):
        try:
            self.Save_Plot(file)
            self.Save_CSV(file)

        except Exception as e:
            common.Print_Error('Plotter -> Save All', f'File: {file}\n{e}')
            

    """
        Save_CSV(file)
        A function to adjust the file location and save the dataframe as csv
        
    inputs:
        - file (string): Input file which was processed
    outputs:
        - 
    """
    def Save_CSV(self, file):
        try:
            # Change the folder to the save folder
            file = file.replace(self.load_folder,
                                self.save_folder)

            # Change the subfolder to the logs folder
            file = file.replace(self.sub_folder_plot,
                                self.sub_folder_logs)

            # Change the file extension to csv
            file = file.replace("png",
                                "csv")

            # Save the entire dataframe but without indexes
            self.df.to_csv(file, index=False)

        except Exception as e:
            common.Print_Error('Plotter -> Save CSV', f'File: {file}\n{e}')

            # Attempt folder creation and then try saving the csv again
            self.Create_Folder(file)
            self.df.to_csv(file, index=False)
            
            
    """
        Save_Plot(file)
        A function to adjust the file location and save matplotlib figures as png
        
    inputs:
        - file (string): File which was processed
    outputs:
        - 
    """
    def Save_Plot(self, file):
        try:
            # Change the folder to the save folder
            file = file.replace(self.load_folder,
                                self.save_folder)

            # Change the subfolder to the plots folder
            file = file.replace(self.sub_folder_logs,
                                self.sub_folder_plot)  # Change the folder for plots

            # Change the file extension to the plot extension
            file = file.replace("csv",
                                "png")

            # Save the entire figure
            plt.savefig(fname=file)

        except Exception as e:
            common.Print_Error('Plotter -> Save File', f'File: {file}\n{e}')

            # Attempt folder creation and then try saving the plot again
            self.Create_Folder(file)
            plt.savefig(fname=file)


"""
The file path is constructed using the following arrangement

Load -> base_path + '/' + load_folder + '/' + year + '/' + sub_folder_logs + '/*.csv'
Save CSV <- base_path + '/' + save_folder + '/' + year + '/' + sub_folder_logs + '/*.csv'
Save PNG <- base_path + '/' + save_folder + '/' + year + '/' + sub_folder_plot + '/*.png'
Save Mve <- base_path + '/' + move_folder + '/' + year + '/' + sub_folder + '/*.png' (or csv)
"""
if __name__ == '__main__':

    # if save_plots is False the pruning routine runs
    save_plots = True  # Should the automatic plotting routine run?
    interactive = False  # Should the interactive labeling routine run?
    delay = 1000        # After how many files should the program offer to exit?
    start_index = 0  # What index should the program start at?
    skip_interval = 1 # How many files should be skipped between plots
    overwrite = True    # Should new plots overwrite old ones?
    plot_legend = False
    low_from_data=True
    high_from_data=True
    model = None
    # Expected number of label points during labeling
    expected_points = 1

    base_path = '../Data'
    #base_path = 'A:/School/Research/Drill/Data/Composite/Expanded-Features/'
    #base_path = 'F:/blaney/Drill/Data/Composite/Expanded-Features'
    years = []
    plot_types = ['Depth']
#    plot_types = ['Force']
#    plot_types = ['Depth', 'Force', 'Depth-vs-Force']              # Header select variable for plotting
    plot_headers = {'Depth': ['index', 'Depth mm'],
                    'Force': ['index', 'Force lbf'],
                    'Depth-vs-Force': ['Depth mm', 'Force lbf'],
                    'Pruning': [['index', 'Depth mm'], ['index', 'Force lbf'], ['Depth mm', 'Force lbf']],
                    'Labeling': [['index', 'Force lbf'], ['Depth mm', 'Force lbf'], ['index', 'Depth mm']]}

    
    if model is None:  # Plot data from log files
        # Year subfolder specifier. See multi-line comment above
        year = '2024-02-12_09-28-00_post'
        years = None
        # Folder where the logs and plots should be loaded
        load_folder = 'generated'
        save_folder = 'generated'               # Folder where the logs and plots should be save
        move_folder = ''
        # Folder where the log files are saved
        sub_folder_logs = 'Log Files'

        plot_arrangement = None#[3,1]
        c = None

    else:  # Plot data from model predictions
        for i in range(2):
            years.append(model + str(i))

        year = '*'
        load_folder = model
        save_folder = model
        sub_folder_logs = 'predictions'
        sub_folder_plot = 'plots'
        headers = [None, 'Depth mm']
        plot_arrangement = [1, 2]
        c = ['prediction', 'label']

    plt.close('all')

    for plot_type in plot_types:
        headers = plot_headers[plot_type]
        sub_folder_plot = f'Plot Files/{plot_type}'
        plotter = Plotter(path=base_path,
                          year=year,
                          headers=headers,
                          expected_points=expected_points,
                          years=years,
                          load_folder=load_folder,
                          save_folder=save_folder,
                          move_folder=move_folder,
                          sub_folder_logs=sub_folder_logs,
                          sub_folder_plot=sub_folder_plot,
                          interactive=interactive,
                          plot_legend=plot_legend,
                          low_from_data=low_from_data,
                          high_from_data=high_from_data,
                          df=None,
                          prompt_delay=delay,
                          start_index=start_index,
                          skip_interval=skip_interval,
                          plot_arrangment=plot_arrangement,
                          c=c,
                          overwrite=overwrite)

        plotter.Build_File_List()

        if interactive:
            binding_id = plt.connect('button_press_event', plotter.on_click)
            plotter.Auto_Plot_Interactive()
        else:
            plotter.Auto_Plot(save=save_plots)
