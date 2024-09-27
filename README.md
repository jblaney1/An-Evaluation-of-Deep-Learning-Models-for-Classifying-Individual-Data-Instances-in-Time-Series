# An Evaluation of Deep Learning Models for Classifying Individual Data Instances in Time Series

[Link to Paper](https://www.proquest.com/openview/d169e8bd85f63a24471dd06d74b7ac32/1?pq-origsite=gscholar&cbl=18750&diss=y)

Advisor: Dr. Suresh Muknahallipatna

Author: Josh Blaney

Last Update: 01/26/2024

## Purpose:
To provide a brief overview of the purpose and provided functionality of each 
python file listed in the "Included with" section below. If you have more 
questions about any file specifically, the comments in the files themselves
should provide sufficient information to answer most questions.

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
 - transitioner.py
 
## Notes:
For more information about the project contact 
 - Dr. Suresh Muknahallipatna -> sureshm@uwyo.edu
 - Josh Blaney -> jblaney1@uwyo.edu

The comment structure for this work includes multi-line comment
blocks for each function and class that explaining the purpose 
of the function/class and how to interface with it. Functions 
which implement non-obvious functionality also have internal 
comments. The top of each file includes a comment block very 
similar to the one at the top of this file but also listing all
functions implemented within the file.

Some files include code at the bottom which illustrates how to 
use and test the functions within the file, and can be used as
the program interface wich runs the code to generate results 
from those files. For an example of this, see <stats.py>.

# File Information
## ann.py
Implements the pytorch model construction functions for both the 
deep neural networks and the generative adverserial networks.
This comment was intentionally kept short because the comments
in the file are extensive and provide a detailed account of 
each function and how they work. For examples of how to interface 
with these functions, see <ann_tester.py> or <gan_tester.py>.

## ann_tester.py
Constructs saved models using the <ann.py> functionality, trains them
using the <trainer.py> functionality, and tests them using the 
<trainer.py> functionality. This file provides both a mask of other
files and some nifty print statements which allow easy interpretation
of how well training went. When model training completes, the results
are automatically saved in a model folder.

## common.py
Provides extensively used functionality such as progress bar display,
result/report saving, and formatted print statements. The utility knife
of the project, any functionality which may be used more than once but 
does not fit in the scheme of any one file was put here.

## data_generator.py
Implements a statistical data generator algorithm which was used as a 
baseline to be compared with the generative adverserial network's
results. An extensive writeup can be found in the report 
"Perturbation Analysis for Creating Synthetic Data.pdf" for detailed 
information about the algorithm.

## dataloader.py
Provides custom dataloader functionality for both simple artificial
neural networks and generative adverserial networks. Specifically
written to be used with time series data, it performs windowing 
and batching automatically, but could easily be adapted to work
with other data.

## gan_tester.py
Provides similar functionality to <ann_tester.py>, but with the exception 
that everything has been expanded to accomodate the generative adverserial
networks. Beyond training and testing, this file also provides the 
functions to create batches of synthetic data as well as the functions for 
testing the similarity of synthetic data to reference data.

## labeler.py
An automated labeler which implements various labeling methods including
loading labels from a points file generated by <plotter.py>, loading 
labels from the original data using the "Alarms" column as the label, and
a semi-supervised model labeling scheme. The purpose of this file is to 
take labels which have been saved outside of the log files and create 
new files with the sequence data and labels.

## metrics.py
Provides an interface to various metric functions such as mean squared
error and the like. This file is only currently used by <gan_tester.py>
but has been written to be used during training if a specific metric
needs to be tracked in the training history.

## plotter.py
This file allows easy visualization of the data during pruning, labeling, 
and displaying results. It was actually used for both pruning and labeling,
therefore it includes functionality for moving and removing files as 
necessary.

## preprocessor.py
A completely automated file which brings in data and performs any of the 
preprocessing techniques discussed in the "Preprocessing" subsection of 
the "Methods" section of the file "blaney_thesis.pdf". This file only
changes the original files when specifically requested, otherwise a 
new folder is created where all the processing results are saved.

## stats.py
Implements the statistical analysis functionality used for preliminary
data analysis, saving the resulting statistical tables, and some nifty 
print statements so that when saving is not desirable, the results can
still be output for use outside the program.

## trainer.py
Provides the functionality for training and testing deep neural networks
as well as generative adverserial networks. This file also implements 
the history visualization functionality. This comment was intentionally 
kept short because the exact training and testing implementations are 
well commented and well documented in the various reports for this 
project. 

## transitioner.py
Analysis specifically requested to determine if the 
models are still usable even when the accuracy is not on par with 
medical standards. More to the point, this file tests for model 
predictions of transitions where class breaks occur in the data 
regardless of the correct classification by the model. This is to 
see if the models are identifying that a new class should be predicted
even when they do not know if it is the correct class.
