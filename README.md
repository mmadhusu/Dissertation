# Dissertation

his folder contains the scrips that were used to implement the modelling project and the corresponding results.
This file gives a brief description of each of the folders,

#Folder "eda" : It comprises 3 files used for EDA
1. eda.py : This file lists all the functions used to carry out EDA
2. eda_plot1.py : This file lists the fucntion to create a 2d catter plot of place_ids with respect to location using Plotly
3. eda_plot2.py : This file lists the fucntion to create a 3d scatter plot of place_ids with respect to location and 'Hour' using Plotly

#Folder "modelling" : It comprises 3 files used for modelling

1. knn-3-partitions.py : Primary file used for modelling. It was modified slightly for use with different data sets , i.e., holdout and test data
set provided by the competition.
2. average_precision.py : This is the script authored by Ben Hamner to implement MAP@n accuracy measures.
3. calculate_precision.py : The average_precision.py script could not be used directly as the structure of the final results
were different than what was expected. This file makes appropriate changes to the structure of the final results file and calls the function
listed in average_precision.py file

#Folder "results" : It lists all the required and auxiliary results
placeids-1km-holdout.csv  : Final prediction list generated for holdout set of 1km grid
placeids-1km-test.csv     : Final prediction list generated for test set of 1km grid
placeids-1km-validate.csv : Final prediction list generated for validation  set of 1km grid
placeids-250m-holdout.csv : Final prediction list generated for holdout set of 250m grid
placeids-250m-test.csv    : Final prediction list generated for test set of 250m grid
placeids-250m-validate.csv : Final prediction list generated for validation  set of 250m grid
results-1km-holdout.txt : Summary of results of the holdout set, 1km grid
results-1km-validate.txt : Summary of results of validate set, 1km grid
results-250m-holdout.txt : Summary of results of holdout set, 250m grid
results-250m-validate.txt : Summary of results of validate set, 250m grid
Sub folder auxiliary-results : Lists three files with summary of results with 3 other grids

