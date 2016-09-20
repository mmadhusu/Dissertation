#!/usr/bin/python
#This file lists all the functions that were used to perform EDA. Appropriate functions were called from the main program as and when needed. 

import pandas as pd
#import matplotlib
from first_model import train_prep
from eda_plot1 import draw_scatter
from eda_plot2 import scatter_plot_3d
test_csv = 'test.csv'
train_csv = 'train.csv'

test =  pd.read_csv(test_csv, index_col = 'row_id')
train = pd.read_csv(train_csv, index_col = 'row_id')

#Basic descriptive statistics
def basic_stats():
	print ("Descriptive statistics of test data set\n")
	print test.describe()
	print ("\n\nDescriptive statistics of train data set\n")
	print train.describe()

#Count number of unique place_id values in the training set
def unique_count():
	g = train.groupby('place_id').place_id.nunique()
	print g.sum()

# Check correlation among columns
def col_correlation():
	print ("Correlation analysis results\n")
	print train.corr(method = 'pearson')

#Create data grid
def create_data_grid (grid, x_cord1, x_cord2, y_cord1, y_cord2):
        print ("Creating data grid")
        grid = grid[(grid['x'] > x_cord1) & (grid['x'] < x_cord2) & (grid['y'] > y_cord1) & (grid['y'] < y_cord2)]
	return grid

#Select records with place_ids which occur more than cut_off times
#All other records are just dropped from the grid
def train_prep (train_grid, cut_off):
        placeid_count = train_grid.place_id.value_counts() #Counts the number of occurences of a place id
        flag = (placeid_count[train_grid.place_id.values] > cut_off).values # Adds boolean if the number of occurences  > cut_off
        modified_train_grid = train_grid.loc[flag] #Considers records whose place_id values occur more than cut_off times
        return modified_train_grid

#Check mean x and standard deviation within the same placeids
def placeid_explore(train, cut_off ):
	modified_train = train_prep(train,cut_off)
	print ("Summary of results with respect to x grouped by place_id")
	print modified_train['x'].groupby(modified_train['place_id']).describe()
	print ("Summary of results with respect to y grouped by place_id")
	print modified_train['y'].groupby(modified_train['place_id']).describe()
	return modified_train

#Transfrom 'Time' variable into other meaningful formats
def transform_time(grid):
	grid ['Hour'] = (grid ['time']/100)%24
	grid ['Weekday'] = (grid ['time']/(60*24))%7
	grid ['Month'] = (grid ['time']/(60*24*30))%12
	grid ['Year'] = grid ['time']/(60*24*365)
	grid ['Day'] = grid ['time']/(60*24) % 365
	return grid
		

#Main program
if __name__ == '__main__':
	#Take random 0.05% of the data
	grid = train.sample (frac = 0.05, replace = False)
	#Write the grid into a csv file for possible further use
	grid.to_csv('random-grid.csv')
	#Read the grid from the csv file
	grid =  pd.read_csv('random-grid.csv', index_col = 'row_id')
	#Select the rows whose place_ids occur more than 10 times and observe x and y variation
	grid = placeid_explore(grid, 10)
	#Transform the time stamp into meaningful time features
	modified_grid = transform_time(grid)
	#Sort the dataframe according to place_id for better visulaization
	new_grid = modified_grid.sort_values(by='place_id')
	#Select the first 500 records
	draw_scatter(new_grid[:500])
	#Visualize the frequency of check-ins with respect to time
	scatter_plot_3d (new_grid[:500])
	





