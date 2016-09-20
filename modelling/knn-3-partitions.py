#!/usr/bin/python
#This file partitions the data set into 3 datasets and produces results with respect to the validation set

import sys
import csv
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from calculate_accuracy import average_score
def data_prep(input_csv,x_cord1,x_cord2,y_cord1,y_cord2):
	
        #Read the csv file
	grid= pd.read_csv(input_csv, index_col = 'row_id')
  
        #Create data grid
        grid = grid [(grid['x'] > x_cord1) & (grid ['x'] < x_cord2) & (grid['y'] > y_cord1) & (grid['y'] < y_cord2)]
	#Feature engineering : Creating the features
	grid ['Hour'] = (grid ['time']/60)%24
	grid ['Weekday'] = (grid ['time']/(60*24))%7
	grid ['Month'] = (grid ['time']/(60*24*30))%12
	grid ['Year'] = grid ['time']/(60*24*365)
	grid ['Day'] = grid ['time']/(60*24) % 365
	grid.to_csv ('test-grid.csv')
	#print grid

	#time provides the same info. Delete the time variable
        del grid['time']

#Mulitply every useful feature with appropriate feature weight
def feature_engg (input_csv):
	#Read the csv file
        grid= pd.read_csv(input_csv, index_col = 'row_id')

	#Feature Engineering : Assigning appropriate weights for each feature
        feature_weights =  [125, 225, 10 , 1, 1, 1, 1]
        grid ['x'] = grid.x.values * feature_weights[0]
        grid ['y'] = grid.y.values * feature_weights[1]
        grid['Hour'] = grid.Hour.values * feature_weights[2]
        grid['Weekday'] = grid.Weekday.values * feature_weights[3]
        grid['Day'] = (grid.Day * feature_weights[4]).astype(int)
        grid['Month'] = grid.Month * feature_weights[5]
        grid['Year'] = grid.Year * feature_weights[6]

	return grid

#Create 3 paritions from the training set
def create_validation_set(df):
	mask  = np.random.rand(len(df)) < 0.2 #Random sampling in ~ 80 20 ratio
	validate = df[mask]
	train =  df[~mask]
	#Divide the validate into test and validate to create a hold-out test data set
	mask2 = np.random.rand(len(validate)) < 0.5
	valid = validate [mask2]
	test   = validate [~mask2] 
	valid.to_csv ('validate-random-sampling.csv')
	print ("**************Printing validatation set characterisitics**********************")
	print valid.describe ()
	test.to_csv ('test-random-sampling.csv')
	print ("**************Printing training set charactersitics****************************")
	print train.describe()
	print ("***************Printing test set characteristics********************************")
	print test.describe()
	train.to_csv ('train-random-sampling')
	return valid, train, test


#Function to delete outliers	
def train_prep (train_grid, cut_off):	
	#Row id with place_ids occuring less than cut_off number of times are dropped from the training data set
	placeid_count = train_grid.place_id.value_counts() #Counts the number of unique occurences of a place id
	flag = (placeid_count[train_grid.place_id.values] > cut_off).values # Adds boolean if the frequency of place_ids > cut_off
	modified_train_grid = train_grid.loc[flag] #Considers only the place_ids that occur more than the cutoff times
	return modified_train_grid

#Count number of unique place_id values in a given data set
def unique_count(data_set):
        g = data_set.groupby('place_id').place_id.nunique()
        return  g.sum()

#Implementing knn_classifier 
def knn_classifier (train_grid, validate):

	#Set index as row id for test_grid
	row_id = validate.index 	
	#Classifier knn
	le = LabelEncoder()
        place_ids = validate['place_id'].tolist()
	del validate['place_id']
    	y = le.fit_transform(train_grid.place_id.values)
	#Delete place_id in grid for
	del train_grid ['place_id']
	clf = KNeighborsClassifier(n_neighbors=15, weights='distance', 
                               metric='manhattan')
    	clf.fit(train_grid, y)
    	y_pred = clf.predict_proba(validate)
    	pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3])    
    	return pred_labels, row_id, place_ids


#Create the results file in a particular format
def create_results_file(preds):
	
	df_aux = pd.DataFrame (preds, dtype=str, columns = ['l1', 'l2', 'l3'])
	ds_sub = df_aux.l1.str.cat ([df_aux.l2, df_aux.l3], sep=' ')

	#Writing to a csv file
	ds_sub.name = 'place_id'
	ds_sub.to_csv ('sub_results.csv', index=True, header =True, index_label= 'row_id') 

	
	#Main program
if __name__ == '__main__':
	print ('Starting program')
	
	#data_prep function is called and the train and test grids are saved as .csv files beforehand to generate quicker results
	#x_cord1 = 3.25
        #x_cord2 = 3.75
        #y_cord1 = 3.25
        #y_cord2 = 3.75
        #train_csv = 'train.csv'
        #test_csv = 'test.csv'
        #train_grid = data_prep (train_csv,x_cord1,x_cord2,y_cord1,y_cord2)
	
	train_set = pd.read_csv ('train-grid.csv', index_col = 'row_id')
	print ("**********Printing training set charcteristics before partitioning********************")
	print train_set.describe()
	print ("Number of unique place_ids in the considered grid is %d" %unique_count(train_set))
	#Delete rows with place_ids occuring fewer than cut_off number of times (k=5) 
	new_train = train_prep (train_set, 5)
	print ("***************Printing training set charactersitics after deleting outliers***********")
	print new_train.describe()
	print ("Number of unique place_ids after deleting outliers is %d" %unique_count (new_train))
	
	#Create validation set
	validate, train, test = create_validation_set(new_train)
	#Create a .csv file of the new training set
	train.to_csv ('random-3-train.csv')
	#Feature engineering step
	train = feature_engg('random-3-train.csv')
	
	#knn classifier
	pred_labels,row_id, place_ids = knn_classifier (train, validate)
	"""The above function can be called differently to generate results with respect
	    to hold out test data set and the test grid provided in the competiton
	To generate results wit the test data set, call the function as,
	pred_labels,row_id, place_ids = knn_classifier (train, validate)""" 
	
	#Converting a list into a list of lists to call the average_score function
	list_of_lists = []
	for a in place_ids:
		list_of_lists.append([a])
	create_results_file (pred_labels)
	
	#Calculate Mean Average score @ 3
	average_score(list_of_lists, pred_labels)
	
	#If it were single prediction, calculate classification accuracy
	first_prediction = pred_labels[:,0]
	first_prediction_list = []
	for a in first_prediction:
		first_prediction_list.append([a])
		average_score (list_of_lists, first_prediction_list)
	 
	




 


