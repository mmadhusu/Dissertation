#!/usr/bin/python
import sys
import csv
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score
from average_precision import apk,mapk
import pdb


#pdb.set_trace()
#actual = [[1],[2]]
#predicted = [[3,8,9],[2,1,7]]
#where actual and predicted should be list of lists
def average_score(actual, predicted):
  global avg_score
  avg_score = 0
  for a,p in zip(actual,predicted):
	avg_score+= apk(a, p,3)
	#print (avg_score)
  mean_average_score = avg_score/len(actual)
  print ("Mean average score is %f "%mean_average_score)
  print ("Average score is %f" %avg_score)




