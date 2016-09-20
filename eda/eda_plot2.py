#!/usr/bin/python
import sys
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from pandas import DataFrame
import plotly.tools as tls
from first_model import train_prep
tls.set_credentials_file(username='meghana.m23', api_key='86xmxpsg75')



#3d scatter plot of place_ids with respect to location (x and y, co-ordinate) and 'Hour'
def scatter_plot_3d (df):
	trace1 = go.Scatter3d(
		x = df.x,
		y = df.y,
		z = df.Hour,
		mode = 'markers',
		marker = dict(
			size=2,
			color=(df.place_id + (df.x * 100)),
			colorscale='Viridis',
			opacity=0.8
	),
		text = df.place_id
	)	

	data = [trace1]
	layout = go.Layout(
		margin=dict(
	    	l=0,
	    	r=0,
            	b=0,
            	t=0
        	)
	)	
	fig = go.Figure(data=data, layout=layout)
	py.plot(data, filename='3d-graph')







 


