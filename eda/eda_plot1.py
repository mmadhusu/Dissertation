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
tls.set_credentials_file(username='meghana.m23', api_key='86xmxpsg75')



#Scatter plot of place_ids with respect to location (x and y, co-ordinate)
def draw_scatter(df):
	trace1 = go.Scatter(
		x = df.x,
		y = df.y,
		mode = 'markers',
		marker = dict(
			size=5,
			color=df.place_id,
			colorscale='Viridis',
			opacity=0.8
			),
		text = df.place_id,
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

	py.plot(data, filename='2d-graph')







 


