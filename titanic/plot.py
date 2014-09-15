# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 17:44:45 2014

@author: wy
"""

import numpy as np
import bokeh.charts as ch
import bokeh.plotting as pl

from preProcess import loadData,buildDataset

colormap = {0:"red",1:"blue"}

def plotHistogram(data):
    data_to_plot = data["Age"]
    data_to_plot = data_to_plot[-np.isnan(data_to_plot)]
    
    hist,edges = np.histogram(data_to_plot,bins=20)
    
    pl.output_file('histogram.html')
    pl.figure()
    pl.hold()
    pl.quad(top=hist,bottom=0,left=edges[:-1],right=edges[1:],
            fill_color="#036564", line_color="#033649",
            title="Age distribution",xlabel="Age",ylabel="Number")
    pl.show()
    
    
def plotScatter(data):
    data_to_plot = data.dropna()
    pl.output_file("scatter.html")
    pl.figure()
    pl.hold()
    pl.scatter(data["Fare"],data["Age"],
               color=data["Survived"].map(colormap),alpha=0.8,
               xlabel="Fare")
    pl.show()





if __name__=="__main__":
    data = loadData("train.csv")
    data,label = buildDataset(data,False)
    
    data_to_plot = data["Age"]
    data_to_plot = data_to_plot[-np.isnan(data_to_plot)]
    
    hist,edges = np.histogram(data_to_plot,bins=20)
    
    pl.output_file('histogram.html')
    pl.figure()
    pl.hold()
    pl.quad(top=hist,bottom=0,left=edges[:-1],right=edges[1:],
            fill_color="#036564", line_color="#033649",
            title="Age distribution",xlabel="Age",ylabel="Number")
            
    pl.line(np.average([edges[1:],edges[:-1]],axis=0),hist,
            line_width=5,color="red",alpha=0.8)            
            
    pl.show()


