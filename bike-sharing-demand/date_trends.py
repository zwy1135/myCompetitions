# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 22:40:06 2014

@author: wy
"""
import os

import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

data_path = './data'

def readData(filename):
    data = pd.read_csv(os.path.join(data_path,filename))
    data.index = pd.DatetimeIndex(data.datetime)
    
    return data
    
    
def computeTrendModel(data):
    count = data['count']
    count_aver = count.resample("1D").dropna()
    index = count_aver.index.to_julian_date()
    trend_model = LinearRegression()
    #reg = ExtraTreesRegressor(n_estimators=100,verbose=1)
    shape = index.shape
    index = index.reshape((shape[0],1))
    values = count_aver.values#.reshape((shape[0],1))
    trend_model.fit(index,values)
    
    return trend_model
    
def computeDistributionModel(distr_data,distr_label):
    distr_model = ExtraTreesRegressor(n_estimators=100,verbose=1)
    distr_model.fit(distr_data,distr_label)
    
    return distr_model
    
def computeDistributionOfDay(model,data):
    count = data['count']
    count_aver = count.resample("1D").dropna()
    index = count_aver.index.to_julian_date()
    shape = index.shape
    index = index.reshape((shape[0],1))
    res = model.predict(index)
    res = pd.Series(res)
    res.index = count_aver.index
    res = res.resample("1h",fill_method="ffill")
    mean_count=res.loc[count.index].fillna(res[-1])
    distribution = count / mean_count
    
    return distribution
    
def buildDataset(data,is_train=False,distribution=None):
    drop_list = ['datetime']
    if is_train:
        data['dis'] = distribution
        drop_list = ['datetime','casual', 'registered','count']
    data['hour'] = data.index.hour
    data = data.drop(drop_list,axis=1)
    if is_train:
        label = data['dis']
        data = data.drop(['dis'],axis=1)
    else:
        label = None
        
    return data,label
    
def makePrediction(test,trend_model,distr_model):
    index = test.resample('1D').index
    test_index = index.to_julian_date()
    test_index = test_index.reshape((test_index.shape[0],1))
    test_trend = trend_model.predict(test_index)
    test_trend = pd.Series(test_trend)
    test_trend.index = index
    test_trend = test_trend.resample("1h",fill_method="ffill")
    test_trend = test_trend.loc[test.index].fillna(test_trend[-1])
    test_data,_ = buildDataset(test)
    test_distr = distr_model.predict(test_data)
    prediction = test_trend*test_distr
    
    return prediction
    
    
    
if __name__=='__main__':
    data = readData('train.csv')
    trend_model = computeTrendModel(data)
    distribution = computeDistributionOfDay(trend_model,data)
    distr_data,distr_label = buildDataset(data,True,distribution)
    distr_model = computeDistributionModel(distr_data,distr_label)
    test = readData('test.csv')
    prediction = makePrediction(test,trend_model,distr_model)
    
