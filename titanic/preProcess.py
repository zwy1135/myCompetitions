# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 16:29:12 2014

@author: wy
"""

import numpy as np
import sklearn.preprocessing as pp
from sklearn.pipeline import Pipeline
import os
import pandas as pd

dataPath = './data'

sexDict = {'female':0,'male':1}
embarkDict = {'S':0,'C':1,'Q':2}

def loadData(filename):
    u'''
    读入数据
    '''
    rawData = pd.read_csv(os.path.join(dataPath,filename))
    return rawData
    
def buildDataset(raw,isTrainData = False):
    u'''
    清理无用数据并数值化，构建适合处理的数据集
    '''
    raw['SexNum'] = raw['Sex'].map(sexDict)
    raw['EmbarkedNum'] = raw['Embarked'].map(embarkDict)
    dropList = ['PassengerId','Name','Sex','Ticket','Cabin','Embarked']
    if isTrainData:
        dropList.append('Survived')
        label = raw['Survived']
    else:
        label = None
    data = raw.drop(dropList,axis = 1)
    return data,label
       
    
def impute(data,imputer = None):
    u'''
    补全数据
    '''
    if imputer == None:
        imputer = pp.Imputer(missing_values='NaN',strategy='mean',axis=0)
        imputer.fit(data)
    newData = imputer.transform(data)
    
    return newData,imputer
    
def standardize(data,scaler = None):
    u'''
    数据规范化
    '''
    if scaler == None:
        scaler = pp.StandardScaler()
        scaler.fit(data)
    newData = scaler.transform(data)
    
    return newData,scaler
    
def piplePreprocess(trainName,testName):
    data = loadData(trainName)
    data,label = buildDataset(data,True)
    
    test = loadData(testName)
    test,tmp = buildDataset(test)
    
    pipel = Pipeline([('impute',pp.Imputer()),('standardize',pp.StandardScaler())])
    data = pipel.fit_transform(data)
    test = pipel.transform(test)
    
    
    return data,label,test
    

def main():
    raw = loadData('train.csv')
    data,label = buildDataset(raw,True)
    data,imputer = impute(data)
    data,scaler = standardize(data)

    
    test = loadData('test.csv')
    test,tmp = buildDataset(test)
    test,tmp = impute(test,imputer)
    test,tmp = standardize(test,scaler)

    
    pipel = Pipeline([('impute',pp.Imputer()),('standardize',pp.StandardScaler())])
    data = pipel.fit_transform(data)
    test = pipel.transform(test)
    print(data,test)
    
        
    
if __name__ == '__main__':
    main()
    

