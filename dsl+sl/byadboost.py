# -*- coding: utf-8 -*-
"""
Created on Wed Nov 06 20:35:11 2013

@author: wy
"""
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import os

def readdata(filename):
    datapath = './data/'
    fullname = os.path.join(datapath,filename)
    data = np.genfromtxt(fullname,dtype=np.float64,delimiter=',')
    print 'data read from %s'%fullname
    print data
    return data
    
if __name__=="__main__":
    data=readdata('train.csv')
    labels = readdata('trainLabels.csv')
    
    model = AdaBoostClassifier(n_estimators=10000)
    model.fit(data,labels)
    data_to_predict = readdata('test.csv')
    result = model.predict(data_to_predict)
    print result
    print len(result)
    id_number = range(1,len(result)+1)
    result_total = np.array([id_number,result]).T
    np.savetxt('result.csv',result_total,fmt='%d',delimiter=',')