# -*- coding: utf-8 -*-
"""
Created on Tue Nov 05 16:15:01 2013

@author: wy
"""

from sklearn.linear_model import SGDClassifier
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
    
    model = SGDClassifier()
    model.fit(data,labels)
    data_to_predict = readdata('test.csv')
    result = model.predict(data_to_predict)
    print result
    print len(result)
    id_number = range(1,len(result)+1)
    result_total = np.array([id_number,result]).T
    np.savetxt('result.csv',result_total,fmt='%d',delimiter=',')