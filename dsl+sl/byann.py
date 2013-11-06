# -*- coding: utf-8 -*-
"""
Created on Tue Nov 05 21:08:50 2013

@author: wy
"""
import numpy as np
import os

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer,SigmoidLayer

import cPickle

input_num = 40
output_num = 2

def readdata(filename):
    datapath = './data/'
    fullname = os.path.join(datapath,filename)
    data = np.genfromtxt(fullname,dtype=np.float64,delimiter=',')
    print 'data read from %s'%fullname
    print data
    return data
    
def train(data,labels):
    samples = SupervisedDataSet(input_num,output_num)
    for i in range(len(data)):
        samples.addSample(data[i],(labels[i],1-labels[i]))
    
    nn = buildNetwork(input_num,20,10,output_num)
    trainer = BackpropTrainer(nn,samples)
    
    print 'train started.'
#    trainer.trainUntilConvergence()
    e = 100
    n=0
    while e>0.01:
        e=trainer.train()
        n+=1
        print n,' done,e=',e
        if n>=1000:break
    
    return nn
    
def test(nn,test_data):
    
    test = SupervisedDataSet(input_num,output_num)
    for i in range(len(test_data)):
        test.addSample(test_data[i],(1,1))
    
    #test = test
    print 'test started.'
    result = nn.activateOnDataset(test)
    #print result
    return np.argmin(result,axis=1)
    
if __name__=='__main__':
    data = readdata('train.csv')
    labels = readdata('trainLabels.csv')
    test_data = readdata('test.csv')
    
    nn = train(data,labels)
    result = test(nn,test_data)
    print result
    print len(result)
    id_number = range(1,len(result)+1)
    result_total = np.array([id_number,result]).T
    np.savetxt('result.csv',result_total,fmt='%d',delimiter=',')
    