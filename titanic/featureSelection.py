# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 22:39:39 2014

@author: wy
"""


from preProcess import piplePreprocess
from sklearn.svm import SVC

from sklearn.feature_selection import VarianceThreshold

def selection(data,test):
    selector = VarianceThreshold(threshold = 0.9*(1-0.9))
    data = selector.fit_transform(data)
    test = selector.transform(test)
    
    return data,test

from sklearn.svm import LinearSVC
if __name__ == "__main__":
    data,label,test = piplePreprocess('train.csv','test.csv')
    data,test = selection(data,test)
    classifier = SVC()
    classifier.fit(data,label)
    result = classifier.predict(test)
    np.savetxt("result.csv",result,fmt="%d")