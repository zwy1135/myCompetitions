# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 19:01:25 2014

@author: wy
"""

from preProcess import piplePreprocess
from featureSelection import selection
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier
from sklearn.grid_search import GridSearchCV as GSC
import numpy as np

from sklearn.cross_validation import ShuffleSplit,cross_val_score
import pickle

svc_parameter = {'C':np.logspace(-3,3,num=7),
                 'kernel':['linear','rbf','sigmoid'],
                 'degree':np.arange(1,5),
                 'gamma':np.arange(0,1,0.1),
                 'coef0':np.arange(0,1,0.1),
                 }
                 
tree_parameter = {'n_estimators':np.arange(1,20),
                  'criterion':['gini','entropy'],
                  'max_features':['sqrt',"log2",None],
                  'min_samples_leaf':np.arange(1,10),
                  'bootstrap':[True,False],
                  #'oob_score':[True,False]
                  }

if __name__ == "__main__":
    data,label,test = piplePreprocess('train.csv','test.csv')
    data,test = selection(data,test)
    classifier = SVC(max_iter=1e6)
#    classifier = ExtraTreesClassifier()
#    classifier = LinearSVC()
    cv = ShuffleSplit(data.shape[0],10)
#    scores = cross_val_score(classifier,data,label,cv=cv)
#    print('Accuracy is %.2f (+/- %.2f)'%(scores.mean(),scores.std()*2))    
    
    gsc = GSC(classifier,tree_parameter,cv=cv,verbose=2)
    print(data.shape,label.shape)
    print("fitting data")
    gsc.fit(data,label)
    print("fitdone")
    bestClassifier = gsc.best_estimator_
    score = gsc.best_score_
    print(best)
    print(score)
    with open('gsc_tree.pickle','wb') as f:
        pickle.dump(gsc,f)
    
    with open('best_tree.pickle','wb') as f:
        pickle.dump(best,f)
    
    
