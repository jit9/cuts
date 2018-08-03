import matplotlib.pyplot as plt
import numpy as np
import cPickle
import os, sys
import math
import scipy
import pandas
from pandas import DataFrame
import sklearn
from sklearn import *
import sklearn.datasets
import string
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis








filename = '1408492556.1408492589.ar2'
#not tried: `filename = '1408498277.1408501897.ar2'
#filename = '1408583161.1408583225.ar2'
#filename = '1408497679.1408501297.ar2'


gk = np.zeros(0)
bk = np.zeros(0)
gk2 = np.zeros(0)
bk2 = np.zeros(0)
result = {}
gb_array = np.zeros(0)

#dpr = open('data_1408498277.1408501897.ar2', 'r')
#dpr = open('data_1408583161.1408583225.ar2', 'r')

dpr = open('data_'+filename,'r')
q = cPickle.load(dpr)


print "=====    =====  ===== "

keys = q.keys()
#print "keys are ... \n", keys
keys.remove('scan_freq')
keys.remove('live')
keys.remove('stable')
keys.remove('dark')
keys.remove('ff')
keys.remove('name')
keys.remove('DEDark')
keys.remove('sel')
print "edited keys .. ", keys

#
#  edited keys ..  ['respSel', 'corrLive', 'resp', 'rmsLive',  'corrDark', 'kurtLive', 'normDark', 'skewLive', 'normLive', 'darkRatioLive', 'MFELive', 'jumpDark', 'gainDark', 'gainLive', 'DELive', 'cal', 'rmsDark', 'jumpLive']
#


dpr.close()

#gd = np.load('good_dets_mr3_pa2_s14.npy')

gd = np.load('good_dets_mr3_pa2_s14_'+filename+'.npy')
#gd = np.load('good_dets_mr3_pa2_s14_1408492556.1408492589.ar2.npy')

print 'loaded gd;  number of good detectors = ', int(gd.shape[0])
#print 'good detectors'  ,  gd

k = 'gainLive'
#print 'q[DELive]  '  , np.shape(q['DELive']), ' and is...\n', q['DELive']
i = 0
while i < 1056:

    if i in gd:
        gb_array = np.append(gb_array, 1)
    else:
        gb_array = np.append(gb_array, 0)
    i += 1
result['target'] =  gb_array

##Remove 'sel' from the list
##

#result['data'] = [q['respSel'], q['corrLive'], q['resp'], q['rmsLive'],  q['corrDark'], q['kurtLive'], q['normDark'], q['DELive'], q['MFELive'], q['skewLive'], q['normLive'], q['darkRatioLive'], q['jumpDark'], q['gainDark'], q['cal'], q['rmsDark'], q['jumpLive']]

# here are the keys used in Loic's cut_threshold code:
result['data'] = [ q['corrLive'],  q['rmsLive'],  q['kurtLive'],  q['DELive'], q['MFELive'], q['skewLive'], q['normLive'], q['darkRatioLive'],   q['jumpLive'], q['gainLive']]

print '  ***   10 parameters  ***  '

#print "diagnostic  result[data] = " , result['data']


#print "result[data][0] ", result['data'][0]
#print "result[data][1] ", result['data'][1]

print ' >>>>>>   shapes\n\n', np.shape(result['target']),  '  ', np.shape(result['data'])

#print 'result[target] is  ...\n', result['target']

#print 'result[data]  is ...\n', result['data']

result['data'] = np.transpose(result['data'])

print np.shape(result['data'])

print ' ><><><><><  ',  np.shape(result['target'])
#
#DF = DataFrame(data = result )



#print DF.keys

DF_result = DataFrame(data = result['data'])
DF_target = DataFrame(data = result['target'])
print 'DF_result... ' , DF_result
#print DF_result


print "   ======   "
#print DF
print("keys of result: \n{}".format(result.keys()))


X_train, X_test, y_train, y_test =  train_test_split(result['data'],result['target'], random_state=784)

print("X-test shape: {}".format(X_test.shape))

print("x-train shape: {}".format(X_train.shape))
print("y_test shape: {}".format(y_test.shape))
print("y-train shape: {}".format(y_train.shape))

resultDF  = DataFrame(result['data'])
targetDF  = DataFrame(result['target'])

print 'shape of resultDF  ',  resultDF.shape
print 'shape of targetDF  ',  targetDF.shape

#result_dataframe = DataFrame(X_train)

scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

result_scaled  = scaler.transform(result['data'])
resultDF_scaled = DataFrame(result_scaled)

print 'StandardScaler is on.'
#print 'X_train:  \n', X_train



#grr = pandas.plotting.scatter_matrix(resultDF_scaled, c = result['target'],  figsize=(7,7), marker = 'o', hist_kwds={'bins':20}, s=40, alpha = 0.8, cmap='jet')

#grr = pandas.plotting.scatter_matrix(resultDF,   figsize=(7,7), marker = 'o', hist_kwds={'bins':20}, s=40, alpha = 0.8, cmap='jet')
#plt.show()

'''

nNeighbors = 2
print 'nNeighbors =  ',  nNeighbors
knn = KNeighborsClassifier(n_neighbors = nNeighbors)
knn.fit(X_train_scaled, y_train)
'''
cls = DecisionTreeClassifier(max_depth=5)


cls.fit(X_train_scaled, y_train)


print " ***********  "

#y_pred = knn.predict(X_test_scaled)

y_pred2 = cls.predict(X_test_scaled)

print 'TOD name:  ', filename
print " y_test ", y_test
#print " predicted:  ", y_pred
print " predicted2 ", y_pred2, "\n"
#print("Test set mean score:{:.3f}".format(np.mean(y_pred == y_test)))
print " mean pred2 ", np.mean(y_pred2 == y_test)
#print("Test set accuracy  - knn:{:.3f}".format(knn.score(X_test_scaled, y_test)))
print("Test set accuracy  - cls:  {:.3f}".format(cls.score(X_test_scaled, y_test)))

print "\n  <><><><><><><><><><><><><><>" 
