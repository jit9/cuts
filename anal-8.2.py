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
print "keys are ... \n", keys

dpr.close()

#gd = np.load('good_dets_mr3_pa2_s14.npy')

gd = np.load('good_dets_mr3_pa2_s14_'+filename+'.npy')

print 'loaded gd;  number of good detectors = ', int(gd.shape[0])
#print 'good detectors'  ,  gd

i = 0
while i < 1056:

    if i in gd:
        gb_array = np.append(gb_array, 1)
    else:
        gb_array = np.append(gb_array, 0)
    i += 1
result['target'] =  gb_array

##
## read in the fourier transform data and add to result['data']
##

rf_h = np.load('p_h_a1408492556.1408492589.ar2.npy')
rf_l = np.load('p_l_a1408492556.1408492589.ar2.npy')
rms  = np.load('rms_1408492556.1408492589.ar2.npy')
rms2 = np.load('rms2_1408492556.1408492589.ar2.npy')


i = 0
while i<1056:
    if np.isnan(rf_h[i]):  rf_h[i] = 0.
    if np.isnan(rf_l[i]):  rf_l[i] = 0.
    if np.isnan(rms[i]):   rms[i]  = 0.
    if np.isnan(rms2[i]):  rms2[i] = 0.
    i += 1




#result['data'] = [ q['corrLive'], q['rmsLive'], q['sel'], q['kurtLive'],  q['DELive'], q['MFELive'], q['skewLive'], q['normLive'], q['darkRatioLive'],  q['jumpLive'], rf_h, rf_l]
kkk = [rf_h, rf_l, rms, rms2]
#kkk = [q['skewLive'],q['rmsLive'],  rf_h, rf_l, rms, rms2]
result['data'] = kkk 

# here are the keys used in Loic's cut_threshold code:
#result['data'] = [ q['corrLive'],  q['rmsLive'],  q['kurtLive'],  q['DELive'], q['MFELive'], q['skewLive'], q['normLive'], q['darkRatioLive'],   q['jumpLive'], q['gainLive']]




#print "result[data][0] ", result['data'][0]
#print "result[data][1] ", result['data'][1]

print ' >>>>>>   shapes\n\n', np.shape(result['target']),  '  ', np.shape(result['data'])

#print 'result[target] is  ...\n', result['target']

#print 'result[data]  is ...\n', result['data']

result['data'] = np.transpose(result['data'])

print np.shape(result['data'])

print ' ><><><><><  ',  np.shape(result['target'])
#
#DF = DataFrame(data = q )



#print DF.keys

DF_result = DataFrame(data = result['data'])
DF_target = DataFrame(data = result['target'])
#print 'DF_result ... '
#print DF_result


print "   ======   "
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
#print 'StandardScaler is OFF '
#print 'X_train:  \n', X_train
#X_train_scaled = X_train
#X_test_scaled = X_test
#result_scaled = result['data']



grr = pandas.plotting.scatter_matrix(resultDF_scaled, c = result['target'],  figsize=(7,7), marker = 'o', hist_kwds={'bins':20}, s=40, alpha = 0.8, cmap='jet')

#grr = pandas.plotting.scatter_matrix(resultDF,   figsize=(7,7), marker = 'o', hist_kwds={'bins':20}, s=40, alpha = 0.8, cmap='jet')
plt.show()



nNeighbors = 3
print 'nNeighbors =  ',  nNeighbors
knn = KNeighborsClassifier(n_neighbors = nNeighbors)
knn.fit(X_train, y_train)

print " ***********  "

y_pred = knn.predict(X_test)
print 'TOD name:  ', filename
print " predicted:  ", y_pred
print("Test set mean score:{:.3f}".format(np.mean(y_pred == y_test)))
print("Test set accuracy:{:.3f}".format(knn.score(X_test, y_test)))
print "\n   8=8=8=8=8=8=8"
