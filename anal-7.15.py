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
from sklearn.tree import DecisionTreeClassifier


gk = np.zeros(0)
bk = np.zeros(0)
gk2 = np.zeros(0)
bk2 = np.zeros(0)
result = {}
d_test_results = {}
gb_array = np.zeros(0)
#dLong = np.zeros(9410016)
dLong = np.zeros(8632800)
d = np.zeros(1000000)
d_test = np.zeros(3000000)


#data = cPickle.load(open('mr3_pa2_s14_results.pickle','r'))
data = cPickle.load(open('mr3_pa3_s16_results.pickle','r'))
keys = data.keys()
print "keys are ... \n", keys,  '\n'



culledKeys = ['sel', 'respSel', 'corrLive', 'rmsLive', 'kurtLive', 'normDark', 'skewLive', 'normLive', 'darkRatioLive', 'MEFLive', 'jumpDark', 'gainDark', 'gainLive' , 'DELive', 'cal', 'rmsDark', 'jumpLive']
print 'culledKeys   ', culledKeys,  '\n'
DF      = DataFrame()
DFa     = DataFrame()
DTesta = DataFrame()
DTest  = DataFrame()
#dataKeys = culledKeys
resultKeys=  ['respSel', 'corrLive', 'rmsLive', 'kurtLive', 'normDark', 'skewLive', 'normLive', 'darkRatioLive', 'MFELive', 'jumpDark', 'gainDark', 'gainLive' , 'DELive', 'cal', 'rmsDark', 'jumpLive']   
for k in resultKeys:
        print ' working on ', k
        dLong = data[k].reshape(8632800)
        d = dLong[1000000:1005000]
        d_test = dLong[2990000:2999999]
        result[k] = d 
        d_test_results[k] = d_test 
        DFa = DataFrame({k: d})
        DF = pandas.concat([DF,DFa], axis =1)
        print 'shape of result[', k, '] is  ', np.shape(result[k])

        DTesta = DataFrame({k:d_test})
        DTest  = pandas.concat([DTest, DTesta], axis = 1)
        print 'shape of d_test_results[',k,'] is  ', np.shape(d_test_results[k])
        
print "\n"

DF = DF.fillna(DF.mean())

dLong = data['sel'].reshape(8632800)
d_truth = dLong[1000000:1005000].reshape(-1,1)
#test_results = dLong[2990000:2999999].reshape(-1,1)
DF_target = DataFrame(d)
print 'shape of result[k]  is  ',  np.shape(result)

print "   =====    =====  ===== "




X_train, X_test, y_train, y_test =  train_test_split(DF, DF_target, random_state=386)

#X_train, X_test, y_train, y_test =  train_test_split(result, d_truth, random_state=524)


print 'got to here'

print("X-test shape: {}".format(X_test.shape))

print("y_test shape: {}".format(y_test.shape))
print("X-train shape: {}".format(X_train.shape))
print("y-train shape: {}".format(y_train.shape))
#print("test_results shape: {}".format(test_results.shape))

scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)


DF_target = DF_target.fillna(value=0.0)
### turn off scaling for DTest...
###

#d_test_scaled = scaler.transform(DTest)


print 'StandardScaler is on.'

#grr = pandas.plotting.scatter_matrix(resultDF_scaled, c = result['target'],  figsize=(12,12), marker = 'o', hist_kwds={'bins':20}, s=40, alpha = 0.8, cmap='jet')

#grr = pandas.plotting.scatter_matrix(resultDF, c = result['target'],    figsize=(12,12), marker = 'o', hist_kwds={'bins':20}, s=40, alpha = 0.8, cmap='jet')

#plt.savefig(filename+'_histo_June27.png')
#plt.show()

'''
y_train = np.ravel(y_train)
nNeighbors = 4
print 'nNeighbors =  ',  nNeighbors
knn = KNeighborsClassifier(n_neighbors = nNeighbors)
knn.fit(X_train, y_train)
'''
print " ***********  "
r = open('knn_pickle_reduced_space', 'r')
knn2 = cPickle.load(r)
r.close()
pca = cPickle.load(open('pca_pickle', 'r'))
X_pca = pca.transform(X_train)

#y_pred = knn2.predict(DF)
y_pred  = knn2.predict(X_pca)
print 'shape of y_pred ... ', np.shape(y_pred)
print 'shape of y_pred[0] ... ',  np.shape(y_pred['0'])
print 'y_pred:  '
print y_pred
print ' '
print 'y_train'
print y_train
#y_pred = y_pred.reshape(-1,1)

#print ' now.. after reshaping :  ', np.shape(y_pred)

#print " predicted:  ", y_pred


#print ' shape of d_truth ', np.shape(d_truth),'\n'
print " right before reporting line:  "


print("shape of y_pred:{}".format(np.shape(y_pred)))
print("shape of y_train : {}".format(np.shape(y_train)))

#print("Test set accuracy:{:.3f}".format(knn2.score(y_pred, d_truth)))

print'diagnostic:  shape of y_pred = ', np.shape(y_pred),'   shape of y_test = ', np.shape(y_test) , 'shape of y_train = ', np.shape(y_train)
#pca_pred = y_pred.reshape(-1,1)
#pca_test = y_train.reshape(-1,1)

#print("shape of pca_pred :{}".format(np.shape(pca_pred)))

#print("shape of pca_test :{}".format(np.shape(pca_test)))

#print("Test set mean score:{:.3f}".format(np.mean(y_pred[0] == y_train[0])))
#print 'pca_pred ', np.shape(pca_pred),  '   pca_test  ', np.shape(pca_test)


#print "shape of d_test_scaled ", d_test_scaled.shape()
'''
i = 0
g_truePositive  = 0
g_trueNegative  = 0
g_falseNegative = 0
g_falsePositive  = 0


while i < 5000:
    if d_truth[i] == 1.0 and y_pred[i] == 1.0: 
        g_truePositive += 1
    if d_truth[i] == 0.0 and y_pred[i] == 0.0:
        g_trueNegative += 1
    if d_truth[i] == 0.00 and y_pred[i] == 1.00:
        g_falseNegative += 1
    if d_truth[i] == 1.00 and y_pred[i] == 0.00:
        g_falsePositive += 1

    i += 1




print "got to pt. A "
print 'True Positives = ', g_truePositive,  '   True Negatives = ', g_trueNegative,  '     false negative = ', g_falseNegative,   '      false Positive = ',  g_falsePositive

print 'mean of d_truth = ', np.mean(d_truth) ,'   note that length of d_truth = ', np.shape(d_truth)
s = cPickle.dumps(knn)
sw = open('knn_pickle','w')
sw.write(s)

print 'completed saving pickle file of model kkn '


test_pred = knn.predict(d_test_scaled)
print("test_pred: {}".format(test_pred.shape))
test_pred= test_pred.reshape(-1,1)
print("test_pred: {}".format(test_pred.shape))
print("d_test_scaled: {}".format(d_test_scaled.shape))
print("test_results: {}".format(test_results.shape))
print(" other way to calc test accuracy of test set: {:.3f}".format(np.mean(test_pred == test_results)))
print '\n'
print ("Test accuracy for test_pred:{:.3f}".format(knn.score(test_pred, test_results)))
'''
print "\n   8=8=8=8=8=8=8"

