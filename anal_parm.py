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
dLong = np.zeros(9410016)
d = np.zeros(1000000)
d_test = np.zeros(3000000)
data = cPickle.load(open('mr3_pa2_s14_results.pickle','r'))

keys = data.keys()
print "keys are ... \n", keys,  '\n'



culledKeys = ['sel',  'corrLive', 'rmsLive', 'kurtLive', 'skewLive', 'normLive',  'jumpDark', 'gainLive' , 'DELive', 'cal', 'jumpLive']
print 'culledKeys   ', culledKeys,  '\n'
DF      = DataFrame()
DFa     = DataFrame()
DTesta = DataFrame()
DTest  = DataFrame()
dataKeys = culledKeys
resultKeys=  ['respSel', 'corrLive', 'rmsLive', 'kurtLive', 'normDark', 'skewLive', 'normLive', 'darkRatioLive', 'MFELive', 'jumpDark', 'gainDark', 'gainLive' , 'DELive', 'cal', 'rmsDark', 'jumpLive']   
for k in culledKeys: 
#        print ' working on ', k
        dLong = data[k].reshape(9410016)
        d = dLong[0:999999]
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

#df.fillna(df.mean())
DF = DF.fillna(DF.mean())

dLong = data['sel'].reshape(9410016)
d = dLong[0:999999]
test_results = dLong[2990000:2999999].reshape(-1,1)
DF_target = DataFrame(d)


print "=====    =====  ===== "


print 'here is result[k]  '
for k in culledKeys:
    print 'k is ', k, '  ',  np.shape(result[k])


X_train, X_test, y_train, y_test =  train_test_split(DF, DF_target, random_state=386)

#X_train, X_test, y_train, y_test =  train_test_split(result, tar, random_state=524)


print 'got to here'

print("X-test shape: {}".format(X_test.shape))

print("y_test shape: {}".format(y_test.shape))
print("X-train shape: {}".format(X_train.shape))
print("y-train shape: {}".format(y_train.shape))
print("test_results shape: {}".format(test_results.shape))

scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)


DTest = DTest.fillna(value=0.0)
### turn off scaling for DTest...
###

#d_test_scaled = scaler.transform(DTest)


print 'StandardScaler is on.'

#grr = pandas.plotting.scatter_matrix(resultDF_scaled, c = result['target'],  figsize=(12,12), marker = 'o', hist_kwds={'bins':20}, s=40, alpha = 0.8, cmap='jet')

#grr = pandas.plotting.scatter_matrix(resultDF, c = result['target'],    figsize=(12,12), marker = 'o', hist_kwds={'bins':20}, s=40, alpha = 0.8, cmap='jet')

#plt.savefig(filename+'_histo_June27.png')
#plt.show()

####
#### experimenting with PCA routine
####

fig, axes = plt.subplots(10, 2, figsize = (10,20))
#bad       =  d_test[d_test['sel'] == 0]
#good      =  d_test[d_test['sel'] == 1]

#ax = axes.ravel()
bins = 100
#for i in range(10):

#    _, bins = np.histogram(d_test[:, i], bins = 20)
#    ax[i].hist(bad[:, i] , bins = bins, color = 'red', alpha = .5)
#    ax[i].hist(good[:,i], bins = bins, color = 'green' ,  alpha = .5)
pandas.hist( DF['corrLiv'], bins = bins , color = 'green', alpha = .5)    
#fig.tight_layout()
plt.show()
  
y_train = np.ravel(y_train)
#nNeighbors = 4
#print 'nNeighbors =  ',  nNeighbors
#knn = KNeighborsClassifier(n_neighbors = nNeighbors)
#knn.fit(X_train, y_train)
print 'model is Decision Tree Classifier, max depth = 10'
cls = DecisionTreeClassifier(max_depth=10)
#cls = GaussianNB


cls.fit(X_train_scaled, y_train)




print " ***********  "
y_pred  = cls.predict(X_test)
#y_pred = knn.predict(X_test)
#print 'TOD name:  ', filename
print " predicted:  ", y_pred
#print("Test set mean score:{:.3f}".format(np.mean(y_pred == y_test)))
print("Test set accuracy:{:.3f}".format(cls.score(X_test, y_test)))




s = cPickle.dumps(cls)
sw = open('decisionTree_pickle','w')
sw.write(s)

print 'completed saving pickle file of Decision Tree model  '


'''
test_pred = knn.predict(d_test_scaled)
print("test_pred: {}".format(test_pred.shape))
test_pred= test_pred.reshape(-1,1)
print("test_pred: {}".format(test_pred.shape))
print("d_test_scaled: {}".format(d_test_scaled.shape))
print("test_results: {}".format(test_results.shape))
print(" other way to calc test accuracy of test set: {:.3f}".format(np.mean(test_pred == test_results)))
print '\n'
print ("Test accuracy for test_pred:{:.3f}".format(knn.score(test_pred, test_results)))

print "\n   8=8=8=8=8=8=8"

'''
