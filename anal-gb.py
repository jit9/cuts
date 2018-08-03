import matplotlib.pyplot as plt
import numpy as np
import cPickle
import os, sys
import math
import scipy
import pandas
from pandas import DataFrame
import mglearn
from mglearn import discrete_scatter
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
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import model_selection



gk = np.zeros(0)
bk = np.zeros(0)
gk2 = np.zeros(0)
bk2 = np.zeros(0)
result = {}
d_test_results = {}
gb_array = np.zeros(0)
#dLong = np.zeros(9410016)
#d = np.zeros(1000000)
#d_test = np.zeros(3000000)
data = cPickle.load(open('mr3_pa2_s14_results.pickle','r'))

keys = data.keys()
#print "keys are ... \n", keys,  '\n'



culledKeys = ['sel', 'respSel', 'corrLive', 'rmsLive', 'kurtLive', 'normDark', 'skewLive', 'normLive', 'darkRatioLive', 'MEFLive', 'jumpDark', 'gainDark', 'gainLive' , 'DELive', 'cal', 'rmsDark', 'jumpLive']
#print 'culledKeys   ', culledKeys,  '\n'
DF      = DataFrame()
DFa     = DataFrame()
DTesta = DataFrame()
DTest  = DataFrame()
Dpartial = DataFrame()
Dwhole   = DataFrame()
Dgood    = DataFrame()
Dbad     = DataFrame()


keys_list = ['sel','corrLive',  'rmsLive', 'kurtLive', 'skewLive','normLive','darkRatioLive', 'MFELive', 'DELive', 'jumpLive',  'gainLive']
print 'keys_list is ',  keys_list ,'\n'

for k in keys_list:
    dLong = data[k].reshape(9410016)
    DFa = DataFrame({k: dLong})
    DF = pandas.concat([DF, DFa], axis = 1)


#print 'DF columns are ...',  DF.columns
print ' '

#print 'shape of data[sel] is'
#print np.shape(data['sel'])
print ' '
d_sel  =  data['sel'].reshape(9410016)
#print ' shape of d_sel  ',   np.shape(d_sel)
#print 'type --- data :  ',  type(data)

print ' '

for k in keys_list:
    dD = DF[k]
    print '    Whole data set '
    print k,  '\n', dD.describe()

print '\n\n\n'


Dgood  =  DF[DF['sel'] == 1.0]
Dbad   =  DF[DF['sel'] == 0.0 ]
print 'shape of Dgood  ', np.shape(Dgood.shape), '    shape of Dbad ', np.shape(Dbad)
#print Dgood.describe()
print ' '
#print Dbad.describe()
print ' '

Dx = DataFrame()
for k in keys_list:
    kgood = str(k+' good')
    kbad  = str(k+' bad')
    gD = Dgood[k]
    bD = Dbad[k]
    print k, '\nGood: ', gD.describe(), '\n Bad:', bD.describe(), '\n\n' 
    gD = np.array(Dgood[k]).reshape(-1,)
    bD = np.array(Dbad[k]).reshape(-1)
    Dx = DataFrame([gD,bD])
    DD = Dx.transpose()
    Dx.columns = [kgood, kbad]
    Dx.plot.hist()
    plt.show()

    
    

#print Dgood
print ' '
#print Dbad



Dgood.hist(grid=True, bins = 1000)
plt.title(" Good TODs" )
#plt.savefig('good_tods_july26.png')
#plt.show()

Dbad.hist(grid = True, bins = 1000)
plt.title(" Bad TODs  ")
#plt.savefig('bad_tods_july26.png')
#plt.show()


print 'ok ok  '
 
'''
dataKeys = culledKeys
resultKeys=  ['respSel', 'corrLive', 'rmsLive', 'kurtLive', 'normDark', 'skewLive', 'normLive', 'darkRatioLive', 'MFELive', 'jumpDark', 'gainDark', 'gainLive' , 'DELive', 'cal', 'rmsDark', 'jumpLive']   

#print ' result keys are ...'
#print resultKeys




for k in resultKeys:
#        print ' working on ', k
        dLong = data[k].reshape(9410016)
        d = dLong[0:999999]
#        d_test = dLong[2990000:2999999]
        result[k] = d 
#        d_test_results[k] = d_test 
        DFa = DataFrame({k: d})
        DF = pandas.concat([DF,DFa], axis =1)
        print 'shape of result[', k, '] is  ', np.shape(result[k])

#        DTesta = DataFrame({k:d_test})
#        DTest  = pandas.concat([DTest, DTesta], axis = 1)
#        print 'shape of d_test_results[',k,'] is  ', np.shape(d_test_results[k])
        Dpartial = DataFrame(data[k])
        Dwhole   = pandas.concat([Dwhole,Dpartial], axis = 1)
 
print "\n"


#df.fillna(df.mean())
DF = DF.fillna(DF.mean())

dLong = data['sel'].reshape(9410016)
d = dLong[0:999999]
#test_results = dLong[2990000:2999999].reshape(-1,1)
DF_target = DataFrame(d)


print "=====    =====  ===== "




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


y_train = np.ravel(y_train)
nNeighbors = 4
print 'nNeighbors =  ',  nNeighbors
knn = KNeighborsClassifier(n_neighbors = nNeighbors)
knn.fit(X_train, y_train)
#print 'model is Decision Tree Classifier, max depth = 10'
#cls = DecisionTreeClassifier(max_depth=10)
#cls = GaussianNB


#cls.fit(X_train_scaled, y_train)




print " ***********  "
#y_pred  = cls.predict(X_test)
y_pred = knn.predict(X_test)
#print 'TOD name:  ', filename
print " predicted:  ", y_pred
#print("Test set mean score:{:.3f}".format(np.mean(y_pred == y_test)))
#print("Test set accuracy:{:.3f}".format(knn.score(X_test, y_test)))

print "starting the PCA routine .... "

#scaler.fit(DTest)
pca = PCA(n_components = 2)
#pca.fit(DTest)
#X_pca = pca.transform(DTest)

pca.fit(X_train_scaled)
X_pca  = pca.transform(X_train_scaled)
#print ("Original shape: {}".format(str(DTest.shape)))
#print ("Reduced shape: {}".format(str(X_pca.shape)))

print ("Original shape: {}".format(str(X_train_scaled)))
print ("Reduced shape: {}".format(str(X_pca.shape)))

s = cPickle.dumps(pca)
sw = open('pca_pickle', 'w')
sw.write(s)
sw.close()


#plt.figure(figsize=(8,8))
#mglearn.discrete_scatter(X_pca[:,0], X_pca[:,1], y_train)
#plt.title("Principal Component Analysis")
#plt.savefig("PCA_July24_2_components")

#plt.show()

DS = DataFrame({'good': X_pca[:,0], 'bad': X_pca[:,1]})
#DS.plot.scatter(DS['good'], DS['bad'])
#plt.scatter(X_pca[:,0], X_pca[:,1], y_train)
#plt.show()
#print DS

#from pandas.plotting import scatter_matrix

## df = pd.DataFrame(np.random.randn(1000, 4), columns=['a', 'b', 'c', 'd'])







DS.hist(grid=True, bins = 1000)
plt.title(" Histogram of Good v. Bad ")
plt.show()

#s = cPickle.dumps(cls)
#sw = open('decisionTree_pickle','w')
#sw.write(s)

#print 'completed saving pickle file of Decision Tree model  '

###now to PCA anaysis...
###

X_train_pca = pca.transform(X_train)
X_test_pca  = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train_pca, y_train)
print ("Test Accuracy of pca model : {:.2f}".format(knn.score(X_test_pca, y_test)))

s = cPickle.dumps(knn)
sw = open('knn_pickle_reduced_space', 'w')
sw.write(s)

sw.close()

print "done  done   done"




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

print 'end end end end'


