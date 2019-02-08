###
###   Reads a TOD features from files and performs analysis 
###


import matplotlib as mpl
import matplotlib.pyplot as plt
import moby2
import pickle as pk
import cPickle as pkc
import math
import numpy as np
from moby2.scripting import products
from moby2.tod.filter import prefilter_tod
from moby2 import libactpol
from moby2 import detectors
import scipy as sc
from scipy import stats
import det_mapper3
import det_mapper4
import time
import datamod
import linregmod
import remslopemod
import sts
import delta1
import d_check
import index1
from moby2.scripting import get_filebase
from moby2.instruments import actpol
from moby2.analysis.tod_ana import visual as v
#import ft
import time
from time import localtime
import pandas
from pandas import DataFrame
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


###
### I don't know whose glitch module this is;  I am not sure it works very well.
###


def glitch(input, length):
    output      = np.zeros(length)
    out         = np.zeros(length)
    dlttt       = np.zeros(length)
    dlt         = delta1.delta1.dlt(input, length, dlttt)

    av          = 0.0
    stdv        = 0.0
    variability = sts.sts.stts( dlttt, length )
    var         = variability[0]

    out      = d_check.d_check.dta(input, dlttt, var, length, output)

    return output

## train results are results_train
## test restuls are  results_test
#results_train = np.load('result_train.npy')
#results_train   = np.load('result_train_tes.npy')
#results = np.load('result_test_31Jan.npy')
#results = np.load('result_test_2Feb.npy')
results  = np.load('result_train_tes_6Feb.npy')

print 'length of results:    ',  np.shape(results)

#  feature1 is the low end fft parameter,  feature2 is high end fft,  feature3 is rms

feat1             = np.load('feature1_train_tes_6Feb.npy')
feat2             = np.load('feature2_train_tes_6Feb.npy')
feat3             = np.load('feature3_train_tes_6Feb.npy')
feat5             = np.load('feature5_train_tes_6Feb.npy')


ff1 = Series(feat1)
ff2 = Series (feat2)
ff3 = Series (feat3)
#ff4 = Series (feat4)
ff5 = Series(feat5)
########################################################
#### section to analyze good vs. bad in features #######
########################################################

feat1_good = np.zeros(0)
feat2_good = np.zeros(0)
feat3_good = np.zeros(0)
feat4_good = np.zeros(0)
feat5_good = np.zeros(0)
feat1_bad  = np.zeros(0)
feat2_bad  = np.zeros(0)
feat3_bad  = np.zeros(0)
feat4_bad  = np.zeros(0)
feat5_bad  = np.zeros(0)

'''
###################################################################
####### analyze good v. bad   #####################################
###################################################################
i = 0
while i < 16112:
    if results_train[i] == 1: 
        feat1_good = np.append(feat1_good, feat1[i])
        feat2_good = np.append(feat2_good, feat2[i])
        feat3_good = np.append(feat3_good, feat3[i])
        feat4_good = np.append(feat4_good, feat4[i])
    else:
        feat1_bad  = np.append(feat1_bad, feat1[i])
        feat2_bad  = np.append(feat2_bad, feat2[i])
        feat3_bad  = np.append(feat3_bad, feat3[i])
        feat4_bad  = np.append(feat4_bad, feat4[i])
    i += 1

f1_g = Series(feat1_good)
f2_g = Series(feat2_good)
f3_g = Series(feat3_good)
f4_g = Series (feat4_good)
f1_bad = Series (feat1_bad)
f2_bad = Series (feat2_bad)
f3_bad = Series (feat3_bad)
f4_bad = Series (feat4_bad)

f1gb = {'Feat1 Good': f1_g, 'Feat1 Bad':f1_bad}
f2gb = {'Feat2 Good': f2_g, 'Feat2 Bad': f2_bad}
f3gb = {'Feat3 Good': f3_g, 'Feat3 Bad': f3_bad}
f4gb = {'feat4 Good': f4_g, 'Feat4 Bad': f4_bad} 
DF_gb_1 = DataFrame(data = f1gb)
DF_gb_2 = DataFrame(data = f2gb) 
DF_gb_3 = DataFrame(data = f3gb)
DF_gb_4 = DataFrame(data = f4gb)

DF_gb_1.hist(bins = 300)
plt.show()

DF_gb_2.hist(bins = 300)
plt.show()

DF_gb_3.hist(bins = 300)
plt.show()

DF_gb_4.hist(bins = 300)
plt.show()
###################################################################
###################################################################

'''

DF_result_test = DataFrame(data = results)

#DF_train = DataFrame(data = [f1, f2, f3,f4, f5, f6, f7, f8, f9, f10, ff1, ff2, ff3]).transpose() 
#DF_target = DataFrame(data=[f1,f2, f3, f4, f5, f6, f7, f8, f9, f10]).transpose()
#DF_test    = DataFrame(data=[ff1,ff2, ff3]).transpose() 
#DF_train    = DataFrame(data=[ff1, ff2, ff3, ff4]).transpose()

DF_test    = DataFrame(data=[ff1, ff2, ff3, ff5]).transpose()

#DF_test     = DataFrame(data=[ff5]).transpose()

#df.fillna(0, inplace=True)
# replace NaN values with 0
DF_test.fillna(0., inplace=True)

'''
##############################################################################
######## play with histograms of the features  ###############################
##############################################################################

DF_train.hist(bins = 200)
plt.title("Fourier 1 and 2 ")
plt.show()

DF2 = DataFrame(data= [ff3, ff4]).transpose()
DF2.hist(bins = 200)
plt.title('RMS and CV')
plt.show()
'''


#print 'DF_result ... '
#print DF_result

#print DF_target


## diagnostic:
#print 'shape of result[data] ', np.shape(result['data']),  'shape of result[target] ', np.shape(result['target']), '\n'

print "   ======   "
#    print("keys of result: \n{}".format(result.keys()))

####  use this, which splits the data into train and test:
X_train, X_test, y_train, y_test =  train_test_split(DF_test, results, random_state=333)
   
scaler = StandardScaler()
print ' shape of DF_train = ', np.shape(DF_test)


scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
print ' shape of X_train_scaled =  ', np.shape(X_train_scaled)
### need to put test info back in...
X_test_scaled  = scaler.transform(X_test)
print ' shape of DF_result_test  ', np.shape(DF_result_test), '\n'
#####
#####  should these be scaled ?  or should I be looking for the other 'x' variables?
#####
#y_scaled       = scaler.transform(DF_result_train)
#y_scaledDF     = DataFrame(y_scaled)


#####
#####   should I be scaling DF_result_train or not !?  
#####
#result_scaled  = scaler.transform(DF_result_train)
#resultDF_scaled = DataFrame(result_scaled)

print 'StandardScaler is on.'
#print 'StandardScaler is OFF '

print("X-test shape: {}".format(X_test.shape))

print("x-train shape: {}".format(X_train.shape))
print("y_test shape: {}".format(y_test.shape))
print("y-train shape: {}".format(y_train.shape))


nNeighbors = 3 
print 'nNeighbors =  ',  nNeighbors
knn = KNeighborsClassifier(n_neighbors = nNeighbors)
knn.fit(X_train_scaled, y_train)

#####   what are the variables I need for this ? 

#kkn.fit(X_train_scaled, resultDF_scaled)

print ' '
print ' '

y_pred = knn.predict(X_test_scaled)
print '\n\n     RESULTS OF edited TRAIN  TOD  DATA SET         \n\n'
#print " predicted: \n", y_pred,   "       length of y_pred  ", np.shape(y_pred)[0]
#print("Test set mean score:{:.3f}".format(np.mean(y_pred == y_test)))
#print("Test set accuracy:{:.3f}".format(knn.score(X_test, y_test)))

#print "\n  y_test  \n", y_test



y_test_array = y_test
print 'shape of y_test = ', np.shape(y_test), '    shape of y_pred = ', np.shape(y_pred), '\n'

i = 0
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

N_tests = np.shape(y_pred)[0]
while i < N_tests:
   if y_pred[i] == 1. and y_test_array[i] == 1. : true_positives += 1
   if y_pred[i] == 1. and y_test_array[i] == 0.0:
       false_positives += 1
   if y_pred[i] == 0. and y_test_array[i] == 1. :
       false_negatives += 1
#       print 'false_neg  i = ', i
   if y_pred[i] == 0. and y_test_array[i] == 0. :
       true_negatives += 1
#       print 'true_neg  i = ', i 

   i += 1
print '\n\n\n'
print "True-positives  \t", true_positives 
print "True-negatives  \t", true_negatives
print "False-positives \t", false_positives
print "False-negatives \t", false_negatives
print '\n\n'

print "total samples  \t", (true_positives+true_negatives+false_positives+false_negatives), '\n\n'

print("True-positives:  {:.1f} %".format(true_positives*100./N_tests))
print("True-negatives:  {:.1f} %".format(true_negatives*100./N_tests))
print("False-positives: {:.1f} %".format(false_positives*100./N_tests))
print("False-negatives: {:.1f} %".format(false_negatives*100./N_tests))

print ' '
print("Accuracy:        {:.1f} %".format(true_positives*100./N_tests + true_negatives*100./N_tests))

prec = 100.* true_positives/(true_positives + false_positives)
sens = 100.* true_positives/(true_positives + false_negatives)
print ("Precision:       {:.1f} %".format(prec))
print ("Sensitivity:     {:.1f} %".format(sens))

print ' '


print 'end   end   end'


