#import matplotlib as mpl
#import matplotlib.pyplot as plt
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
import time
import os, sys
import pandas
from pandas import DataFrame
from pandas import Series
#import ft
import ft_sp
import f_pad




### This function calculates the weird ~60 sec feature in the data, as feature5

def feat5(data):
    t_start = time.time()
    len = int(np.shape(data)[0])
    data24280 = np.zeros(len)
    j = 0
    while j < len- 24280:
        data24280[j] = data[j+24280] - data[j]
        j += 1
    S = Series(data=data24280)
#    print 'manual mean of data24280 = ', S.mean()
    feature5 = S.mean()
#    S.plot.hist(bins = 30)
    t_end = time.time()
    return feature5






result   = np.zeros(0)
feature1 = np.zeros(0)
feature2 = np.zeros(0)
feature3 = np.zeros(0)
feature4 = np.zeros(0)
feature5 = np.zeros(0)

### use object f to control TODs to be processed
###
### name of tod list file is tod_list
###

tod_list = 'texlist_train_edited.npy'

f = np.load(tod_list)
print "tod list to be processed is  ", tod_list, "\n"


#f = np.load('onetod.npy')
print f

if np.shape(f)[0] > 1: 

    len = int(np.shape(f)[0])
else: len = 1

print   ('Number of TODs to be processed = ',len) 




### get Yulin's list of tes detectors

t = np.load('tes.npy')
print 'shape of t is  ', np.shape(t)
len_t = int(np.shape(t)[0])
print 't: \n', t

print "len_t   = ", len_t
i = 0 

while i < len:
    
    basename = str(f[i])
    print (basename)
    print ( '  ' )

    gds = np.array

    print ('processing tod  ', basename)
    params = {
        "tod_conf": {
        "fix_sign": False,
        "read_data": True},
        "cuts": {
        "depot": '/mnt/act3/users/lmaurin/depot/',
        "tag": 'mr3c_pa3_f090_s16',
        "apply": True},
        "cal": {
        "config": [
            {"type": "depot_cal",
             "depot": '/mnt/act3/users/lmaurin/depot/',
             "tag": 'actpol1_2013_c7_v5_cal1'},
            ],
        "apply": True},
        "detrend": True,
        "removeMean": True,
        "preFilter": {
        "apply": True,
        "deButterworth": True,
        "timeConst": {
            "source": "columns_file",
            "columns": None,
            "filename": "/mnt/act3/users/rdunner/actpol_depots/shared/TimeConstants/2014/time_const_2014_ar1_20141027.txt"
        },
        "detrend": False,
        "retrend": False,},
        }


    fb = moby2.scripting.get_filebase()

    tod = moby2.scripting.get_tod({
        'filename': fb.get_full_path(basename)})

    name = fb.get_full_path(basename)
    print name

 


#
# Get the cuts object for this TOD
#




    cuts = moby2.scripting.get_cuts(
    {'depot': {'path': '/mnt/act3/users/lmaurin/depot/'},
     'tag': 'mr3_pa3_s16'},
    tod=tod)

#    print 'shape of tod.data  '
#    print np.shape(tod.data)

    ndet = int(np.shape(tod.data)[0])
 
#    print ' ndet =  ', ndet

    good_dets = np.array

# The good detectors?
    good_dets = cuts.get_uncut()

#    print 'There are %i good detectors.' % len(good_dets)
#    print ('good dets are ...')
#    print (good_dets)
#    print ( type(good_dets)) 
#    if len(good_dets)>0: print "The good detectors are ...\n", good_dets



    
#    res = np.zeros(ndet)

#    res = np.zeros(0)
    j = 0
#    while j < ndet:
    
    for j in t:
# diagnostic
        print 't value (j):  ',  j
#        if j in good_dets : res[j] = 1
        if j in good_dets: result = np.append(result, 1)
        else:              result = np.append(result, 0)
#        result = np.append(result, res[j])
        j += 1

#    result = np.append(result, res)
## diagnostic:
    print ' shape of result = ', np.shape(result), 'j = (after the good_det loop)  ' , j, '\n' 

## now calculate the three features for this TOD
##

    tod = moby2.scripting.get_tod({'filename':basename})
    N_dets = int(np.shape(tod.data)[0])
#    print 'N_dets = ', N_dets

    Nsamp =  int(np.shape(tod.data)[1])
#    print 'Nsamp = ', Nsamp
    NN = f_pad.nextregular(Nsamp)
#    print " NN = ", NN, "\n"
    p_l_a = np.zeros(len_t)
    p_h_a = np.zeros(len_t)
    rmsx  = np.zeros(len_t)
    f5    = np.zeros(len_t)

    k = 0

#############################################################
############ feature1 is the 'low' fourier feature ##########
############ feature2 is the 'high' fourier feature #########
############ feature3 is the rms feature ####################
############ feature4 is the cv feature  ####################
############ feature5 is the weird 60 sec. feature ##########
#############################################################

    for k in t:
        p_l, p_h = ft_sp.ft(tod.data[k], NN)
        rmx      = np.std(tod.data[k])
        ff5      = feat5(tod.data[k])
        feature1    = np.append(feature1, p_l)
        feature2    = np.append(feature2, p_h)
        feature3    = np.append(feature3, rmx)
        feature5    = np.append(feature5, ff5)

 
        k += 1
    print ' k count =  ', k, '\n'

    
    
    print " finished, TOD ", basename

    i += 1

##  append res onto the growing 1D array result which will contain the 1's and 0's for each detector of each TOD:
print "exited from TOD loop; shape of result = ", np.shape(result)
print "exited from TOD loop;  shape of the four parameters are ",'\t', np.shape(feature1),'\t',np.shape(feature2),'\t', np.shape(feature3), '\t', np.shape(feature5)
 

np.save('result_train_tes_6Feb', result)
np.save('feature1_train_tes_6Feb', feature1)
np.save('feature2_train_tes_6Feb', feature2)
np.save('feature3_train_tes_6Feb', feature3)
#np.save('feature4_train_tes', feature4)
np.save('feature5_train_tes_6Feb', feature5)


print 'got to end of whole code'



print "end"

