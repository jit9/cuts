"""
    Builds keras model from pickle parameters
    This code build the model using only ff_sel == 1 and tes_sel == 1 (the working detectors), called 
      "trimmed"
    This code can combine two or three TODs into a data package on which to build the model.
    Incorporates sklean's StandardScaler for the data.
    Has a mechanism to log and store the 'good' and 'bad' det's by TOD for futher analysis.
"""

import numpy as np
import sys
# sys.path.append(r'/projects/ACT/feynman-users/treu/sklearn')
import sklearn
import sklearn.semi_supervised
import sklearn.neural_network
import sklearn.decomposition
from sklearn.decomposition import PCA 
from sklearn import preprocessing as prep
import tensorflow
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras import models
from keras import metrics
from keras import losses
import pandas
from pandas import DataFrame, Series
from pandas import plotting
import h5py
import pickle as pk
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg", force = True)


def pickle_in():
    """ 
    pickle_in function reads a pickle file from Yilun and returns a dictionary d containing all of the 
    so-called pickle-parameters.
    """
# location of the pickle file of interest is done in two steps, the path_start parameter
# gives the pathway to the file;  
# the pickle_file_name is the name of the pickle file itself.

#    path_start = "/projects/ACT/yilung/depot/Postprocess/pickle_cuts"
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pa6_f090_s17_c11_v4"
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pa4_f220_s17_c11_v4"    
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pa3_f150_s16_c11_v0"
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pa6_f090_s18_c11_v1"
#    pa5_f150_s19_c11_v0
    path_start = "/projects/ACT/yilung/depot/Postprocess/pa5_f090_s19_c11_v0"

#    path_start = "/scratch/gpfs/yilung/depot/Postprocess/pa7_f030_s20_bri_v0" 
#    path_start = "/scratch/gpfs/yilung/depot/Postprocess/pa7_f040_s20_bri_v0"
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pickle_cuts"
    pickle_file_name = "pa5_f090_s19_c11_v0_results.pickle"
#    pickle_file_name = "pa6_f090_s18_c11_v1_results.pickle"    
#    pickle_file_name = "pa7_f030_s20_bri_v0_results.pickle"
#    pickle_file_name = "pa7_f040_s20_bri_v0_results.pickle"
#    pickle_file_name = "pa5_f090_s19_c11_v0_results.pickle"
#
    f = open(path_start+"/"+pickle_file_name, "rb")
    print("Working on file:  \n{}\n".format(f))
    d        = pk.load(f)
    print("Diagnostic from pickle_in fct.   shape of d['tes_sel']= {}".format(np.shape(d['tes_sel'])))
    f.close()
    tod_names = d['name']
    return d, pickle_file_name, tod_names

def transfer(d, tod_num, tod_names):
    """ transfer function receives the dictionary d and creates a DataFrame E
        containig the features to be used in the neural net algorithm.
  
        This function works on one TOD at a time:  The output DataFrame E contains
        the data from one TOD, namely tod_num which was transferred in the function
        call.   

        Also in the function call are the string variables tod_names for 
        all the TOD names found in the pickle file d.

        This script also contains functions called createSeries, appendSeries, and
        createDataFrame.   These do the same things as this function transfer but
        are used to combine two or three TODs into one DataFrame for use in
        building the neural network model.      

        Before creating DataFrame E,  first a DataFrame called D is created, which
        contains each detector in the TOD and for each detector (the rows of the
        DataFrame) there are the features (the columns).  The DataFrame E is
        built to contains only those detectors which satisfy ff_sel == 1 and
        tes_sel == 1.   

        The function returns the DataFrame E plus:
            sel which is a numpy array of the sel values from Yilun's pickle file
                for this TOD and each detector

            det_num which is a numpy array containing the detector number for each
                respective row of the DataFrame E.  This is needed because the
                DataFrame D simply uses the row number (that is D.iloc ) as the 
                detector number.  When DataFrame E is created we lose this bookkeeping
                convenience and need a separate list of the detector numbers.

            tod_name which is a sting variable containing the name of the TOD.

            adj_length which is the number of detectors (rows) in DataFrame E.
 
    """ 
    print("number of TOD available = {}".format(len(d['name'])))
    tod_name = d['name'][tod_num]
    print("entering transfer function,  tod name = {}".format(tod_name))
    tes_sel = d['tes_sel'].transpose()        
    ff_sel = d['ff_sel'].transpose()
    rms = d['rmsLive'].transpose()
    rms = rms[tod_num]
    S_rms = Series(rms)
    sel = d['sel'].transpose()
    sel = sel[tod_num]
    length = len(sel)
    det_num = np.arange(length) 
    S_det_num = Series(det_num)
    isel = np.zeros(length) 
    i = 0
    while i < length:
        isel[i] = int(sel[i]) 
        i += 1
    S_sel     = Series(isel)
    kurt     = d['kurtLive'].transpose()
    kurt     = kurt[tod_num]
    S_kurt   = Series(kurt)
    S_ff_sel = Series(ff_sel)
    S_tes_sel = Series(tes_sel)
    gainLive = d['gainLive'].transpose()[tod_num]
    S_gainLive = Series(gainLive)
    corrLive = d['corrLive'].transpose()[tod_num]
    S_corrLive = Series(corrLive)
    normLive = d['normLive'].transpose()[tod_num]
    S_normLive = Series(normLive)
    skewLive   = d['skewLive'].transpose()[tod_num]
    S_skewLive = Series(skewLive)
    MFELive    = d['MFELive'].transpose()[tod_num]
    S_MFELive  = Series(MFELive)
    DELive     = d['DELive'].transpose()[tod_num]
    S_DELive   = Series(DELive)
    jumpLive   = d['jumpLive'].transpose()[tod_num]
    S_jumpLive = Series(jumpLive)
    D = DataFrame(data=[S_ff_sel, S_tes_sel, S_sel, S_det_num, S_rms, S_kurt, S_gainLive, S_corrLive, S_normLive, skewLive,  S_MFELive, S_DELive, S_jumpLive]).transpose()
    
       
# E is a DataFrame with only those rows for which S_ff_sel and S_tes_sel == 1

    E          = D[D[0] == 1]

    E          = E[E[1] == 1]
    sel        = np.array(E[2])
    det_num = E.transpose().to_numpy()[3]
    E          = E.transpose()[4:].transpose()
    adj_length = int(np.shape(sel)[0])
# Save DataFrame E for external analysis; or comment out if not in use.
#    flg = False 
#    if (not flg):
#        epick = open("E_stored.pk","wb")
#        dpick = open("D_stored.pk", "wb")
#        pk.dump(D, dpick)
#        pk.dump(E, epick)
#        dpick.close()
#        epick.close()
#        np.save("sel_saved.npy", sel)
#        flg = True
#    else:
#        flg = True 

    return E, sel,det_num, tod_name, adj_length

def buildModel(D, sel, length):
    """    
       buildModel function builds the neural network model.
    """
    model = Sequential()
    model.add(Dense(12, input_dim=9, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

# for experimental purposes,  various types of models are tried and 
#    simply stored as comments for future reference.

#    print("from buildModel,  shape of D:  {}\n".format(D.shape))
# compile the keras model
#    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['FalseNegatives'])
# fit the keras model on the dataset
#    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[metrics.SpecificityAtSensitivity(0.2)])
#    model.compile(loss=[losses.Poisson()], optimizer='adam', metrics=[metrics.PrecisionAtRecall(0.25)])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.Accuracy()])
#    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=[metrics.SparseCategoricalAccuracy()])
#    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=[metrics.Recall()])
    model.fit(D, sel, epochs=50, batch_size=200)
#    model.fit(scaled_D, sel, epochs=40, batch_size=50)
    _, Accuracy = model.evaluate(D, sel)
    predictions  = model.predict(D)
    print("\nRecall of training set = {}\n".format(Accuracy))
#    print("accuracy of trained model = {}".format(accuracy))

#    Save the experimental model for future use or comment out.
#    model.save("pa5_f150_s19_tod655_trimmed_30Dec2020")
#    model.save("pa7_f030_bri_temp","wb")
#    model.save("pa5_f090_s19_c11_v0_trim_8Feb_tod#0")
#    model.save("pa7_f030_s20_tod49_trimmed_30Dec2020")
#    model.save("pa7_f030_bri_tod_0_8Feb")
#    model.save("pa6_f090_s18_c11_v1_tod_0_8Feb") 
#    model.save("pa7_f040_s20_bri_v0_twoTODs_4and9")
    print("from buildModel:  type(D) = {}".format(type(D)))
    DD = DataFrame(data=D)  
    tc, fc, tu, fu, E_cut, s_cut = anal(DD, predictions, sel)
    print("from inside model build, stage 1, tc, fc, tu, fu = ")
    print("                                  {}  {}  {}  {}".format(tc, fc, tu, fu)) 
    return model 


# Function to create initial Series for each parameter, including Sel,  ff_Sel, and tes_Sel
# Return a dictionary of all the Series

def createSeries(d, tod_num, tod_names):
    """
    this function is for use in creating the data package consisting of two or even three
    TODs for use in building the neural network model.

    createSeries function reads in the pickle file's dictionary d and the tod_num (tod number)
    of interest and the tod_names.

    This is a step towards the creation of a DataFrame with the parameters ('features') to be used
    in the neural network algorithm.  

    The output of this fucntion is a dictionary sd containing the features to be used in the
    neural net algorthm, and this dictionary will be passed to a function which adds
    another TOD to be put into the DataFrame that will be analyzed in the neural net algorthim.
    """
    
    print("number of TOD available = {}".format(len(d['name'])))
# put the name of the TOD into the DataFrame
    name = d['name'][tod_num]
    print("DIAGNOSTIC:   name = \n{}".format(name))
    tes_sel = d['tes_sel'].transpose()
    ff_sel = d['ff_sel'].transpose()
    rms = d['rmsLive'].transpose()
    rms = rms[tod_num]
    S_rms = Series(rms)
    sel = d['sel'].transpose()
    sel = sel[tod_num]
#    i = 0
    
    length = int(len(sel))
    isel = np.zeros(length)
    det_num = np.arange(length) 
    for i in range(0,length): 
#     while i < length:
        isel[i] = int(sel[i])
#        i += 1
    S_sel     = Series(isel)
    S_det_num = Series(det_num)
    kurt     = d['kurtLive'].transpose()
    kurt     = kurt[tod_num]
    S_kurt   = Series(kurt)
    S_ff_sel = Series(ff_sel)
    S_tes_sel = Series(tes_sel)
    gainLive = d['gainLive'].transpose()[tod_num]
    S_gainLive = Series(gainLive)
    corrLive = d['corrLive'].transpose()[tod_num]
    S_corrLive = Series(corrLive)
    normLive = d['normLive'].transpose()[tod_num]
    S_normLive = Series(normLive)
    skewLive   = d['skewLive'].transpose()[tod_num]
    S_skewLive = Series(skewLive)
    MFELive    = d['MFELive'].transpose()[tod_num]
    S_MFELive  = Series(MFELive)
    DELive     = d['DELive'].transpose()[tod_num]
    S_DELive   = Series(DELive)
    jumpLive   = d['jumpLive'].transpose()[tod_num]
    S_jumpLive = Series(jumpLive)
    sd = {}
    sd["ff_sel"] = S_ff_sel
    sd["tes_sel"] = S_tes_sel 
    sd["sel"] = S_sel
    sd["rms"] = S_rms
    sd["kurt"] = S_kurt
    sd["gainLive"] = S_gainLive
    sd["corrLive"] = S_corrLive
    sd["normLive"] = S_normLive
    sd["skewLive"] = S_skewLive
    sd["MFELive"] = S_MFELive
    sd["DELive"] = S_DELive
    sd["jumpLive"] = S_jumpLive
    sd["det_num"] = S_det_num
    return sd

# Function to append additional TODs to the Series,  returning a dictionary of all the expanded Series

def appendSeries(d, oldDict,  tod_num, tod_names):
    """
        For use in creating the multi-TOD data package with which to build the
        neural network model.
    """



    print("In appendSeries function\nNumber of TOD available = {}".format(len(d['name'])))
# Put TOD name into the DataFrame
# The old Series is oldDict
# Here are the new param values, from d, using the new tod_num
#    print("Diagnostic,  at beginning of appendSeries,  oldDict['det_num'] is\n{}".format(oldDict["det_num"]))
    name = d['name'].transpose()
    tes_sel = d['tes_sel'].transpose()
    ff_sel = d['ff_sel'].transpose()
    rms = d['rmsLive'].transpose()
    rms = rms[tod_num]
    S_rms = Series(rms)
    S_rms = oldDict["rms"].append(S_rms, ignore_index=True)
    sel = d['sel'].transpose()
    sel = sel[tod_num]
    length = int(np.shape(sel)[0])
    det_num = np.arange(length)
    i = 0
#    print("length = {}".format(length))
    S_det_num = Series(det_num)
#     print("Diagnostic, mid way through appendSeries,  S_det_num is\n{}".format(S_det_num))
    S_det_num = oldDict["det_num"].append(S_det_num, ignore_index=True)
    isel = np.zeros(length)
    while i < length:
        isel[i] = int(sel[i])
        i += 1
    S_sel     = Series(isel)
   
    S_sel    = oldDict["sel"].append(S_sel, ignore_index=True)

    kurt     = d['kurtLive'].transpose()
    kurt     = kurt[tod_num]
    S_kurt   = Series(kurt)
    S_kurt   = oldDict["kurt"].append(S_kurt, ignore_index=True)
    S_ff_sel = Series(ff_sel)
    S_ff_sel = oldDict["ff_sel"].append(S_ff_sel, ignore_index=True)
    S_tes_sel = Series(tes_sel)
    S_tes_sel = oldDict["tes_sel"].append(S_tes_sel, ignore_index=True)
    gainLive = d['gainLive'].transpose()[tod_num]
    S_gainLive = Series(gainLive)
    S_gainLive = oldDict["gainLive"].append(S_gainLive, ignore_index=True)
    corrLive = d['corrLive'].transpose()[tod_num]
    S_corrLive = Series(corrLive)
    S_corrLive = oldDict["corrLive"].append(S_corrLive, ignore_index=True)
    normLive = d['normLive'].transpose()[tod_num]
    S_normLive = Series(normLive)
    S_normLive = oldDict["normLive"].append(S_normLive, ignore_index=True)
    skewLive   = d['skewLive'].transpose()[tod_num]
    S_skewLive = Series(skewLive)
    S_skewLive = oldDict["skewLive"].append(S_skewLive, ignore_index=True)
    MFELive    = d['MFELive'].transpose()[tod_num]
    S_MFELive  = Series(MFELive)
    S_MFELive  = oldDict["MFELive"].append(S_MFELive, ignore_index=True)
    DELive     = d['DELive'].transpose()[tod_num]
    S_DELive   = Series(DELive)
    S_DELive   = oldDict["DELive"].append(S_DELive, ignore_index=True)
    jumpLive   = d['jumpLive'].transpose()[tod_num]
    S_jumpLive = Series(jumpLive)
    S_jumpLive = oldDict["jumpLive"].append(S_jumpLive, ignore_index=True)
    sd = {}
    sd["ff_sel"] = S_ff_sel
    sd["tes_sel"] = S_tes_sel
    sd["sel"] = S_sel
    sd["det_num"] = S_det_num
    sd["rms"] = S_rms
    sd["kurt"] = S_kurt
    sd["gainLive"] = S_gainLive
    sd["corrLive"] = S_corrLive
    sd["normLive"] = S_normLive
    sd["skewLive"] = S_skewLive
    sd["MFELive"] = S_MFELive
    sd["DELive"] = S_DELive
    sd["jumpLive"] = S_jumpLive
#    print("from appendSeries,  keys of sd are:  \n{}".format(sd.keys()))
#    print("from appendSeries, sel = \n\n{}".format(sd["sel"]))
#    print("from appendSeries, sum of sel = {}".format(np.sum(sd["sel"])))
#    for i in range(length):
#        print("At end of createDataFrame, \ndet_num  {}".format(sd["det_num"][i]))

    return sd

# Function to create the DataFrame holding all the parameters, including sel, ff_sel and tes_sel for the 
# expanded set of tods on which the model will be built.

def createDataFrame(sd):
    """
        This function is part of the pipeline which builds the DataFrame with two or three TODs
        for use in building the neural net model.
    """
    D = DataFrame(data=[sd["ff_sel"], sd["tes_sel"], sd["sel"],sd["det_num"], sd["rms"], sd["kurt"],  sd["gainLive"], sd["corrLive"], sd["normLive"], sd["skewLive"], sd["MFELive"], sd["DELive"], sd["jumpLive"]]).transpose()

# E is a DataFrame with only those rows for which S_ff_sel and S_tes_sel == 1

    E          = D[D[0] == 1]
    E          = E[E[1] == 1]
    sel        = np.array(E[2])
    adj_length = int(np.shape(sel)[0])
# Save DataFrame E for external analysis; or comment out as desired.
#    flg = False 
#    if (not flg):
#        epick = open("E_stored.pk","wb")
#        dpick = open("D_stored.pk", "wb")
#        pk.dump(D, dpick)
#        pk.dump(E, epick)
#        dpick.close()
#        epick.close()
#        np.save("sel_saved.npy", sel)
#        flg = True
#    else:
#        flg = True 
    det_num = E.transpose().to_numpy()[3]
    E          = E.transpose()[4:].transpose()
    return E, sel, det_num, adj_length

def anal(E, predictions, s):
    """
        Function anal performs the book keeping function also performed 
        in the main body of the script.   This function was put in
        as part of the development of the multi-TOD model building.
    """

    length = len(s)
    print("from anal,  length of s = {}".format(length))
    tc = 0
    fc = 0
    tu = 0
    fu = 0
    print("From anal functin... type of E :  {}".format(type(E)))
    E_cut = DataFrame()
    s_cut = np.zeros(0)
    pred = np.array(predictions[:])
    grand_tot = 0
    i = 0
    while i < length:
        grand_tot += 1
        x4 = det_number[i]
#   Select the cut off value for the prediction parameter defining
#   good detector or bad detector:   
#        if pred[i] < 0.01: 
        if pred[i] < 0.1:
            xxx = 0
        else:
            xxx = 1
        if xxx == 0 and s[i] == 0:
            E_cut = E_cut.append(E.iloc[i])
            s_cut = np.append(s_cut, s[i])
            tc += 1
        if xxx == 0 and s[i] == 1:
            E_cut = E_cut.append(E.iloc[i])
            s_cut = np.append(s_cut, s[i])
            fc += 1
        if xxx == 1 and s[i] == 0:
            fu += 1
        if xxx == 1 and s[i] == 1:
            tu += 1

        i += 1


    return tc, fc, tu, fu, E_cut, s_cut



# START OF MAIN PROGRAM
#
# load pickle file

d, pickle_file_name, tod_names = pickle_in()
t_cut      = 0
t_uncut    = 0
f_cut      = 0
f_uncut    = 0
t2_cut     = 0
t2_uncut   = 0
f2_cut     = 0
f2_uncut   = 0

# print("length of d['name'] is  {}".format(np.shape(d['name'])))
aaa = int(np.shape(d['name'])[0])
i = 0


# np.save( 'tod_names_pa7_f030.npy', d['name'])

# Prepare to build a new model

tod_num = 0 
pwv_1   = d['pwv'][tod_num]
modelTOD = d['name'][tod_num]
# E, sel, adj_length = transfer(d, tod_num, tod_names)

# print("Build model on two TODs: ")
# print("Build model on tod # {}\t{}".format(tod_num, tod_names[tod_num])) 
seriesDict = createSeries(d, tod_num, tod_names)
# print("seriesDict = {}".format(seriesDict))
# add a second tod to the data on which to build the model
tod_num2 = 1 
pwv_2 = d['pwv'][tod_num2]
tod_num3 = 2 
pwv_3 = d['pwv'][tod_num3]
seriesDict = appendSeries(d, seriesDict, tod_num2, tod_names)

seriesDict = appendSeries(d, seriesDict, tod_num3, tod_names)

print("Build model on tod # {}, also:\t{}".format(tod_num2, tod_names[tod_num2])) 

# print("Build model on tod # {}, also:\t{}".format(tod_num3, tod_names[tod_num3])) 
# print("Diagnostic:  seriesDict =  {}".format(seriesDict))
E, sel, det_number, length = createDataFrame(seriesDict)

# print("Back in main program, here is E\n\n{}".format(E))
# print("Back in main program, here is sel: \n\n{}".format(sel))

# If building a new model,  print the tod used:
# print("Build model on this TOD:  {}".format(d['name'][tod_num]))
# comment out model = buildModel ....  if using a stored model:

scaler = prep.StandardScaler().fit(E)
# scaler = prep.RobustScaler().fit(E)
Et = scaler.transform(E)
print("Using StandardScaler")


model = buildModel(Et, sel, length)

model_notes = "In model, metrics=Accurary; optimizer = 'adam'\nUsing Standard Scaler\nprediction cut-off set at 0.01"
# notes = "one TOD model"

notes = "three TOD model"
# notes = "two TOD models;  uses tod_num 9 and 0"
# model.fit(E,sel, epochs = 50, batch_size = 100)
# model.fit(E,sel, epochs = 50)
# print("Diagnostic, from main:  here is the tod_names list: {}\t{}".format(type(tod_names), tod_names))
# print("Diagnostic,  name of tod #{}:  {}".format(tod_num, tod_names[tod_num]))
# N_samples = len(E[4]) - 1
# N_samples = adj_length 

# model_name = "pa6_f090_s18_c11_v1_tod_0_8Feb"
# model_name = "pa7_f030_s20_tod49_trimmed_30Dec2020
# model_name = "pa7_f030_bri_tod_0_8Feb"
# model_name = "pa5_f090_s19_c11_v0_trim_7Feb_tod#0"
# model_name = "pa5_f090_s19_c11_v0_trim_8Feb_tod#0"
# print(" Model Name  =    {}\n".format(model_name))

# model_name = "second_pass_on_cuts_pa6"
# model = models.load_model(model_name)


grand_tot = 0
tod_index = 0

N_samples = len(tod_names)
# Change N_samples to a small number for development testing
#while tod_index < N_samples: 
for tod_index in range(3, 30 ):
    E, s, tod_number, tod_name, length     = transfer(d, tod_index, tod_names)
        
#    print("N_samples =   {}".format(N_samples)) 
    print("tod_num = {}\tlength =  {}\t  {}".format(tod_index, length, tod_name))
# make class predictions with the model
#    print("-------from loop-----------\nshape of E = {}".format(E.shape))
# predictions = model.predict_classes(D)
    scaler =  prep.RobustScaler().fit(E)
    Et = scaler.transform(E)
    predictions = model.predict(Et) 
    _, Recall = model.evaluate(Et, s)
    print("{}\tRecall = {}".format(tod_name, Recall)) 
    pred = np.array(predictions[:])
    i = 0
#    true_cuts = []
#    false_cuts = []
#    false_uncuts = []
#    true_uncuts= []
    tc_list = []
    fc_list = []
    fu_list = []
    tu_list = []
    list_dict = {}
    E_fc = DataFrame()
    E_tc = DataFrame()
    E_tu = DataFrame()
    E_fu = DataFrame()
    E_false = DataFrame()
    E_cut = DataFrame()
    s_false = np.zeros(0)
    s_cut = np.zeros(0) 
    while i < length: 
        grand_tot += 1 
        x4 = det_number[i]
#        print("i= {}\tpred[i] = {}\tx4 = {}".format(i,pred[i],x4))
#        print("Vector = \n{}".format(E.iloc[i])) 

# the line with the " < 0.01 " is to avoid rounding issues with identifying pred[i] == 0    
# can set to other cut-off thresholds, such as 0.1
        if pred[i] < 0.01: 
#        if pred[i] < 0.5:
            xxx = 0
        else:
            xxx = 1
        if xxx == 0 and s[i] == 0:
            t_cut += 1
#            E_tc = E_fu.append(E.iloc[i])
#            E_cut = E_cut.append(E.iloc[i])
#            s_cut = np.append(s_cut, s[i])
#            tc_list.append(x4)
        if xxx == 0 and s[i] == 1: 
            f_cut += 1
#            fc_list.append(x4)
#            E_fc = E_fc.append(E.iloc[i])
#            E_false = E_false.append(E.iloc[i])
#            s_false = np.append(s_false,s[i])
#            E_cut = E_cut.append(E.iloc[i])
#            s_cut = np.append(s_cut, s[i])  
        if xxx == 1 and s[i] == 0: 
            f_uncut += 1
#            fu_list.append(x4)
#            print("from loop; det_number = {}".format(x4)) 
#            E_fu = E_fu.append(E.iloc[i])
#            E_false = E_false.append(E.iloc[i])
#            s_false = np.append(s_false, s[i])
        if xxx == 1 and s[i] == 1: 
            t_uncut += 1
#            tu_list.append(x4)
#            E_tu = E_tu.append(E.iloc[i])
                  
        i += 1
#    print("tc: \n{}\n\nfc: \n{}".format(tc_list, fc_list))
            
#    print("fu: \n{}\n\ntu: \n{}".format(fu_list, tu_list))
    if tod_index == 3:
        list_pk = open("good_bad.pk","wb")
        list_dict["tc"] = tc_list
        list_dict["fc"] = fc_list
        list_dict["tu"] = tu_list
        list_dict["fu"] = fu_list
        list_dict["name"] = tod_name
        pk.dump(list_dict, list_pk)

        list_pk.close()        
#    print("Detector #   {}".format(tod_num))
#    print("predictions: \n ")
#    l_cuts = len(s_cut)
#    predictions2 = model2.predict(E_cut)    
#    tc, fc, tu, fu, E_cut, s_cut = anal(E_cut,predictions2, s_cut) 
#    print ("  tc    fc    tu    fu")
#    print (" {}    {}    {}    {}".format(tc, fc, tu, fu))
#   t2_uncut = t_uncut + tu
#    f2_uncut = f_uncut + fu
#    t2_cut   = t2_cut
#    f2_cut   = f2_cut
    tod_index += 1

print("-----------------------------------\n")
# print("E_fc is  {}".format(E_fc.describe()))
# print("E_fc is \n{}".format(E_fc))


tot = t_cut + t_uncut + f_cut + f_uncut

print ("grand total = {}\ttotal analyzed = {}".format(grand_tot, tot))
print("Model was trained after eliminating samples for ff_sel and tes_sel bad detectors.\nPercentages are for only such samples, after cutting by ff_sel and tes_sel:")
print("true-uncut = {:4.1f}%\ttrue-cut = {:4.1f}%\tfalse-uncut = {:4.1f}%\tfalse-cut = {:4.1f}%\n".format(100.*t_uncut/tot, 100.*t_cut/tot, 100.*f_uncut/tot, 100.*f_cut/tot))

# print("\n percentages of grand totals")

# print("true-uncut = {:4.1f}%\ttrue-cut = {:4.1f}%\tfalse-uncut = {:4.1f}%\tfalse-cut = {:4.1f}%\n".format(100.*t_uncut/grand_tot, 100.*t_cut/grand_tot, 100.0*f_uncut/grand_tot, 100.*f_cut/grand_tot))

print(" ")

tot_accuracy = t_uncut/ (t_uncut + f_uncut)
print("Total accuracy of uncuts: {:4.1f}%".format(100.*tot_accuracy))
print("\nNumbers, not percentages:\n  ")
print("t_uncut = {}\tt_cut = {}\tf_uncut = {}\tf_cut {}".format(t_uncut, t_cut, f_uncut, f_cut))

# print("Build model on two TODs: ")
# print("model built on {}".format(modelTOD))

# print("Build model on tod # {}, also:\t{}".format(tod_num, tod_names[tod_num]))
# print("model name =   {}".format(model_name))
print(model_notes)
print(notes)
print("pwv for tod[{:3d}] = {:6.3g}".format(int(tod_num), pwv_1))

print("pwv for tod[{:3d}] = {:6.3g}".format(int(tod_num2), pwv_2))
print("pwv for tod[{:3d}] = {:6.3g}".format(int(tod_num3), pwv_3))
print("working on:    {}".format(pickle_file_name))
print("\n\nend end end")
