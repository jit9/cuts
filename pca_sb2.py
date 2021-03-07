"""
    Builds keras model from pickle parameters
    > contains a reverse block of code which switches sel == 0 for sel == 1 etc.; watch for commenting it out
      for normal configuration
    > this code build the model using only ff_sel == 1 and tes_sel == 1 (the working detectors), called 
      "trimmed"
    > DataFrame E is the original DataFrame D except those detectors lacking ff_sel == 1 and tes_sel == 1 are removed.
"""

import numpy as np
import sys
# sys.path.append(r'/projects/ACT/feynman-users/treu/sklearn')
import sklearn
import sklearn.decomposition
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import tensorflow
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras import models
import pandas
from pandas import DataFrame, Series
from pandas import plotting
import h5py
import pickle as pk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import axes
from matplotlib.pyplot import imshow
# matplotlib.use("qt5Agg", force = True)

def pickle_in(): 
#    f        = open("/projects/ACT/yilung/depot/Postprocess/pickle_cuts/pa4_f150_s19_c11_v0_results.pickle", "rb")
#    f        = open("/projects/ACT/yilung/depot/Postprocess/pickle_cuts/pa6_f150_s19_c11_v0_results.pickle", "rb")
#    f        = open("/projects/ACT/yilung/depot/Postprocess/pickle_cuts/pa4_f220_s19_c11_v0_results.pickle", "rb")



#    path_start = "/projects/ACT/yilung/depot/Postprocess/pickle_cuts"
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pa6_f090_s17_c11_v4"
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pa4_f220_s17_c11_v4"    
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pa3_f150_s16_c11_v0"
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pa6_f090_s18_c11_v1"
#    pa5_f150_s19_c11_v0
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pa5_f090_s19_c11_v0"
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pa4_f220_s17_c11_v4"
#    path_start = "/scratch/gpfs/yilung/depot/Postprocess/pa7_f030_s20_bri_v0" 
    path_start = "/scratch/gpfs/yilung/depot/Postprocess/pa7_f040_s20_bri_v0"
#    path_start = "/projects/ACT/yilung/depot/Postprocess/pickle_cuts"
#    pickle_file_name = "pa5_f090_s19_c11_v0_results.pickle"
#    pickle_file_name = "pa6_f090_s18_c11_v1_results.pickle"    
#    pickle_file_name = "pa7_f030_s20_bri_v0_results.pickle"
    pickle_file_name = "pa7_f040_s20_bri_v0_results.pickle"
#    pickle_file_name = "pa5_f090_s19_c11_v0_results.pickle"

    f = open(path_start+"/"+pickle_file_name, "rb")
    print("Working on file:  \n{}\n".format(f))
    d        = pk.load(f)
    print("Diagnostic from pickle_in fct.   shape of d['sel'] =    {}".format( np.shape(d['sel'])))
    print("Diagnostic from pickle_in fct.   shape of d['ff_sel'] = {}".format(np.shape(d['ff_sel'])))
    print("Diagnostic from pickle_in fct.   shape of d['tes_sel']= {}".format(np.shape(d['tes_sel'])))
    f.close()
    tod_names = d['name']
    return d, pickle_file_name, tod_names

def transfer(d, tod_num, tod_names):
    print("number of TOD available = {}".format(len(d['name'])))
    tes_sel = d['tes_sel'].transpose()        
    ff_sel = d['ff_sel'].transpose()

# Switch sel so that 0 means uncut and 1 means cut
# Commented out to retain conventional configuration which is 0 means cut and
#   1 means good
#  
#    i = 0
#    while i < len(sel):
#        sel[i] = not(sel[i])
#        ff_sel[i] = not(ff_sel[i])
#        tes_sel[i] = not(tes_sel[i]) 
#        i += 1 

    rms = d['rmsLive'].transpose()
    rms = rms[tod_num]
    S_rms = Series(rms)
    sel = d['sel'].transpose()
    sel = sel[tod_num]
    S_sel     = Series(sel)
    kurt     = d['kurtLive'].transpose()
    kurt     = kurt[tod_num]
    S_kurt   = Series(kurt)
    length = len(S_sel)
    i = 0
    while i < length:
        if sel[i] == 0:
            sel[i] = 0
        if sel[i] == 1:
            sel[i] = 1
        i += 1
    S_ff_sel = Series(ff_sel)
    S_tes_sel = Series(tes_sel)
    print("diagnostic:   length of S_sel = {}".format(len(S_sel)))
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

    D = DataFrame(data=[S_ff_sel, S_tes_sel, S_sel, S_rms, S_kurt, S_gainLive, S_corrLive, S_normLive, skewLive,  S_MFELive, S_DELive, S_jumpLive]).transpose()

    features = ['rms', 'kurt', 'gainLive', 'corrLive', 'normLive', 'skewLive', 'MFELive','DELive', 'jumpLive']
    print(str(features))   
    print("DataFrame shape of D: {}".format(D.shape))
# E is a DataFrame with only those rows for which S_ff_sel and S_tes_sel == 1

    E          = D[D[0] == 1]
#    print("Diagnostic:  here is E, first trim:  \n{}".format(E))

    E          = E[E[1] == 1]
#    print("Diagnostic:  here is E, second trim: \n{}".format(E))
# Diagnostic
#    print("Diagnostic:  length of E[2] is  {}\n".format(len(E[2])))   
    sel        = np.array(E[2])
    print("Diagnostic:  shape of sel after trimming  =  {}\n".format(np.shape(sel)))
    E          = E.transpose()[3:].transpose()
# Diagnostic
#    print("Diagnostic:  here is E after removal of first three parameters   \n{}\n\n".format(E))
    print("DataFrame E shape: {}".format(E.shape))
    adj_length = int(np.shape(sel)[0])
    print("AFTER trimming....shape of sel={}".format(np.shape(sel))) 
# Save DataFrame E for external analysis
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
    print("diagnostic:  from transfer fct,  shape of sel =  {}".format(np.shape(sel)))
    print("                                 adj_length   =  {}".format(adj_length))
    return E, sel, adj_length, features

def buildModel(D, sel, length):
# Builds the model

    model = Sequential()
    model.add(Dense(12, input_dim=9, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    print("from buildModel,  shape of D:  {}\n".format(D.shape))
# compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Accuracy'])
# fit the keras model on the dataset
    model.fit(D, sel, epochs=50, batch_size=10)

    _, accuracy = model.evaluate(D, sel)
    print("accuracy of trained model = {}".format(accuracy))
#    model.save("pa5_f150_s19_tod655_trimmed_30Dec2020")
#    model.save("pa7_f030_bri_temp","wb")
#    model.save("pa5_f090_s19_c11_v0_trim_8Feb_tod#0")
#    model.save("pa7_f030_s20_tod49_trimmed_30Dec2020")
#    model.save("pa7_f030_bri_tod_0_8Feb")
#    model.save("pa6_f090_s18_c11_v1_tod_0_8Feb") 
    return model 

# START OF MAIN PROGRAM
#
# load pickle file

d, pickle_file_name, tod_names = pickle_in()
t_cut      = 0
t_uncut    = 0
f_cut      = 0
f_uncut    = 0

print("length of d['name'] is  {}".format(np.shape(d['name'])))
aaa = int(np.shape(d['name'])[0])
i = 0


# np.save( 'tod_names_pa7_f030.npy', d['name'])


tod_num = 0 

E, sel, adj_length, features = transfer(d, tod_num, tod_names)
print("Build model on tod # {}\t{}".format(tod_num, tod_names[tod_num])) 

print("Diagnostic just before calling model:  shape of sel = {}   adj_length = {}".format(np.shape(sel), adj_length))
modelTOD = d['name'][tod_num]
print("Features: \n{}".format(features))


# Experiment with sklearn.PCA

pca = PCA(n_components = 4)
X = StandardScaler().fit_transform(E)
pca.fit(X)
matplotlib.use("qt5Agg", force=True)
ax = plt.axes()

im = ax.imshow(np.corrcoef(X.T), cmap="RdBu_r", vmin=-1, vmax=1)

# ax.set_xticks([0, 1, 2, 3])
# ax.set_xticklabels(list(feature_names), rotation=90)
# ax.set_yticks([0, 1, 2, 3])
# ax.set_yticklabels(list(feature_names))

plt.colorbar(im).ax.set_ylabel("$r$", rotation=0)
ax.set_title("PCA Experimenting")
# plt.tight_layout()

print("GOT TO HERE")
n_comps = 9

# methods = [('PCA', PCA()),
#           ('Unrotated FA', FactorAnalysis()),
#           ('Varimax FA', FactorAnalysis(rotation='varimax'))]

methods = [('PCA', PCA()), ('Unrotated FA', FactorAnalysis())]

fig, axes = plt.subplots(ncols=len(methods), figsize=(10, 8))

for ax, (method, fa) in zip(axes, methods):
    fa.set_params(n_components=n_comps)
    fa.fit(X)

    components = fa.components_.T
    print("\n\n %s :\n" % method)
    print(components)

    vmax = np.abs(components).max()
    ax.imshow(components, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
    ax.set_yticks(np.arange(len(features)))
    if ax.is_first_col():
        ax.set_yticklabels(features)
    else:
        ax.set_yticklabels([])
    ax.set_title(str(method))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Comp. 1", "Comp. 2"])
fig.suptitle("Features")
plt.tight_layout()
plt.show()

# If building a new model,  print the tod used:
# print("Build model on this TOD:  {}".format(d['name'][tod_num]))

# comment out model = buildModel ....  if using a stored model:
model = buildModel(E, sel, adj_length)

# fit the keras model on the dataset
# the model should be fit in the buildModel function not here 
model.fit(E,sel, epochs = 50, batch_size = 100)
# model.fit(E,sel, epochs = 50)

print("Diagnostic, from main:  here is the tod_names list: {}\t{}".format(type(tod_names), tod_names))
print("Diagnostic,  name of tod #{}:  {}".format(tod_num, tod_names[tod_num]))
# N_samples = len(E[4]) - 1
# N_samples = adj_length 

# model_name = "pa6_f090_s18_c11_v1_tod_0_8Feb"
# model_name = "pa7_f030_s20_tod49_trimmed_30Dec2020
# model_name = "pa7_f030_bri_tod_0_8Feb"
# model_name = "pa5_f090_s19_c11_v0_trim_7Feb_tod#0"
# model_name = "pa5_f090_s19_c11_v0_trim_8Feb_tod#0"
# print(" Model Name  =    {}\n".format(model_name))
print("Uses model generated within this data run.")
# model = models.load_model(model_name)

grand_tot = 0
tod_index = 0

N_samples = len(tod_names)
# Change N_samples to a small number for development testing
# N_samples = 50
print("Diagnostic:   N_samples = {}".format(N_samples))
while tod_index < N_samples: 
    E, s, length, features     = transfer(d, tod_index, tod_names)
    print("N_samples =   {}".format(N_samples)) 
    print("tod_num = {}\tlength =  {}\t  {}".format(tod_index, length, d['name'][tod_index]))
# make class predictions with the model

# predictions = model.predict_classes(D)
    predictions = model.predict(E) 
    _, accuracy = model.evaluate(E, s)
    
    pred = np.array(predictions[:])
# the line with the " < 0.01 " is to avoid rounding issues with identifying pred[i] == 0    
    i = 0
    while i < length: 
        grand_tot += 1 
        if pred[i] < 0.01: 
            xxx = 0
        else:
            xxx = 1
#        print("ff = {}\t\ttes = {}".format(ff_sel[i], tes_sel[i])) 
#        if ff_sel[i] == 0  and tes_sel[i] == 0:
        if xxx == 0 and s[i] == 0: t_cut += 1
        if xxx == 0 and s[i] == 1: f_cut += 1
        if xxx == 1 and s[i] == 0: f_uncut += 1
        if xxx == 1 and s[i] == 1: t_uncut += 1
        i += 1

#    print("Detector #   {}".format(tod_num))
#    print("predictions: \n ")

    tod_index += 1
print("------------------------------")
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

print("model built on {}".format(modelTOD))
# print("model name =   {}".format(model_name))
print("working on:    {}".format(pickle_file_name))
print("\n\nend end end")
