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
dLong = np.zeros(9410016)
d = np.zeros(1000000)
d_test = np.zeros(3000000)
data = cPickle.load(open('mr3_pa2_s14_results.pickle','r'))


print (type(data))

print data['name'][data['sel'] == 1.0][0:10]

