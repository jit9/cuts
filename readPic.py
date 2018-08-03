import numpy as np
import cPickle
import os, sys
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame
from pandas.plotting import scatter_matrix


d = {}
data = cPickle.load(open('mr3_pa2_s14_results.pickle','r'))

keys = data.keys()
print "keys are ... \n", keys 

#todName =  '1408492556.1408492589.ar2'
todName2 = '1408583161.1408583225.ar2'
todName = '1408497679.1408501297.ar2'


for k in keys:
    if data['name'] == todName:
        print "yes !!! "
        d = data['name']

print type(d)
print data['name'][0:10]
kks = []
culdKeys =  ['sel','corrLive',  'rmsLive',  'kurtLive',  'skewLive', 'normLive', 'darkRatioLive', 'MFELive', 'jumpDark', 'gainDark', 'gainLive', 'DELive','jumpLive'] 

print 'culdKeys   ', culdKeys


D = DataFrame()
Da = DataFrame()

print 'ok'  
for k in culdKeys:
    print ' working on ', k 
    d = data[k].transpose()
    print "shape of ", k, " = ", np.shape(d)
    i = 0
    while i < 8000:
        if data['name'] == todName:
            iName = i
        i += 1
    print 'name number is ', i
    print 'shape is ', np.shape(d[i])
    Da = DataFrame({k:d[i]})
    D  = pandas.concat([D,Da], axis = 1 )


print 'D is  '
print D
        
scatter_matrix(D, alpha=0.2, figsize=(8, 8), diagonal='kde')
plt.show()   
                
Dgood = DataFrame()
Dgood = D[D['sel'] == 1.0]

Dbad  = DataFrame()
Dbad  = D[D['sel'] == 0.0]


print 'Dgood'
print Dgood
print '\n\n\n'
print 'Dbad'
print Dbad

Dgood.plot.hist()
plt.savefig('Dgood_'+todName+'.png')
plt.show()

Dbad.plot.hist()
plt.savefig('Dbad_'+todName+'.png')
plt.show()

'''
###
###  it would be nice to enlongate the data sets to encompass more than one TOD
###


i = 0
while i < 1056:
    if data['name'] == todName2:
        print " match # 2 "
        for k in keys:
            data3 = data[k]
            data3 = np.transpose(data3)
            d[k] = np.append(d[k], data3)
            print k, '\t', np.shape(d[k])
    i += 1





dpw = open('data_'+todName, 'w')

cPickle.dump(d, dpw)

dpw.close()

print 'end;  now read back the data...'

dpr = open('data_'+todName, 'r')

q = cPickle.load(dpr)

dpr.close()
'''
print "\n"


### instead of read back --- just move forward with analysis
###

''' 
print np.shape(d)

#print q['darkRatioLive'][0:20]

plt.scatter(d['corrDark'], d['sel'])
plt.show()




d = data['sel']
d2 = data['name']
print 'shape of d is  ',  np.shape(d), ' shape of name is ', np.shape(d2)



d1 = data['respSel'].reshape(9410016)
d2 = data['corrLive'].reshape(9410016)

plt.scatter(d1, d2)
plt.show()
'''
