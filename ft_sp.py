###
###    Reads tod data from other code
###    Shows TOD plot of detector selected as detnum before any pre-treatment of the TOD data
###    Calculates power spectrum after applying a Hann window
###    Also calculated power spectrum using moby2 toolkit tool.power module
###    Seems to get a slightly different normalized power spectrum via tool.power compared to
###         numpy.fft routine        

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from scipy import signal
from scipy.signal import blackman
import tools
from tools import power
import time



def ft(tod, n):

#    print 'time of start of ft ',time.localtime(time.time()) 

    len = int(np.shape(tod)[0])
    N = len

    T = N/400.
    x = np.linspace(0.0, N * T, N)
    y = tod
    window = signal.hann(len)


###
### check units:   does abs function below return a 'power' ?
###
#    ywf = fft.real((2.0/N)*y*window)*fft.real((2.0/N)*y*window) + fft.imag((2.0/N)*y*window)*fft.imag((2.0/N)*y*window)
    ywf = abs(fft((2.0/N)*y*window,n))
#    print ' after fft line:  ', time.localtime(time.time()) 
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

#    plt.semilogy(xf[1:N/2], 2.0/N*np.abs(yf[1:N/2]), 'b')
#### hold the graphing:
#    plt.semilogy(xf[1:N/2], 2.0/N*np.abs(ywf[1:N/2]), 'g')
#    plt.legend(['FFT  -   TOD'])
#    plt.grid()
#    plt.axis(xmin = 0.00000001 ,  xmax = 0.00015)
#plt.savefig('FFT_TOD_500.png') 
#    plt.show()


    av = np.mean(ywf)
#y_power = power((2.0/N)*y*window, 2.*T, binsize = 0, Hann = True)
#y_power1 = y_power[0][:34999]

#A = xf[1:N/2]
#B = y_power1
#print 'shape of A ', np.shape(A)
#print 'shape of B ', np.shape(B)


#plt.semilogy(A, B, 'g')
#plt.show()

#print " shape, y_power1", np.shape(y_power1)

#print "  shape, xf ", np.shape(xf)


#print 'np.mean(ywf[0:1000])/av= ', np.mean(ywf[0:1000])/av, '    [1100:3000]/av=  ', np.mean(ywf[1100:3000])/av
#av_power = np.mean(B)

#print "Now for the moby2 power tool: "
#print 'np.mean(B[0:1000])/av_power= ', np.mean(B[0:1000])/av_power, '    [1100:3000]/av_power=  ', np.mean(ywf[1100:3000])/av_power


#print ' end end end '
    if av == 0.0: pav_low = 0.00
    else:      pav_low =  np.mean(ywf[0:1000])/av

    if av == 0.0: pav_hi = 0.00
    else:     pav_hi  =  np.mean(ywf[1100:3000])/av
#    print 'time at end of fft  ', time.localtime(time.time()) ,  '\n'
    return pav_low, pav_hi

