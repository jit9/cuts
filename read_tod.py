import numpy as np
import pickle
import matplotlib.pyplot as plt


t836 = np.load('tod_836.npy')

t1000 = np.load('tod_1000.npy')

plt.plot(t836,'r')
plt.title('t_836')
plt.show()
plt.plot(t1000, 'g')
plt.title('t_1000')
plt.show()

