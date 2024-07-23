import math
import time
import numpy as np
import matplotlib.pylab as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

def test(x):
    return math.sin(x)/x-1

x=np.linspace(0,5,100)
y=[test(i) for i in x]

a=plt.figure()
plt.plot(x,y)
plt.grid()
a.show()

lim=(0,x[-1])
time._startTime = time.time()
xout,h=tl.rootN(1,test,1)
xi=xout[-1]
print('Elapsed time sims: ', time.time() - time._startTime)
print(xi)
print(h)
plt.plot(xi,test(xi),'ro')
plt.show()
