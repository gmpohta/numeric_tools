
import matplotlib.pylab as plt
import time
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

en=3/500

def int1(y):
    return (1-4/3*y-5/6*en)*np.exp(-y/(en+2/5*y))/(1-y)**2.5/(en+2/5*y)

lim=[0,0.1]
x=np.linspace(lim[0],lim[1],1000)


time._startTime = time.time()
U=tl.intEitken(lim=lim,fName=int1,nPow=16,diagnostic=1)
print('Elapsed time R: ', time.time() - time._startTime)
print(U)
u0=np.pi/3**0.5
print(u0,U[0]-u0)

plt.figure()
plt.plot(x,int1(x))
plt.grid()
plt.show()