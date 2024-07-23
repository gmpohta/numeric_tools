import matplotlib.pylab as plt
import time
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

def test(x):
    if x[1]>=0 and x[1]<=2 and x[0]>=0 and x[0]<=2*x[1]:
        return x[1]**2*np.exp(-x[0]*x[1]/8)
    else:
        return 0

def test3(x):
    if 0<=x[1] and x[1]<=1 and x[0]>=0 and x[0]<=1-x[1] and x[2]>=0 and x[2]<=x[0]+x[1]:
        return 15*(x[1]**2+x[2]**2)
    else:
        return 0

nint=2**np.linspace(3,17)
interv=[]
integr=[]
for ii in nint.astype(int):
    interv.append(ii)
time._startTime = time.time()
for ii in nint.astype(int):
    integr.append(abs(tl.intMonte([[0,4],[0,2]],test,ii)-16/np.exp(1)))
print('Elapsed time un: ', time.time() - time._startTime)
time._startTime = time.time()

integr3=[]
for ii in nint.astype(int):
    integr3.append(abs(tl.intMonte([[0,1],[0,1],[0,2]],test3,ii)-2))
print('Elapsed time un: ', time.time() - time._startTime)
plt.loglog(interv,integr,'k')
plt.loglog(interv,integr3,'r')
plt.grid()
plt.show()