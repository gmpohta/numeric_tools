import matplotlib.pylab as plt
import time
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

def test(x):
    return 1/(-x**3+x**2)**(1/3)

def test2(x):
    return 1/(1-x)**2.5

lim=[0,1]
x=np.linspace(lim[0],lim[1],100)

time._startTime = time.time()
U=tl.intEitken(lim=lim,fName=test2,nPow=22,diagnostic=1)
print('Elapsed time R: ', time.time() - time._startTime)
print(U)
u0=2*np.pi/3**0.5
print(u0,U[0]-u0)
plt.show()