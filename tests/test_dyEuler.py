import numpy as np
import matplotlib.pylab as plt
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

y0=-1
nump=10

x1=np.linspace(0,2*np.pi,nump+1)
y1=0.5*np.sin(x1)-0.5*np.cos(x1)+(y0+0.5)*np.exp(-x1)
#y' =-y+sin(t)

def f(t,y):
    return -y+np.sin(t)

x,y,err=tl.dyEuler([y0],[0,2*np.pi],[f],nintervals=nump,out_err=1)
e=abs(y-y1)
newe=[]
for itt in e:
    if math.isnan(itt):
        newe.append(0)
    else:
        newe.append(abs(itt))
e=max(newe)
print(err,e)
plt.plot(x,y)
plt.plot(x1,y1,'ro')
plt.show()
