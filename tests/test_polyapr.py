import matplotlib.pylab as plt
import math as mat
import time
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

xi=np.linspace(0,1.5*mat.pi,50)
t=xi[0].copy()
xi[0]=xi[-1]
xi[-1]=t

yi=np.sin(xi)


xs=np.linspace(0,1.5*mat.pi,500)
ys=np.sin(xs)

time._startTime = time.time()
k=tl.polyapr(xi,yi,50)

x=np.linspace(0,1.5*mat.pi,500)
y=tl.polyval(np.array(k),x)

print('Elapsed time sims: ', time.time() - time._startTime)

plt.plot(xi,yi,'wo',mew=2,ms=5,mec='r')
plt.plot(x,y,'k')
plt.plot(xs,ys,'b')
plt.show()
