import time
import numpy as np
import matplotlib.pylab as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

ndat=25
npintr=6
nspl=ndat//npintr

wi=np.linspace(0,2*np.pi,ndat)
yi=np.sin(wi)+np.random.uniform(-0.1,0.1,ndat)
w=np.linspace(0,2*np.pi,1000)
yderreal=np.cos(w)

time._startTime = time.time()
p=tl.B_Spline(wi,yi,npintr=npintr,pspl=5)
y1=p.calcspl(w)
yder=p.calcderspl(w)
print('Elapsed time: ', time.time() - time._startTime)
b,x=p.getbasis()

yp=np.zeros(len(p.x)-2)

ints=p.getintervals()
yints=p.calcspl(ints)

plt.figure()
plt.plot(wi,yi,'sr')
plt.plot(ints,yints,'bo')
plt.plot(w,y1)

plt.figure()
for itt in b:
    plt.plot(x,itt)

plt.plot(p.x[1:-1:1],yp,'ob')

plt.figure()
plt.plot(w,yder,'b')
plt.plot(w,yderreal,'r')
plt.show()
