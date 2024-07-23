import time
import numpy as np
import matplotlib.pylab as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

nint=20

wi=np.linspace(0, 2 * np.pi, nint)
yi=np.sin(wi)+np.random.uniform(-0.1,0.1,nint)
yn=np.cos(wi)

time._startTime = time.time()
p=tl.Spline(wi,yi,[0,0])
print(p.calcspln(3*np.pi))
print('Elapsed time sims: ', time.time() - time._startTime)
pol=tl.polyapr(wi,yi,8)
dp=p.splder()

w=np.linspace(0,2*np.pi,1000)
y=dp.calcspln(w)
y1=p.calcspln(w)

pold=tl.polyder(pol)
ypold=tl.polyval(pold,w)
ypol=tl.polyval(pol,w)

ysin=np.sin(w)
ydsin=np.cos(w)

plt.figure()
plt.plot(w,ysin)
plt.plot(wi,yi,'or')
plt.plot(w,ypol,'k')
plt.plot(w,y1,'--')

plt.figure()
plt.plot(w,ydsin)
plt.plot(w,y)
plt.plot(w,ypold,'k')
plt.show()
