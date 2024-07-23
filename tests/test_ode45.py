import sys
import numpy as np
import matplotlib.pylab as plt
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

nump=100

# ДУ второго
# y''=sinx
# гран условия
# y(0)=1
# y'(0)=0
def f(y,t):
    return y[1],np.sin(t)

x1=np.linspace(0,1,nump+1)
y0=-np.sin(x1)+x1+1
y1=-np.cos(x1)+1

x,y,err,q=tl.ode45([1,0],[0,1],f,n_int=nump,out_mode='diagn')
e=abs(y[:,0]-y0)
newe=[]
for itt in e:
    if math.isnan(itt):
        newe.append(0)
    else:
        newe.append(abs(itt))
e=max(newe)
print(err,e)


plt.plot(x1,y0,'ko')
plt.plot(x1,y1,'bo')
plt.plot(x,y[:,0],'k')
plt.plot(x,y[:,1],'b')
print(q)
'''

y0=1
nump=100

x1=np.linspace(0,100,nump+1)
y1=y0*np.exp(-100*x1)
#y' =y

def f(t,y):
    return -100*y

x,y,err,q=tl.ode45([y0],[0,100],f,n_int=nump,out_mode='diagn')
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
print(q)'''

plt.show()
