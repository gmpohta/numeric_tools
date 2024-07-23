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

x, y, err = tl.ode45([1,0], [0,1], f, n_int=nump, out_err=True)

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

plt.show()
