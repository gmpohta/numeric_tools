import numpy as np
import matplotlib.pylab as plt
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

y0 = -1
n_int = 10

#y' =-y+sin(t)
def f(t,y):
    return -y+np.sin(t)

x_an=np.linspace(0,2*np.pi,n_int+1)
y_an=0.5*np.sin(x_an)-0.5*np.cos(x_an)+(y0+0.5)*np.exp(-x_an)

x, y, theor_err=tl.ode_euler([y0], [0,2*np.pi],[f], n_int, out_err=True)

real_err = max([0 if math.isnan(itt) else abs(itt) for itt in abs(y-y_an)])

print('Threor error = ', theor_err)
print('Real error = ', real_err)

plt.plot(x,y)
plt.plot(x_an,y_an,'ro')
plt.show()
