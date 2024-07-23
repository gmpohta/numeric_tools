import matplotlib.pylab as plt
import numpy as np
import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

A=820

def fun(x):
    return (1-A**2/(1+A**2))*np.exp(-A*x)+A/(1+A**2)*(np.sin(x)+A*np.cos(x))

def r_fun(x, y):
    return A*np.cos(x)-A*y

x, y, theor_err = tl.ode_cros(1.0, [0,np.pi], r_fun, 200, out_err=True)

plt.plot(x, y, 'k')
plt.plot(x, fun(x), 'r--', label='analytical')

real_err = max([0 if math.isnan(itt) else abs(itt) for itt in abs(y-fun(x))])

print('Threor error = ', theor_err)
print('Real error = ', real_err)
plt.legend()
plt.show()