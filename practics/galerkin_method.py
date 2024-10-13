import matplotlib.pylab as plt
import numpy as np

#
# Galerkin's method for solving integral equations
#
#                       x
#                      /\
# The equation y = x + | y dx
#                     \/
#                      0
#
#
# Basis function w(x) = (1 - (x-x0)/d) * y0 +(x-x0)/d * y1 or w(x) = y0 + (y1 - y0)/d * (x-x0),
# where x0 - start interval, x1 - end interval, d = x1 - x0, y0, y1 - weight
# 
#                                                         x        x0
#                                                        /\       /\
# residual R = y0 + (y1 - y0)/d * (x-x0) - (x-x0) - x0 - | y dx - | y dx
#                                                       \/       \/
#                                                       x0        0
#
#                                                x
#                                               /\
# R = y0 + (y1 - y0)/d * (x-x0) - (x-x0) - x0 - | w(x) dx - (y0 - x0)
#                                              \/       
#                                              x0
# 
# R = (y1 - y0)/d * (x-x0) - (x-x0) - y0*(x-x0) - (y1 - y0)/2/d*(x-x0)**2
#
# Galerkin's method
#
#  x0+d
# /\
#  | R*(x-x0)dx = 0
#  \/
#  x0
#
# and
#
#  x0+d
# /\
#  | R*(1 - (x-x0)/d))dx = 0
#  \/
#  x0
#
# Equation for get weights y1, y0
#
# -(8/d + 5)*y0 +(8/d - 3)*y1 = 8
#
# when x0 = 0 y0 = 0 from bondary condition


N =10
d=0.1
x0=0
y0=0

def addRange(x0, y0):
    y1=(8 + (8/d + 5) * y0) / (8/d - 3)
    return [d+x0, y1]

x = np.linspace(0, 2, 100)
xa=np.zeros(N+1)
ya=np.zeros(N+1)
xa[0]=x0
ya[0]=y0

plt.plot(x, np.exp(x)-1, label=r'Analytical solution $y = e^x - 1$')

for ii in range(N):
    xtmp, ytmp = addRange(xa[ii], ya[ii])
    xa[ii+1] = xtmp
    ya[ii+1] = ytmp

err=np.max(np.abs(ya-np.exp(xa)+1)/(np.exp(xa)+1)*100)

plt.plot(xa, ya, 'r', label="Galerkin's method, max error = %.2f%%" % (err))
plt.plot(xa, ya, 'ro')

plt.grid()
plt.legend()
plt.show()
