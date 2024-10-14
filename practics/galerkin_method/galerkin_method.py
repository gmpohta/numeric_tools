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

N =10
d=0.1

def addRange(x0, y0, sum=0):
    y1=(d/3+x0/2-(1/6-8*d/24+d/3)*y0+d/4*sum)/(1/3-d/8)
    return [d+x0, y1]

x = np.linspace(0, 2, 100)
xa=np.zeros(N+1)
ya=np.zeros(N+1)
xa[0]=0
ya[0]=0

plt.plot(x, np.exp(x)-1, label=r'Analytical solution $y = e^x - 1$')

for ii in range(N):
    sum = ya[0] + ya[ii+1] + 2*np.sum(ya[1:ii+1])
    xtmp, ytmp = addRange(xa[ii], ya[ii], sum)
    xa[ii+1] = xtmp
    ya[ii+1] = ytmp

err=np.max(np.abs(ya-np.exp(xa)+1)/(np.exp(xa)+1)*100)

plt.plot(xa, ya, 'r', label="Galerkin's method, max error = %.2f%%" % (err))
plt.plot(xa, ya, 'ro')

plt.grid()
plt.legend()
plt.show()
