import matplotlib.pylab as plt
import numpy as np

#
# Grid method for solving integral equations
#
#                        x
#                       /\
# The equation y' = x + | y dx
#                      \/
#                       0
#
# Grid method:
#
# (yi+1-yi-1)/2/d - d*sum(yj+1+yj,j=0, j=i-1)/2 = xi

N = 10
lim = [0, 2]
y0 = 1

def solve(lim, N):
    d = (lim[1] - lim[0])/N

    xa=np.linspace(lim[0], lim[1], N+1)
    A = np.zeros((N+1,N+1))
    B = np.zeros(N+1)

    A[0, 0] = 1
    B[0] = y0

    for ii in range(1,N):
        B[ii] = xa[ii]

        A[ii, 0:ii] = -d
        A[ii, 0] = -d/2
        A[ii, ii-1] += -1/2/d
        A[ii, ii] = -d/2
        A[ii, ii+1] = 1/2/d

    A[-1, :] = -d
    A[-1, 0] = -d/2
    A[-1, -1] = -d/2

    A[-1, -1] += 3/2/d
    A[-1, -2] += -2/d
    A[-1, -3] += 1/2/d
    B[-1] = xa[-1]

    ya = np.linalg.solve(A, B)

    return [xa, ya]

xa, ya = solve(lim, N)
xa1, ya1 = solve(lim, N*2)
xa2, ya2 = solve(lim, N*4)

dy1 = np.max(np.abs(ya - ya1[::2]))
dy2 = np.max(np.abs(ya1 - ya2[::2]))
q=dy1/dy2
theor_err = np.log(dy1/(q-1))/np.log(10)

real_err=np.log(np.max(np.abs(ya-1*np.exp(xa)-1*np.exp(-xa)+1)))/np.log(10)

plt.plot(xa, ya, 'r', label="Numerical method, real error = %.2f; theor error = %.2f" % (real_err, theor_err))
plt.plot(xa, ya, 'ro')

x = np.linspace(0, 2, 100)
plt.plot(x, 1*np.exp(x)+1*np.exp(-x)-1, label=r'Analytical solution $y = e^x + e^{-x} - 1$')

plt.grid()
plt.legend()
plt.show()
