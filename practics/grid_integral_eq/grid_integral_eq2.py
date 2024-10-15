import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#
# Grid method for solving integral equations
#
#                                x
#                                /\
# The equation xy' + y = 1 + x + | y dx
#                               \/
#                                0
#
# Grid method:
#
# x_i(y_{i+1}-y_{i-1})/2/d + y_i - d*sum(y_{j+1}+y_j,j=0, j=i-1)/2 = x_i + 1 for i = 1 .. N-1
# x_i(-3y_0 + 4y_1 - y_2)/2/d + y_0 = x_0 + 1
# x_i(3y_N - 4y_{N-1} + y_{N-2})/2/d + y_N - d*sum(y_{j+1}+y_j,j=0, j=N-1)/2 = x_N + 1

def system(t, Y):
    y, z = Y

    if t == 0:
        t = 1e-8
    dydt = z
    dzdt = (1 + y - 2*z) / t
    return [dydt, dzdt]

y0 = 1
dy0 = 1/2

t_span = (1e-8, 4)
Y0 = [y0, dy0]

sol = solve_ivp(system, t_span, Y0, t_eval=np.linspace(1e-8, 4, 1000))

N = 100
lim = [0, 4]

def solve(lim, N):
    d = (lim[1] - lim[0])/N

    xa=np.linspace(lim[0], lim[1], N+1)
    A = np.zeros((N+1,N+1))
    B = np.zeros(N+1)

    A[0, 0] = -3*xa[0]/2/d + 1
    A[0, 1] = 2*xa[0]/d
    A[0, 2] = -xa[0]/2/d
    B[0] = xa[0] + 1

    for ii in range(1,N):
        B[ii] = xa[ii] + 1

        A[ii, 0:ii] = -d
        A[ii, 0] = -d/2
        A[ii, ii] = -d/2

        A[ii, ii-1] += -xa[ii]/2/d
        A[ii, ii] += 1
        A[ii, ii+1] = xa[ii]/2/d

    A[-1, :] = -d
    A[-1, 0] = -d/2
    A[-1, -1] = -d/2

    A[-1, -1] += 3*xa[-1]/2/d + 1
    A[-1, -2] += -2*xa[-1]/d
    A[-1, -3] += xa[-1]/2/d
    B[-1] = xa[-1] + 1

    ya = np.linalg.solve(A, B)

    return [xa, ya]

xa, ya = solve(lim, N)

plt.plot(xa, ya, 'r')
plt.plot(xa, ya, 'ro')

plt.plot(sol.t, sol.y[0], label="y(t)")
plt.xlabel("x")
plt.ylabel("y, y'")
plt.title("Решение ОДУ")
plt.legend()
plt.grid(True)
plt.show()