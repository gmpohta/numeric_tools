import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

u = 10

def analyt(xg, tg):
    out = np.zeros(xg.shape)
    for ii in range(xg.shape[0]):
        for jj in range(xg.shape[1]):
            if xg[ii, jj] - u * tg[ii, jj] >= 0:
                out[ii, jj] = 4 - (xg[ii, jj] - u * tg[ii, jj])**2
            else:
                out[ii, jj] = 4 + np.sin(10 * (-xg[ii, jj] / u + tg[ii, jj]))
    return out


def init_cond(x):
    #return 4 - x**2
    out=np.zeros(x.shape)
    out[10:60]=1
    return out


def bound_cond(t):
    #return 4 + np.sin(10 * t)
    return np.zeros(t.shape)

def perenos(lims, init, bound, Nx, Nt):
    T, X = map(lims.get, ["t", "x"])

    x = np.linspace(*X, Nx + 1)
    t = np.linspace(*T, Nt + 1)

    dt = (T[1] - T[0]) / Nt
    dx = (X[1] - X[0]) / Nx

    U = np.zeros((Nx + 1, Nt + 1))

    U[0, :] = bound(t)
    U[:, 0] = init(x)

    for nt in range(1, Nt + 1):
        for nx in range(1, Nx + 1):
            U[nx, nt] = U[nx, nt - 1] - u * dt / dx * (U[nx, nt - 1] - U[nx - 1, nt - 1])

    return U, np.meshgrid(x, t)


lims = dict(t=[0, 1], x=[-100, 100])
Nx = 500
Nt = 500

#accurate_result, xr1 = accurate(lim, N)
result, xr = perenos(lims, init_cond, bound_cond, Nx, Nt)

fig = plt.figure(figsize=(10, 8))
#plt.plot(xr1, accurate_result, "-o", label="fun")

x, t = xr
uan = analyt(x, t).T

axes = fig.add_subplot(projection='3d')
axes.plot_wireframe(x, t, result, color="black")
#axes.plot_wireframe(x, t, uan, color="green")

axes.set_xlim(*lims["x"])
axes.set_ylim(*lims["t"])
axes.set_xlabel("X")
axes.set_ylabel("t")



r=2
result1, xr1=perenos(lims, init_cond, bound_cond, Nx, Nt)
x1, t1 = xr1
result2, xr2=perenos(lims, init_cond, bound_cond, Nx*r, Nt*r)
x2, t2 = xr2
uan = analyt(x2, t2).T

R_real=np.max(abs(uan-result2))
R_theor=np.max(abs(result2[::2,::2]-result1)/(1-1/r))
print('treor=',R_theor,'real=',R_real)
plt.show()