import matplotlib.pylab as plt
import numpy as np

def fun(x):
    return (x+1)**2

def int_middle(lim,fun,n):
    out=0
    dx=(lim[1]-lim[0])/n
    x=np.linspace(lim[0]+dx/2,lim[1]-dx/2,n)
    for ii in range(n):
        out+=fun(x[ii])
    return out*dx

I=7/3

lim=[0,1]
r=2
nPow=17
n_grid = 2 ** np.arange(nPow) #1,2,4,8,..
p = np.ones(nPow)*np.nan
R_theor=np.ones(nPow)*np.nan
U = np.zeros(nPow)
for ii in range(nPow):
    U[ii] = int_middle(lim, fun, n_grid[ii])

for ii in range(2,nPow):
    p[ii] = 1 / np.log(r) * np.log((U[ii-1]-U[ii-2])/(U[ii]-U[ii-1]))
    R_theor[ii] = abs(U[ii] - U[ii - 1]) / (r ** p[ii] - 1)


plt.plot(n_grid,np.log(abs(U-I)) / np.log(10),'r',label='Real error')
plt.plot(n_grid,np.log(abs(R_theor)) / np.log(10),'k',label='Theoretical error')
plt.grid()
plt.xlabel('n_grid')
plt.ylabel('lg(R)')
plt.xscale('log')
plt.legend()

plt.figure()
plt.plot(n_grid,p,'r',label='Middle rectangle')
plt.legend()
plt.grid()
plt.xscale('log')
plt.xlabel('n_grid')
plt.ylabel('p')
plt.show()