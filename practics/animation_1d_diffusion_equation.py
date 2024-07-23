import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from matplotlib.animation import FuncAnimation

def init(x):
    out=np.zeros(len(x))
    out[20:40]=2
    out[60:65]=4
    return out
def bound(t):
    return[np.zeros(t.shape),np.zeros(t.shape)]

def parab_eq(nt,nx,init,bound):
    def tridiag_alg(a,b,c,r):
        #a - first element not used
        #c - end element not used
        #b - main diagonal
        #a - lower diaganal
        #c - upper diaganal
        n=len(b)
        alf=np.zeros(n)
        beta=np.zeros(n)
        out=np.zeros(n)
        alf[0]=-c[0]/b[0]
        beta[0]=r[0]/b[0]
        for ii in range(1,n):
            alf[ii]=-c[ii]/(b[ii]+alf[ii-1]*a[ii])
            beta[ii]=(r[ii]-a[ii]*beta[ii-1])/(b[ii]+alf[ii-1]*a[ii])
        out[-1]=beta[-1]
        for ii in reversed(range(n-1)):
            out[ii]=alf[ii]*out[ii+1]+beta[ii]
        return out
    #nx,nt - number of intervals
    def solve_layer(u,coun_t):
        a=-kdif*dt/dx**2/2.0*np.ones(nx+1)
        b=(1+kdif*dt/dx**2.0)*np.ones(nx+1)
        c=-kdif*dt/dx**2/2.0*np.ones(nx+1)
        r=np.zeros(nx+1)
        for ii in range(1,nx):
            op1=kdif*dt/dx**2/2.0*(u[ii+1,coun_t]-2*u[ii,coun_t]+u[ii-1,coun_t])
            r[ii]=u[ii,coun_t]+op1
        r[1]+=-u[0,coun_t+1]*a[1]
        r[-2]+=-u[-1,coun_t+1]*c[-2]
        return tridiag_alg(a[1:-1],b[1:-1],c[1:-1],r[1:-1])

    t=np.linspace(0,tmax,nt+1)
    x=np.linspace(0,xmax,nx+1)

    dt=tmax/nt; dx=xmax/nx
    u=np.zeros((nx+1,nt+1))
    u[:,0]=init(x)
    u[0,:]=bound(t)[0]
    u[-1,:]=bound(t)[1]

    for count_t in range(nt):
        u[1:-1,count_t+1]=solve_layer(u,count_t)
    return t,x,u

Nx=100
K=2000
kdif=2
tmax=0.1
xmax=2

t,x ,result = parab_eq(K, Nx,init,bound)

f=plt.figure()
ax=plt.axes(xlim=(0,xmax),ylim=(-2,5))

tex = 'Numerical solution for: $\\frac{\\partial^2{u}}{\\partial{t^2}}=k\\frac{\\partial^2{u}}{\\partial{x^2}}$'
plt.title(tex, fontsize=20, color='black')

ax.grid()
plt.ylabel('u')
plt.xlabel('x')
line,=plt.plot([],[],'k')
def init():
    line.set_data([],[])
    return line
def update(frame):
    line.set_data(x,result[:,frame])
    return line

ani = FuncAnimation(f,update, init_func=init,frames=range(0,K,2), interval=100,repeat=True)
plt.show()

