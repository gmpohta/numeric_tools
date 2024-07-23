import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def an_fun(xg,tg):
    return np.exp(-kdif*np.pi**2/xmax**2*tg)*np.sin(np.pi*xg/xmax)

kdif=2
tmax=1.5
xmax=2
def init(x):
    return np.sin(np.pi*x/xmax)
def bound(t):
    return[np.zeros(t.shape),np.zeros(t.shape)]

def parab_eq(nt,nx,init,bound):
    def tridiag_alg(a,b,c,r):
        #a first element not used!!
        #c end element not used!!
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
    #nx,nt - number of interval
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
    return np.meshgrid(t,x),u

Nx=20
Nt=21

xr,result = parab_eq(Nx, Nt,init,bound)
x, t = xr

fig=plt.figure()
axes = fig.add_subplot(projection='3d')
print(x.shape,result.shape)
axes.plot_wireframe(x, t, result, color="black")
axes.plot_wireframe(x, t, an_fun(t,x), color="red")

r=2
xr,u1=parab_eq(Nx, Nt,init,bound)
x1, t1 = xr
xr,u2=parab_eq(Nx*r, Nt*r,init,bound)
x2, t2 = xr
xr,u3=parab_eq(Nx*r**2, Nt*r**2,init,bound)
x3, t3 = xr

R_real=np.max(abs(u3-an_fun(t3,x3)))
p=np.log((u2[::2,::2]-u1)/(u3[::4,::4]-u2[::2,::2]))/np.log(r)

print(p[1,:])
p=p[~np.isnan(p)]
p_min=np.min(p)
R_theor=np.max(abs(u3[::2,::2]-u2)/(r**p_min-1))

print('treor=',R_theor,R_real)
plt.legend()
plt.show()