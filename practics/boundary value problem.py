import matplotlib.pylab as plt
import numpy as np

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

def fun(x):
    c1=(3-1/3)/(np.exp(3)-1)
    c2=1-c1
    return c1*np.exp(3*x)-2/3*x+c2

def problem(lim,init,N):
    x=np.linspace(lim[0],lim[1],N+1)
    dx=(lim[1]-lim[0])/N
    u=np.zeros(N+1)
    u[0]=init[0]
    u[-1]=init[1]
    a=np.ones(N-1)*(1/dx**2+3/2/dx)
    b=np.ones(N-1)*(-2/dx**2)
    c=np.ones(N-1)*(1/dx**2-3/2/dx)
    r=np.ones(N-1)*2
    r[0]+=-u[0]*a[0]
    r[-1]+=-u[-1]*c[-1]
    u[1:-1]=tridiag_alg(a,b,c,r)
    return x,u

N1=2000
x,u=problem([0,1],[1,3],N1)
plt.plot(x,u,'k')
plt.plot(x,fun(x),'r--',label='analytical')

r=2
x1,u1=problem([0,1],[1,3],N1)
x2,u2=problem([0,1],[1,3],N1*r)
x3,u3=problem([0,1],[1,3],N1*r**2)

R_real=np.max(abs(u3-fun(x3)))
p=np.log((u2[::2]-u1)/(u3[::4]-u2[::2]))/np.log(r)
p=p[~np.isnan(p)]
p_min=np.min(p)
R_theor=np.max(abs(u3[::2]-u2)/(r**p_min-1))
print(p_min)
print('treor=',R_theor,R_real)
plt.legend()
plt.show()