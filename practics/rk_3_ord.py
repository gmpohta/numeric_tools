import matplotlib.pylab as plt
import numpy as np

def fun(x):
    return (3-2*np.cos(x))**0.5

def r_fun(x,y):
    return np.sin(x)/y

def RK3(lim,init,N,r_fun):
    x=np.linspace(lim[0],lim[1],N+1)
    dx=(lim[1]-lim[0])/N
    u=np.zeros(N+1)
    u[0]=init
    for ii in range(1,N+1):
        w1=r_fun(x[ii-1],u[ii-1])
        w2=r_fun(x[ii-1]+dx/2,u[ii-1]+dx/2*w1)
        w3=r_fun(x[ii-1]+3/4*dx,u[ii-1]+3*dx/4*w2)
        u[ii]=u[ii-1]+dx/9*(2*w1+3*w2+4*w3)
    return x,u

N1=100
x,u=RK3([0,np.pi],1.0,N1,r_fun)
plt.plot(x,u,'k')
plt.plot(x,fun(x),'r--')

r=2
x1,u1=RK3([0,np.pi],1.0,N1,r_fun)
x2,u2=RK3([0,np.pi],1.0,N1*r,r_fun)
x3,u3=RK3([0,np.pi],1.0,N1*r*r,r_fun)

R_real=np.max(abs(u3-fun(x3)))
p=np.log((u2[::2]-u1)/(u3[::4]-u2[::2]))/np.log(r)
p=p[~np.isnan(p)]
p_min=np.min(p)
R_theor=np.max(abs(u3[::2]-u2)/(r**p_min-1))
print(p_min)
print(R_theor,R_real)
plt.show()