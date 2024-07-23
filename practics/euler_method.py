import matplotlib.pylab as plt
import numpy as np

def fun(x):
    return (3-2*np.cos(x))**0.5

def r_fun(x,y):
    return np.sin(x)/y

def Euler(lim,init,N,r_fun):
    x=np.linspace(lim[0],lim[1],N+1)
    dx=(lim[1]-lim[0])/N
    u=np.zeros(N+1)
    u[0]=init
    for ii in range(1,N+1):
        u_tmp=u[ii-1]+r_fun(x[ii-1],u[ii-1])*dx/2
        u[ii]=u[ii-1]+r_fun(x[ii-1]+dx/2,u_tmp)*dx
    return x,u

N1=100
x,u=Euler([0,np.pi],1.0,N1,r_fun)
plt.plot(x,u)
plt.plot(x,fun(x))

r=2
x1,u1=Euler([0,np.pi],1.0,N1,r_fun)
x2,u2=Euler([0,np.pi],1.0,N1*r,r_fun)

R_real=np.max(abs(u2-fun(x2)))
R_theor=np.max(abs(u2[::2]-u1)/(r**2-1))
print(R_theor,R_real)
plt.show()