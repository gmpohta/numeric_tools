import matplotlib.pylab as plt
import numpy as np
A=820

def fun(x):
    return (1-A**2/(1+A**2))*np.exp(-A*x)+A/(1+A**2)*(np.sin(x)+A*np.cos(x))

def r_fun(x,y):
    return A*np.cos(x)-A*y

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

def CROS(lim, init, N, r_fun):
    def der_fun(ii):
        h=1e-5
        return (r_fun(x[ii-1],u[ii-1]*(1+h))-r_fun(x[ii-1],u[ii-1]*(1-h)))/(2*h*u[ii-1])

    x=np.linspace(lim[0],lim[1],N+1)
    dx=(lim[1]-lim[0])/N
    u=np.zeros(N+1)
    u[0]=init
    a=np.complex128('1+j')/2.0
    for ii in range(1,N+1):
        w=r_fun(x[ii-1]+dx/2,u[ii-1])/(1-a*der_fun(ii)*dx)
        u[ii]=u[ii-1]+np.real(w)*dx
    return x,u

N1=200
x,u=CROS([0,np.pi],1.0,N1,r_fun)
plt.plot(x,u,'k')
plt.plot(x,fun(x),'r--',label='analytical')

r=2
x1,u1=CROS([0,np.pi],1.0,N1,r_fun)
x2,u2=CROS([0,np.pi],1.0,N1*r,r_fun)
x3,u3=CROS([0,np.pi],1.0,N1*r*r,r_fun)

R_real=np.max(abs(u3-fun(x3)))
p=np.log((u2[::2]-u1)/(u3[::4]-u2[::2]))/np.log(r)
p=p[~np.isnan(p)]
p_min=np.min(p)
R_theor=np.max(abs(u3[::2]-u2)/(r**p_min-1))
print(p_min)
print('treor error = ', R_theor)
print('real error = ', R_real)
plt.legend()
plt.show()