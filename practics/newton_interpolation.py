import matplotlib.pylab as plt
import numpy as np
from scipy.integrate import quad

def _poly_newton_coefficient(x,y):
    n=len(x)
    c=np.ones((n,n))*np.nan
    c[:,0]=y
    c_out=[y[0]]
    for jj in range(1,n):
        for ii in range(jj,n):
            c[ii,jj]=(c[ii,jj-1]-c[ii-1,jj-1])/(x[ii]-x[ii-jj])
            if ii==jj:
                c_out.append(c[jj,jj])
    return c_out

def newton_polynomial(x_d,y_d,x):
    n=len(x_d) # Degree of polynomial

    x_data=x_d.copy()
    y_data=y_d.copy()

    x_t=x_data[int(n/2)]
    x_data[int(n / 2)]=x_data[-1]
    x_data[-1]=x_t
    y_t = y_data[int(n / 2)]
    y_data[int(n / 2)] = y_data[-1]
    y_data[-1] = y_t

    c = _poly_newton_coefficient(x_data, y_data)
    tmp=1
    p=0
    for k in range(0,n-1):
        p+=c[k]*tmp
        tmp*=(x-x_data[k])
    tmp *= (x - x_data[-1])
    err=c[-1]*tmp
    return p,err

def fun(x):
    return np.sin(x)

n=12
x=np.sort(3*np.random.random(n)-1)
y=fun(x)

xr=np.linspace(x[0],x[-1],500)
yr=fun(xr)
plt.plot(x,y,'or')
plt.plot(xr,yr,'k')

int_f,err=newton_polynomial(x,y,xr)
plt.plot(xr,int_f,'b--')

plt.figure()
plt.plot(xr,abs(int_f-yr),label='real')
plt.plot(xr,abs(err),label='theor')
plt.plot(xr,np.max(abs(err))*np.ones(len(xr)),label='$theor_{max}$')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Error')
plt.legend()
plt.show()
