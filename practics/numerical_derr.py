import matplotlib.pylab as plt
import numpy as np
from scipy.integrate import quad

def fun(x):
    return np.sin(x)

def an_der(x):
    return np.cos(x)
def an_der2(x):
    return -np.sin(x)
def an_der3(x):
    return -np.cos(x)
def an_der4(x):
    return np.sin(x)

def num_der(x,h):
    return (fun(x+h)-fun(x-h))/2/h
def num_der2(x,h):
    return (fun(x+h)-2*fun(x)+fun(x-h))/h**2
def num_der3(x,h):
    return (fun(x+3*h/2)-3*fun(x+h/2)+3*fun(x-h/2)-fun(x-3*h/2))/h**3
def num_der4(x,h):
    return (fun(x+2*h)-4*fun(x+h)+6*fun(x)-4*fun(x-h)+fun(x-2*h))/h**4

plt.figure()
h=10**np.linspace(-1,-15,45)
plt.plot(h,abs(num_der(0.1, h)-an_der(0.1)),label='1')
plt.plot(h,abs(num_der3(0.1, h)-an_der3(0.1)),label='3')
plt.plot(h,abs(num_der4(0.1, h)-an_der4(0.1)),label='4')
plt.plot(h,abs(num_der2(0.1, h)-an_der2(0.1)),label='2')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()