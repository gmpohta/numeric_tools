import matplotlib.pylab as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

#Система (x-4)^2+y^2=4
#         x^2+y^2=4
#
#

def f(x):
    return (x[0]-4)**2+x[1]**2-4, x[0]**2+x[1]**2-4#,-x[2]+5*x[0]**2,x[3]-1

x,h=tl.rootNSystem([1,1],f,1)
print(x[-1])
print("*************")
print(h)
