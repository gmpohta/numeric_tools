import numpy as np
import matplotlib.pylab as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

x=np.linspace(0,3,50)
y=np.sin(x)+np.random.random(50)*0.1

sp=tl.Spline(x,y)
xi=np.linspace(0,3,500)
yi=sp.calcspln(xi)
#print(xi,yi)

plt.plot(x,y,'o')
plt.plot(xi,yi)
yder=sp.splder().calcspln(xi)
plt.plot(xi,yder,'r')
plt.show()