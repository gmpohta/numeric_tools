
import numpy as np
import matplotlib.pylab as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

wi=np.linspace(0,2*np.pi,500)
yi=np.sin(wi)+np.random.uniform(-0.1,0.1,500)

p=tl.polyapr(wi,yi,50)
w=np.linspace(0,2*np.pi,500)
y=tl.polyval(p,w)

plt.plot(wi,yi,'or')
plt.plot(w,y)
plt.show()
