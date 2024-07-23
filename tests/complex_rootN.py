
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

def test(x):
    return x**3+1

x=np.linspace(0,5,100)
y=test(x)

xout,h=tl.complex_rootN(np.complex128('1'),test,1)
xi=xout[-1]
print(xi)
print(h)

