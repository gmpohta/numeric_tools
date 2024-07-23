import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

a=np.array([999,5,1])
b=np.array([2,4,-3])
c=np.array([-1,2,666])
r=np.array([3,6,2])

x=tl.tridiag_alg(a,b,c,r)

print(x)
print(1.49,-0.02,-0.68)