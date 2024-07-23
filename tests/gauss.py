import time
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools as tl

a= [[0,   3, -1,   3],
    [0,  -2,  1,   4],
    [1, 0.6,  2,   1],
    [0, 3.6,  2, 0.5]]

b=[0,1,2,3]

a=np.random.uniform(-10, 10, [200, 200])
b=np.random.uniform(-10, 10, 200)
'''fa=open('outMa.txt','w')
fb=open('outMb.txt','w')
np.savetxt(fa,a)
np.savetxt(fb,b)
fa.close()
fb.close()'''
time._startTime = time.time()
h=tl.lingauss(a,b)
print('Elapsed time my: ', time.time() - time._startTime)
time._startTime = time.time()
x=np.linalg.solve(a, b)
print('Elapsed time builin: ', time.time() - time._startTime)
print(h)
print(x)

