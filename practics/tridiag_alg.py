import matplotlib.pylab as plt
import numpy as np
from scipy.integrate import quad

#b - main diagonal
#a - lower diaganal
#c - upper diaganal
def tridiag_alg(a,b,c,r):
    n=len(b)
    alf=np.zeros(n)
    beta=np.zeros(n)
    out=np.zeros(n)
    alf[0]=-c[0]/b[0]
    beta[0]=r[0]/b[0]
    for ii in range(1,n):
        alf[ii]=-c[ii]/(b[ii]+alf[ii-1]*a[ii])
        beta[ii]=(r[ii]-a[ii]*beta[ii-1])/(b[ii]+alf[ii-1]*a[ii])
    out[-1]=beta[-1]
    for ii in reversed(range(n-1)):
        out[ii]=alf[ii]*out[ii+1]+beta[ii]
    return out

M=np.array([[1,2,0,0,0,0],
            [3,4,5,0,0,0],
            [0,6,7,8,0,0],
            [0,0,9,10,11,0],
            [0,0,0,12,13,14],
            [0,0,0,0,15,16]])
a=np.hstack(([999],np.diag(M,-1)))
b=np.diag(M,0)
c=np.hstack((np.diag(M,1),[666]))
r=np.array([1,2,3,4,5,6])
x=tridiag_alg(a,b,c,r)

print(x)
print(np.linalg.solve(M,r))
