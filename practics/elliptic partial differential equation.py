import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def an_fun(xg,yg):
    k_end=100
    out=0
    for k in range(k_end):
        out+=np.sinh((2*k+1)*(xmax-xg)*np.pi/ymax)*np.sin((2*k+1)*np.pi*yg/ymax)/(2*k+1)/np.sinh((2*k+1)*np.pi*xmax/ymax)
    return 4*V/np.pi*out

def elliptic_eq(nt,nx,ny):
    def tridiag_alg(a,b,c,r):
        #a first element not used!!
        #c end element not used!!
        #b - main diagonal
        #a - lower diaganal
        #c - upper diaganal
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
    #nx,nt - number of interval
    def solve_layer(u):
        tmp_v=np.zeros((nx+1,ny+1))
        out=np.zeros((nx+1,ny+1))
        for jj in range(1,ny):
            a=-dt/2/dx/dx*np.ones(nx+1)
            b=(1+dt/dx/dx)*np.ones(nx+1)
            c=-dt/2/dx/dx*np.ones(nx+1)
            r=np.zeros(nx+1)
            for ii in range(1,nx):
                op_x=(u[ii-1,jj]-2*u[ii,jj]+u[ii+1,jj])/dx/dx
                op_y=(u[ii,jj-1]-2*u[ii,jj]+u[ii,jj+1])/dy/dy
                r[ii]= op_x+op_y
            r[1]+=-u[0,jj]*a[1]
            r[-2]+=-u[-1,jj]*c[-2]
            tmp_v[1:-1,jj]=tridiag_alg(a[1:-1],b[1:-1],c[1:-1],r[1:-1])

        for ii in range(1,nx):
            a=-dt/2/dy/dy*np.ones(ny+1)
            b=(1+dt/dy/dy)*np.ones(ny+1)
            c=-dt/2/dy/dy*np.ones(ny+1)
            r=np.zeros(ny+1)
            for jj in range(1,ny):
                op_y=(u[ii,jj-1]-2*u[ii,jj]+u[ii,jj+1])/dy/dy
                r[jj]=tmp_v[ii,jj]*dt+u[ii,jj]-dt/2*op_y
            r[1]+=-u[ii,0]*a[1]
            r[-2]+=-u[ii,-1]*c[-2]
            out[ii,1:-1]=tridiag_alg(a[1:-1],b[1:-1],c[1:-1],r[1:-1])
        return out[1:-1,1:-1]

    t=np.linspace(0,tmax,nt+1)
    x=np.linspace(0,xmax,nx+1)
    y=np.linspace(0,ymax,ny+1)

    dt=tmax/nt; dx=xmax/nx; dy=ymax/ny
    u=np.zeros((nx+1,ny+1,nt+1))
    u[:,:,0]=V*np.ones((nx+1,ny+1))#(1-x/xmax)*V*np.ones((nx+1,ny+1))
    u[:,0,:]=0.0
    u[:,-1,:]=0.0
    u[0,:,:]=V
    u[-1,:,:]=0.0

    for count_t in range(nt):
        u[1:-1,1:-1,count_t+1]=solve_layer(u[:,:,count_t])
    return t,x,y,u

tmax=1.5
xmax=2
ymax=2.5
V=3
nt=100
ny=101
nx=102

t,x,y,result = elliptic_eq(nt,nx,ny)

fig=plt.figure()
axes = fig.add_subplot(projection='3d')
tg,xg=np.meshgrid(t,x)
print(xg.shape,result.shape)
axes.plot_wireframe(xg, tg, result[:,50,:], color="black")

fig=plt.figure()
yg,xg=np.meshgrid(y,x)
axes = fig.add_subplot(projection='3d')
axes.plot_wireframe(xg, yg, result[:,:,-1], color="black")
axes.plot_wireframe(xg,yg,an_fun(xg,yg), color="red")
fig=plt.figure()
plt.contour(yg,xg,result[:,:,-1],100)

plt.show()