import sphericalAcoustics as sac
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.io import loadmat, savemat, wavfile
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator

pi = np.pi

# Loading a icosphere distribution
def icosphereDist(subdiv, r=1):
    x = loadmat('matFiles/icosphere/icolr'+str(subdiv)+'.mat')['x']
    x = r*x
    L = np.shape(x)[0]
    N = int(np.floor(np.sqrt(L)-1))
    return x, L, N

# Creating a equiangular distribution (Driscoll)
def equiangularDist1(N, r=1):
    theta = np.linspace(-pi, pi, 2*N+2, endpoint=False )
    phi = np.linspace(-pi/2, pi/2, 2*N+2, endpoint=False)
    theta, phi = np.meshgrid(theta,phi)
    X = np.cos(phi.flatten())*np.cos(theta.flatten())
    Y = np.cos(phi.flatten())*np.sin(theta.flatten())
    Z = np.sin(phi.flatten())
    x = r*np.array([X,Y,Z]).T
    L = np.shape(x)[0]
    return x, L

# Creating a equiangular distribution
def equiangularDist2(numTheta, numPhi, r=1):
    theta = np.linspace(-pi,pi, numTheta, endpoint = False)
    phi = np.linspace(-pi/2, pi/2, numPhi)
    theta,phi = np.meshgrid(theta,phi)
    X = np.cos(phi.flatten())*np.cos(theta.flatten())
    Y = np.cos(phi.flatten())*np.sin(theta.flatten())
    Z = np.sin(phi.flatten())
    x = r*np.array([X,Y,Z]).T
    L = np.shape(x)[0]
    return x, L

# Creating a aleatory distribution
def randDist(L, r=1):
    x = np.random.randn(L,3)
    norm = np.linalg.norm(x, axis = 1).reshape(L,1)
    x = (x/norm)*r
    return x

# Distribution limits
def distLimits(x,axis,lowerlim=-1,upperlim=1):
    if axis in ['Z', 'z']:
        filterArrayU = x[:, 2]<= upperlim
        filterArrayL = x[:, 2]>= lowerlim
    elif axis in ['X','x']:
        filterArrayU = x[:, 0]<= upperlim
        filterArrayL = x[:, 0]>= lowerlim
    elif axis in ['Y','y']:
        filterArrayU = x[:, 1]<= upperlim
        filterArrayL = x[:, 1]>= lowerlim
    else:
        return print("Error: Axis must be a character : 'X', 'Y', or 'Z'." )

    filterArray = filterArrayL*filterArrayU
    x = x[filterArray]
    L = np.shape(x)[0]
    return x, L

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Distribution plotting in the sphere
def plotDist(x):
    PHI, THETA = np.mgrid[0:2*pi:50j, 0:pi:50j]
    L = np.shape(x)[0]
    p = x[0,:].flatten()
    R = 0.94*sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize = (7,7))
    fig = plt.figure(figsize=[7,7])
    ax = fig.gca(projection='3d',adjustable = 'box')
    ax.plot_surface(X, Y, Z, alpha = 1, rstride =1, cstride = 1, color = 'c')
    ax.scatter(x[:,0],x[:,1],x[:,2], c='k', s=5)
    #ax.set_title('L = '+str(L))
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    return fig

def aleatorySignals(x,y,N,maxValue=1,SNR=90):
    Snm = np.random.normal(0,0.1,size=((N+1)**2,1))*maxValue
    Sa = sac.inverseSphericalFourierTransform(Snm,x)
    Sa,_ = addNoise(Sa,SNR)
    St = sac.inverseSphericalFourierTransform(Snm,y)
    St,_ = addNoise(St,SNR)
    return Sa, St, Snm


# Adding noise to signal
def addNoise(signal,SNR):
    noise = np.random.normal(0,0.1,np.shape(signal)[0])
    noise = np.reshape(noise,(np.shape(signal)[0],1))
    snr=10**(SNR/20)
    RMSsignal=float(np.sqrt((signal**2).mean()))
    RMSnoise=float(np.sqrt((noise**2).mean()))
    k = RMSsignal/(RMSnoise*snr)
    newSignal = signal + k*noise
    if SNR == 0:
        return signal, k*noise
    else:
        return newSignal, k*noise


# Global error calculation
def plotGlobalError(signal,target,N,x,y,minLambda,maxLambda,resolution):
    i = 0
    L1 = np.shape(x)[0]
    L2 = np.shape(y)[0]
    arrayLambda = np.logspace(start =minLambda,stop = maxLambda,num = resolution)
    error = np.zeros(resolution)
    for regLambda in arrayLambda:
        Sb = sac.SFTinterpolation(signal, N, x, y, regLambda = regLambda)
        error[i] = sac.error(Sb,target)
        i += 1

    fig, ax = plt.subplots(1,1)
    ax.plot(arrayLambda,error,'b.', label = '')
    ax.set_title('Globar Error, N = '+str(N)+ ', $L_{ini}$ = '+str(L1)+' $,L_{fin} = $'+str(L2), fontsize =9)
    ax.set_ylabel('20Log(|Error|)')
    ax.set_xlabel('Lambda')
    ax.set_xscale('log')
    return fig


# Plotting singular values
def plotSigma(N,x):
    Y = sac.sphericalHarmonicsMatrix(N, x)
    U,S,Vt = np.linalg.svd(Y,full_matrices=False)
    fig, ax = plt.subplots(1,1, figsize=(10,1))
    yy = np.ones(np.shape(S)[0])
    ax.plot(S,yy,'r.',ms=5)
    ax.set_xlabel('Sigma')
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='both')
    ax.set_yticks([])
    fig.tight_layout()
    #ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.grid(b=True, which='minor',axis='both')
    return fig


def scatterPlotFunction(signal, x, polarForm = True):
    theta, phi, rho = sac.cart2sph(x)
    signal = signal.flatten().real #Real part of signal
    signalAbs=np.abs(signal)       #Absolute value

    #-Convert to cartesian coordinates
    if polarForm==True:
        X = signalAbs*np.cos(phi.flatten())*np.cos(theta.flatten())
        Y = signalAbs*np.cos(phi.flatten())*np.sin(theta.flatten())
        Z = signalAbs*np.sin(phi.flatten())

    if polarForm==False:
        X = np.cos(phi.flatten())*np.cos(theta.flatten())
        Y = np.cos(phi.flatten())*np.sin(theta.flatten())
        Z = np.sin(phi.flatten())

    fig = plt.figure(figsize = (8,5))
    ax = plt.axes(projection='3d')
    ax.scatter(X,Y,Z,c=signal,s=10,cmap='jet')
    ax.view_init(elev=15, azim=30)
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    ax.set_xlabel('$X$',fontsize=15)
    ax.set_ylabel('$Y$',fontsize=15)
    ax.set_zlabel('$Z$',fontsize=15)

    #--Adding colorbar-------------------------------------------------------------
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(signal)
    fig.colorbar(m, shrink=0.5,label='Presión sonora')
    ax.tick_params(labelsize = 7)
    return fig, ax


#  Function to plot sound presure in a spherical distribution.
def plotFunction(signal, x, title='', polarForm = True):
    theta, phi, rho = cart2sph(x)
    signal = signal.flatten().real #Real part of signal
    signalAbs=np.abs(signal)       #Absolute value

    #-Convert to cartesian coordinates
    if polarForm==True:
        X = signalAbs*np.cos(phi.flatten())*np.cos(theta.flatten())
        Y = signalAbs*np.cos(phi.flatten())*np.sin(theta.flatten())
        Z = signalAbs*np.sin(phi.flatten())

    if polarForm==False:
        X = np.cos(phi.flatten())*np.cos(theta.flatten())
        Y = np.cos(phi.flatten())*np.sin(theta.flatten())
        Z = np.sin(phi.flatten())
    #-triangulation and plotting
    hull = ConvexHull(x)
    #--calculating matrix "colors" for the color of the surfaces
    colors = np.mean(signal[hull.simplices], axis=1)

    fig = plt.figure()
    ax = fig.gca(projection='3d',adjustable='box')
    ax.set_title(title,fontsize = 15,style='italic')

    #--Plotting using trisurf---------------------------------------------------
    s = ax.plot_trisurf(X,Y,Z, triangles=hull.simplices,
                        cmap='jet', alpha=1,edgecolor='none')
    s.set_array(colors)  #Colores de superficies en funcion del valor de "colors"
    ax.set_xlabel('$X$',fontsize=15)
    ax.set_ylabel('$Y$',fontsize=15)
    ax.set_zlabel('$Z$',fontsize=15)

    #--Adding colorbar
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(signal)
    fig.colorbar(m, shrink=0.5,label='Presión sonora')
    ax.tick_params(labelsize = 7)
    ax.view_init(elev=15, azim=30)
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    return fig, ax
