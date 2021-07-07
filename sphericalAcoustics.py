import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmath
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from scipy.spatial import ConvexHull
from scipy.io import loadmat, savemat


#---------------------DEFINING FUNCTIONS----------------------------------------
#  Convert cartesian coordinates to spherical coordinates
def cart2sph(x):
    '''
    Converts a 'x' matrix with size Lx3 in Cartesian coordinates to 3 vectors of
    size 'L' in spherical coordinates. The first vector returned is the azimuth
    angle, the second is the elevation, and the third is the radius.
    The range of azimuth angle is [-pi, pi> and the elevation is [-pi/2, pi/2].

    Parameters
    ----------
    x : array_like
        Lx3 input matrix.

    returns
    -------
    theta : ndarray
            Lx1 vector, azimuthal angle with range [-pi, pi>.
    phi :   ndarray
            Lx1 vector, elevation angle with range [-pi/2, pi/2].
    rho :   ndarray
            Lx1 vector, radius value.

    '''
    xx,xy,xz = np.split(x,3, axis=1)    #split the columns
    # Transforming to cartesians coordinates
    rho=np.sqrt(xx**2+xy**2+xz**2)
    phi = np.arctan2(xz,np.sqrt(xx**2+xy**2))
    theta = np.arctan2(xy,xx)
    return theta, phi, rho


#  Normalized Spherical Harmonics
def sphericalHarmonic(n,m,theta,phi):

    absm = np.abs(m)
    if m == 0:
        norm = np.sqrt((2*n+1)/(np.pi*4))
        f1 = 1
        legMatrix = sp.lpmv(m, n, np.sin(phi))
    elif m > 0:
        norm = (-1)**absm*np.sqrt(((2*n+1)/(2*np.pi))*(np.math.factorial(n-m)/np.math.factorial(n+m)))
        f1 = np.cos(m*theta)
        legMatrix = sp.lpmv(m, n, np.sin(phi))
    else:
        norm = (-1)**absm*np.sqrt(((2*n+1)/(2*np.pi))*(np.math.factorial(n-np.abs(m))/np.math.factorial(n+np.abs(m))))
        f1 = np.sin(np.abs(m)*theta)
        legMatrix = sp.lpmv(np.abs(m), n, np.sin(phi))

    Y = norm*legMatrix*f1
    return Y


#  load the values of the uniform distribution on the sphere
def icoDistribution(icoEdgeDiv, r = 1):
    filName = 'icolr'+ str(icoEdgeDiv)+ '.mat'
    distribution = loadmat('matFiles/icosphere/'+ filName)
    verts = distribution['x']*r

    # Calculating L, N and load weights
    L = np.shape(verts)[0]      #Number of vertices in the icosphere
    N = int(np.floor(np.sqrt(L)-1))  #Maximun order
    weights = distribution['weight']
    return weights, verts, L, N


#  Spherical harmonic function of degree 'n' and order 'm'
def baseFunction(n,m,x):
    theta, phi, _ = cart2sph(x)
    L = np.shape(x)[0]
    Y = sphericalHarmonic(n, m, theta, phi) #calculating the spherical harmonics
    Y = np.reshape(Y,(L,1))         #we order to get a column with L elements
    return Y


#  Function for build the matrix
def sphericalHarmonicsMatrix(N,x):
    theta, phi, rho = cart2sph(x)
    L = np.shape(x)[0]
    Ynm = np.zeros((L, (N+1)**2))
    for n in range(0,N+1):     #finding the values of m and n
        for m in range(-n,n+1):
            q = n**2 + n + m
            Ynm[:,q:(q+1)] = sphericalHarmonic(n, m, theta, phi)
    return Ynm


#  Pseudo inverse of Ynm using SVD Regularization
def pinvReg(Ynm,regLambda):
    U,S,Vt = np.linalg.svd(Ynm,full_matrices=False)   #SVD decomposition
    S2 = np.absolute(S)**2              #Absolute value of sigma squared

    # Regularized inverse of Sigma Matrix
    Sreginv = np.diag((S2/(S2+regLambda**2))*(1/S))

    V = np.transpose(Vt)
    Ut = np.transpose(U)
    # Regularized pseudoinverse of Y matrix
    Yreginv = np.dot(V,np.dot(Sreginv,Ut))
    return Yreginv


#  Spherical Fourier transform (SFT)
def sphericalFourierTransform(signal, x, N, regLambda=0):
    L = np.shape(x)[0]
    Ynm = sphericalHarmonicsMatrix(N, x)
    Yregpinv = pinvReg(Ynm,regLambda)
    Snm = np.dot(Yregpinv,signal)
    return Snm


# Plot SFT coefficients
def plotSnm(Snm, dBscale = False):
    N = int(np.sqrt(np.shape(Snm)[0])-1)
    SnmPlot = Snm
    if dBscale==True:
        SnmPlot = 20*np.log10(np.absolute(Snm))
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(0,(N+1)**2,1)
    ax.plot(x,SnmPlot.flatten(),'o--')
    plt.xticks(np.floor(x**(0.5))**2+np.floor(x**(0.5)))
    ticks_x = ticker.FuncFormatter(lambda x,pos:'{:.0f}'.format(
                    np.floor((np.floor(x**(0.5))**2+np.floor(x**(0.5)))**(0.5))))
    ax.xaxis.set_major_formatter(ticks_x)
    plt.xticks(fontsize=6)
    n = np.arange(0,N+1,1)            #arrangement with the values of N
    lines = n**2-0.5                   #square and subtract 0.5 to match the lines
    for i in lines:                    #Plot the lines
        ax.axvline(i, color='red',ls='dotted')
    ax.set_xlabel('n: Order')
    if dBscale==True:
        ax.set_ylabel('$20Log(|Snm|)$',size=12)
    if dBscale==False:
        ax.set_ylabel('Snm')
    ax.set_title('Snm Values',size=15,style='italic')
    return fig,ax


#  InverseSphericalFourierTransform for an arbitrary distribution
def inverseSphericalFourierTransform(Snm,y):
    #--New Ynm construction
    N = int(np.sqrt(np.shape(Snm)[0])-1)     # N :maximum order of Snm
    Ynm2 = sphericalHarmonicsMatrix(N, y)
    #interpolated points
    Sb = np.dot(Ynm2,Snm)
    return Sb


#  Interpolation function
def SFTinterpolation(signal, N, x, y, regLambda = 0):
    Snm = sphericalFourierTransform(signal, x, N, regLambda = regLambda)
    Sb = inverseSphericalFourierTransform(Snm, y)
    return Sb


#  Function to calculate error
def error(signal,target):
    error = (np.sqrt(((signal - target) ** 2).mean()))/(np.sqrt(((target)** 2).mean()))
    errordB = 20*np.log10(np.abs(error))
    return float(errordB)
