import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from scipy.io import loadmat, savemat


#---------------------DEFINING FUNCTIONS----------------------------------------
#  Convert cartesian coordinates to spherical coordinates
def cart2sph(x):
    '''
    Converts a 'x' matrix with size Lx3 in Cartesian coordinates to 3 vectors of
    size 'L' in spherical coordinates.

    Parameters
    ----------
    x: array_like.
        Lx3 array; points distribution on the sphere, given in Cartesian
        coordinates.

    returns
    -------
    theta : ndarray
            Lx1 vector; azimuthal coordinate with range [-pi, pi>.
    phi :   ndarray
            Lx1 vector; elevation coordinate range [-pi/2, pi/2].
    rho :   ndarray
            Lx1 vector; radius value.

    '''
    xx, xy, xz = np.split(x, 3, axis=1)  # split the columns
    # Transforming to cartesians coordinates
    rho = np.sqrt(xx**2+xy**2+xz**2)
    phi = np.arctan2(xz, np.sqrt(xx**2+xy**2))

    theta = np.arctan2(xy, xx)
    return theta, phi, rho


#  Normalized Spherical Harmonics
def sphericalHarmonic(n, m, theta, phi):
    '''
    Calculates the spherical harmonic with order "n" and degree "m" for all
    values of theta and phi.

    Parameters:
    ----------
    n: Scalar.
        Order of harmonic(int); must have n >= 0.
    m: Scalar.
        Degree of harmonic(int); must have |m| <= n.
    theta: array_like
        Lx1 vector; azimuthal coordinate; must be in [-pi, pi>.
    phi: array_like
        Lx1 vector; elevation coordinate; must be in [-pi/2, pi/2].

    Returns:
    ------
    Y: ndarray.
        Lx1 vector; the harmonic with order "n" and degree "m" sampled at theta
        and phi.
    '''

    absm = np.abs(m)
    if m == 0:
        norm = np.sqrt((2*n+1)/(np.pi*4))
        f1 = 1
        legMatrix = sp.lpmv(m, n, np.sin(phi))
    elif m > 0:
        norm = (-1)**absm*np.sqrt(((2*n+1)/(2*np.pi)) *
                                  (np.math.factorial(n-m)/np.math.factorial(n+m)))
        f1 = np.cos(m*theta)
        legMatrix = sp.lpmv(m, n, np.sin(phi))
    else:
        norm = (-1)**absm*np.sqrt(((2*n+1)/(2*np.pi)) *
                                  (np.math.factorial(n-np.abs(m))/np.math.factorial(n+np.abs(m))))
        f1 = np.sin(np.abs(m)*theta)
        legMatrix = sp.lpmv(np.abs(m), n, np.sin(phi))

    Y = norm*legMatrix*f1
    return Y


#  Spherical harmonic function of degree 'n' and order 'm'
def baseFunction(n, m, x):
    '''
    Calculates a spherical harmonic in a 'x' distribution given in Cartesian
    coordinates.

    Parameters:
    ----------
    n: Scalar.
        Order of harmonic(int); must have n >= 0.
    m: Scalar.
        Degree of harmonic(int); must have |m| <= n.
    x: array_like.
        Lx3 array; points distribution on the sphere, given in Cartesian
        coordinates.

    Returns:
    ------
    Y: ndarray.
        Lx1 vector; the harmonic with order "n" and degree "m" in all points of
        'x'.
    '''

    theta, phi, _ = cart2sph(x)
    L = np.shape(x)[0]
    # calculating the spherical harmonics
    Y = sphericalHarmonic(n, m, theta, phi)
    Y = np.reshape(Y, (L, 1))  # we order to get a column with L elements
    return Y


#  Function for build the matrix
def sphericalHarmonicsMatrix(N, x):
    '''
    Calculates all spherical harmonics up to order N.

    Parameters:
    ----------
    N: Scalar.
        Max order to be calculated. All possibles values of degree 'm' are
         calculated.
    x: arral_like.
        Lx3 array; points distribution on the sphere, given in Cartesian
        coordinates.

    Returns:
    -------
    Ynm: ndarray.
        Lx(N+1)^2 array; Contains all spherical harmonics up to order N.
    '''

    theta, phi, rho = cart2sph(x)
    L = np.shape(x)[0]
    Ynm = np.zeros((L, (N+1)**2))
    for n in range(0, N+1):  # finding the values of m and n
        for m in range(-n, n+1):
            q = n**2 + n + m
            Ynm[:, q:(q+1)] = sphericalHarmonic(n, m, theta, phi)
    return Ynm


#  Pseudo inverse of Ynm using SVD Regularization
def pinvReg(Ynm, regLambda):
    '''
    Calculates pseudo-inverse of spherical harmonics matrix (Ynm).

    Parameters:
    ----------
    Ynm: array_like.
        Lx(N+1)^2 array; Contains all spherical harmonics up to order N.
    regLambda: Scalar.
        Regularization parameter (float); must be greater than 0.

    Returns:
    -------
    Yreginv: ndarray.
        Lx(N+1)^2 array; Regularized pseudo-inversa of spherical harmonics
        matrix (Ynm).
    '''

    U, S, Vt = np.linalg.svd(Ynm, full_matrices=False)  # SVD decomposition
    S2 = np.absolute(S)**2  # Absolute value of sigma squared

    # Regularized inverse of Sigma Matrix
    Sreginv = np.diag((S2/(S2+regLambda**2))*(1/S))
    V = np.transpose(Vt)
    Ut = np.transpose(U)
    # Regularized pseudoinverse of Y matrix
    Yreginv = np.dot(V, np.dot(Sreginv, Ut))
    return Yreginv


#  Regularized spherical Fourier transform (SFT)
def sphericalFourierTransform(signal, x, N, regLambda=0):
    '''
    Calculates the regularized spherical Fourier transform (SFT) to 'signal'
    in a 'x' distribution.

    Parameters:
    ----------
    signal: array_like.
        Lx1 vector; Initial sound pressure in spherical distribution 'x'.
    x: arral_like.
        Lx3 array; points distribution on the sphere, given in Cartesian
        coordinates.
    N: Scalar.
        Max order to be calculated. All possibles values of degree 'm' are
        calculated.
    regLambda: Scalar.
        Regularization parameter (float); must be greater than 0.

    Returns:
    -------
    Snm: ndarray.
        (N+1)^2x1 array; Regularized spherical Fourier transform coefficients.

    '''
    Ynm = sphericalHarmonicsMatrix(N, x)
    Yregpinv = pinvReg(Ynm, regLambda)
    Snm = np.dot(Yregpinv, signal)
    return Snm


#  InverseSphericalFourierTransform for an arbitrary distribution
def inverseSphericalFourierTransform(Snm, y):
    '''
    Calculates the inverse spherical Fourier transform of regularized SFT
    coefficients (Snm) in 'y' distribution on the sphere.

    Parameters:
    ----------
    Snm: array_like.
        (N+1)^2x1 array; Regularized spherical Fourier transform coefficients.
    y: array_like.
        Ix3 array; new points distribution on the sphere, given in Cartesian
        coordinates.

    Returns:
    -------
    Sb: ndarray.
        Ix1 array; Interpolated sound pressure in spherical distribution 'y'.
    '''

    #--New Ynm construction
    N = int(np.sqrt(np.shape(Snm)[0])-1)     # N :maximum order of Snm
    Ynm2 = sphericalHarmonicsMatrix(N, y)
    #interpolated points
    Sb = np.dot(Ynm2, Snm)
    return Sb


#  Interpolation function
def SFTinterpolation(signal, N, x, y, regLambda=0):
    '''
    Complete interpolation process of 'signal' with regularized SFT.

    Parameters:
    ----------
    signal: array_like.
        Lx1 vector; Initial sound pressure in spherical distribution 'x'.
    N: Scalar.
        Max order to be calculated. All possibles values of degree 'm' are
        calculated.
    x: arral_like.
        Lx3 array; initial points distribution on the sphere, given in Cartesian
        coordinates.
    y: array_like.
        Ix3 array; new points distribution on the sphere, given in Cartesian
        coordinates.
    regLambda: Scalar.
        Regularization parameter (float); must be greater than 0.

    Returns:
    -------
    Sb: ndarray.
        Ix1 array; Interpolated sound pressure in spherical distribution 'y'.

    '''

    Snm = sphericalFourierTransform(signal, x, N, regLambda=regLambda)
    Sb = inverseSphericalFourierTransform(Snm, y)
    return Sb


#  Function to calculate error
def error(interpolated, target):
    '''
    Calculates the interpolation error RMS between the 'interpolated' signal
    and the 'target' signal.

    Parameters:
    ----------
    interpolated: array_like.
        Ix1 array; Interpolated sound pressure in the sphere.
    target: array_like.
        Ix1 array; Target sound pressure in the sphere.
    Returns:
    -------
    errordB: Scalar.
        Interpolation error RMS (float) in dB.
    '''

    error = (np.sqrt(((interpolated - target) ** 2).mean())) / \
        (np.sqrt(((target) ** 2).mean()))
    errordB = float(20*np.log10(np.abs(error)))
    return errordB
