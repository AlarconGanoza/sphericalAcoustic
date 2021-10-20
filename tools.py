import sphericalAcoustics as sac
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.io import loadmat, savemat, wavfile
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator
from scipy.spatial import ConvexHull
import matplotlib.ticker as ticker

pi = np.pi


# Loading a icosphere distribution
def icosphereDist(icoDirectory, subdiv, r=1):
    '''
    Load the icosphere distribution. The icosphere is a distribution based on
    an icosahedron, whose edges have a subdivision 'subdiv'.

    Parameters:
    ----------
    icoDirectory: Str.
        Directory of previously calculated coordinate files.
    subdiv: Scalar.
        Edges subdivision of the icosahedron.
    r:  Scalar, optional.
        Radius of the sphere containing the icosahedron. Default r = 1.

    Returns:
    -------
    x: ndarray.
        Lx3 array; icosahedron distribution points, in Cartesian coordinates.
    L: scalar.
        Total number of points.
    '''

    x = loadmat(icoDirectory + str(subdiv) + '.mat')['x']
    x = r*x
    L = np.shape(x)[0]
    N = int(np.floor(np.sqrt(L)-1))
    return x, L


# Creating a equiangular distribution (Driscoll)
def equiangularDist1(N, r=1):
    '''
    Calculates distribution points based on the equiangular
    distribution proposed by Driscoll (1994, Computing Fourier Transforms
    and Convolutions on the 2-Sphere). Here, the maximun order N, defines the
    number of points.

    Parameters:
    ----------
    N:  Int.
        Max orden for Spherical Fourier Transform  (SFT).
    r:  Scalar, optional.
        Radius of the sphere containing the icosahedron. Default r = 1.

    Returns:
    -------
    x:  ndarray.
        Lx3 array; equiangular distribution points, in Cartesian coordinates.
    L: scalar.
        Total number of points.
    '''

    theta = np.linspace(-pi, pi, 2*N+2, endpoint=False)
    phi = np.linspace(-pi/2, pi/2, 2*N+2, endpoint=False)
    theta, phi = np.meshgrid(theta, phi)
    X = np.cos(phi.flatten())*np.cos(theta.flatten())
    Y = np.cos(phi.flatten())*np.sin(theta.flatten())
    Z = np.sin(phi.flatten())
    x = r*np.array([X, Y, Z]).T
    L = np.shape(x)[0]
    return x, L

# Creating a equiangular distribution


def equiangularDist2(numTheta, numPhi, r=1):
    '''
    Calculates distribution points of a regular equiangular distribution.

    Parameters:
    ----------
    numTheta:   int.
        Number of azimuthal angles. Must be > 0.
    numPhi: int.
        Number of elevation angles. Must be > 0.
    r:  Scalar, optional.
            Radius of the sphere. Default r = 1.

    Returns:
    -------
    x:  ndarray.
        Lx3 array; equiangular distribution points, in Cartesian coordinates.
    L: scalar.
        Total number of points.
    '''

    theta = np.linspace(-pi, pi, numTheta, endpoint=False)
    phi = np.linspace(-pi/2, pi/2, numPhi)
    theta, phi = np.meshgrid(theta, phi)
    X = np.cos(phi.flatten())*np.cos(theta.flatten())
    Y = np.cos(phi.flatten())*np.sin(theta.flatten())
    Z = np.sin(phi.flatten())
    x = r*np.array([X, Y, Z]).T
    L = np.shape(x)[0]
    return x, L

# Creating a aleatory distribution


def randDist(L, r=1):
    '''
    Generates a random distribution of points on the sphere.

    Parameters:
    ----------
    L:  int.
        Total number of points.
    r:  Scalar, optional.
        Radius of the sphere. Default r = 1.

    Returns:
    -------
    x: ndarray.
        Lx3 array; random distribution points, in Cartesian coordinates.
    '''

    x = np.random.randn(L, 3)
    norm = np.linalg.norm(x, axis=1).reshape(L, 1)
    x = (x/norm)*r
    return x

# Distribution limits


def distLimits(x, axis='z', lowerlim=-1, upperlim=1):
    '''
    Discard the points of the 'x' distribution, which are outside
    the range [lowerlim; upperlim] on the indicated axis.

    Parameters:
    ----------
    x: like_array.
        Lx3 array; distribution points, given in Cartesian coordinates.
    axis:   str, optional.
        Axis on which the limitation will be applied. Can be 'Z','z','X','x'
        'Y' or 'y'. Default axis = 'z'.
    lowerlim:   scalar.
        Lower limit on the indicated axis.
    upperlim:   scalar.
        Upper limit on the indicated axis.

    Returns:
    -------
    x: ndarray.
        Lx3 array; final points distribution, in Cartesian coordinates.
    L:  int.
        Total number of points.
    '''

    if axis in ['Z', 'z']:
        filterArrayU = x[:, 2] <= upperlim
        filterArrayL = x[:, 2] >= lowerlim
    elif axis in ['X', 'x']:
        filterArrayU = x[:, 0] <= upperlim
        filterArrayL = x[:, 0] >= lowerlim
    elif axis in ['Y', 'y']:
        filterArrayU = x[:, 1] <= upperlim
        filterArrayL = x[:, 1] >= lowerlim
    else:
        return print("Error: Axis must be a character : 'X', 'Y', or 'Z'.")

    filterArray = filterArrayL*filterArrayU
    x = x[filterArray]
    L = np.shape(x)[0]
    return x, L

# Plotting distribution in the sphere.


def plotDist(x):
    '''
    Plot a distribution of points on the sphere.
    Parameters:
    ----------
    x:  array_like.
        Lx3 array; points distribution, given in Cartesian coordinates.

    Returns:
    -------
    fig: figure.
        Figure that represent 'x' in the space.
    '''
    PHI, THETA = np.mgrid[0:2*pi:50j, 0:pi:50j]
    L = np.shape(x)[0]
    p = x[0, :].flatten()
    R = 0.94*sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize = (7,7))
    fig = plt.figure(figsize=[7, 7])
    ax = fig.gca(projection='3d', adjustable='box')
    ax.plot_surface(X, Y, Z, alpha=1, rstride=1, cstride=1, color='c')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c='k', s=15)
    #ax.set_title('L = '+str(L))
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    ax.set_xlabel('$X$', fontsize=15)
    ax.set_ylabel('$Y$', fontsize=15)
    ax.set_zlabel('$Z$', fontsize=15)

    return fig


# Generate random signals in specific spherical distributions
def aleatorySignals(x, y, N, maxValue=1, SNR=90):
    '''
    It generates a random signal of a specific 'SNR' for two point distributions
    on the sphere (usually one low resolution and the other high).
    The signal is bound to a maximum value of 'maxValue'.

    Parameters:
    ----------
    x:  array_like.
        Lx3 array; points distribution, given in Cartesian coordinates.
    y:  array_like.
        Ix3 array; points distribution, given in Cartesian coordinates.
    N:  Int.
        Maximun SFT order of the signal.
    maxValue:   scalar, optional.
        Maximun value of the random signal.
    SNR:    scalar.
        Signal to noise ratio for random signal.

    Returns:
    -------
    Sa: ndarray.
        Lx1 array; random signal in 'x' distribution.
    St: ndarray.
        Ix1 array; random signal in 'y' distribution.
    Snm:ndarray.
        (N+1)**2 x 1 array; random signal in SFT coeficients domain.
    '''

    Snm = np.random.normal(0, 0.1, size=((N+1)**2, 1))*maxValue
    Sa = sac.inverseSphericalFourierTransform(Snm, x)
    Sa, _ = addNoise(Sa, SNR)
    St = sac.inverseSphericalFourierTransform(Snm, y)
    St, _ = addNoise(St, SNR)
    return Sa, St, Snm


# Adding noise to signal
def addNoise(signal, SNR):
    '''
    Adds noise to a signal.

    Parameters:
    ----------
    signal: like_array.
        Signal to be processed
    SNR:    scalar.
        Signal to noise ratio for random signal.

    Returns:
    -------
    newSignal:  ndarray.
        New noisy signal
    noise:  ndarray.
        Noise signal.
    '''

    noise = np.random.normal(0, 0.1, np.shape(signal)[0])
    noise = np.reshape(noise, (np.shape(signal)[0], 1))
    snr = 10**(SNR/20)
    RMSsignal = float(np.sqrt((signal**2).mean()))
    RMSnoise = float(np.sqrt((noise**2).mean()))
    k = RMSsignal/(RMSnoise*snr)
    noise = k*noise
    newSignal = signal + noise
    return newSignal, noise


# Global error calculation
def plotGlobalError(signal, target, N, x, y, minLambda, maxLambda, resolution):
    '''
    Plot the interpolation error obtained when comparing a 'target' signal
    and an interpolated signal. The interpolated signal is obtained by applying
    the interpolation by the regularized SFT with order 'N' to the 'signal'.
    The regularization parameter ('lambda') of the regularized SFT, is a
    logarithmically spaced array of range [10**minLambda, 10**maxLambda>.
    The number of lambda values is determined by the variable 'resolution'.

    Parameters:
    ----------
    signal: like_array.
        Lx1 array; initial signal measured in a spherical array with an 'x'
        micrphones distribution.
    target: like_array.
        Ix1 Expected signal obtained by simulation in a spherical array with a
        'y' distribution of microphones.
    N:  Int.
        Maximun SFT order of the signal.
    x:  array_like.
        Lx3 array; points distribution, given in Cartesian coordinates.
    y:  array_like.
        Ix3 array; points distribution, given in Cartesian coordinates.
    minLambda:  scalar.
        Lower bound exponent; this is: 10**minLambda.
    maxLambda:  scalar.
        Higher bound exponent; this is: 10**maxLambda.
    resolution: int.
        Number of lambda values.

    Returns:
    -------
    fig:    figure.
        Figure that contains the interpolation error.
    '''
    i = 0
    L1 = np.shape(x)[0]
    L2 = np.shape(y)[0]
    arrayLambda = np.logspace(start=minLambda, stop=maxLambda, num=resolution)
    error = np.zeros(resolution)
    for regLambda in arrayLambda:
        Sb = sac.SFTinterpolation(signal, N, x, y, regLambda=regLambda)
        error[i] = sac.error(Sb, target)
        i += 1

    fig, ax = plt.subplots(1, 1)
    ax.plot(arrayLambda, error, 'b.', label='')
    ax.set_title('Globar Error, N = '+str(N) +
                 ', $L_{ini}$ = '+str(L1)+' $,L_{fin} = $'+str(L2), fontsize=9)
    ax.set_ylabel('20Log(|Error|)')
    ax.set_xlabel('Lambda')
    ax.set_xscale('log')
    return fig


# Plotting singular values
def plotSigma(N, x):
    '''
    Plot the singular values obtained by singular value decomposition (SVD)
    of the spherical harmonics matrix with maximum order N.
    The spherical harmonics matrix is calculated at 'x', where 'x' is a
    points distribution on the sphere.

    Parameters:
    ----------
    N:  Int.
        Maximun order of the spherical harmonics matrix.
    x:  array_like.
        Lx3 array; points distribution, given in Cartesian coordinates.

    Returns:
    -------
    fig:    figure.
        Figure that contains the singular values.
    '''
    Y = sac.sphericalHarmonicsMatrix(N, x)
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    fig, ax = plt.subplots(1, 1, figsize=(10, 1))
    yy = np.ones(np.shape(S)[0])
    ax.plot(S, yy, 'r.', ms=5)
    ax.set_xlabel('Sigma')
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='both')
    ax.set_yticks([])
    fig.tight_layout()
    #ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.grid(b=True, which='minor', axis='both')
    return fig


def scatterPlotFunction(signal, x, polarForm=True, s=30):
    '''
    Scatter plot of a signal sampled in a 'x' distribution on the sphere.

    Parameters:
    ----------
    signal: like_array.
        Lx1 array; signal to be plotted.
    x: like_array.
        Lx3 array; 'signal' sampling points.
    polarForm: Boolean, optional.
        If polarForm is True, the 'signal' is plotted in polar form, else
        it is plotted in spherical form. By default polarmform = True.
    s:  int.
        Point size in scattered plot.

    Returns:
    -------
    fig:    figure.
        Figure that contains the scattered plot.
    ax:     axis.
        Figure axis.
    '''

    theta, phi, rho = sac.cart2sph(x)
    signal = signal.flatten().real  # Real part of signal
    signalAbs = np.abs(signal)  # Absolute value

    #-Convert to cartesian coordinates
    if polarForm == True:
        X = signalAbs*np.cos(phi.flatten())*np.cos(theta.flatten())
        Y = signalAbs*np.cos(phi.flatten())*np.sin(theta.flatten())
        Z = signalAbs*np.sin(phi.flatten())

    if polarForm == False:
        X = np.cos(phi.flatten())*np.cos(theta.flatten())
        Y = np.cos(phi.flatten())*np.sin(theta.flatten())
        Z = np.sin(phi.flatten())

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(X, Y, Z, c=signal, s=s, cmap='jet', marker='o')
    ax.view_init(elev=15, azim=30)
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    ax.set_xlabel('$X$', fontsize=15)
    ax.set_ylabel('$Y$', fontsize=15)
    ax.set_zlabel('$Z$', fontsize=15)

    #--Adding colorbar
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(signal)
    fig.colorbar(m, shrink=0.5, label='Sound pressure')
    ax.tick_params(labelsize=7)
    return fig, ax


#  Function to plot sound presure in a spherical distribution.
def plotFunction(signal, x, polarForm=True):
    '''
    Plot of a signal sampled in a 'x' distribution on the sphere using convex
    hull.

    Parameters:
    ----------
    signal: like_array.
        Lx1 array; signal to be plotted.
    x: like_array.
        Lx3 array; 'signal' sampling points.
    polarForm: Boolean, optional.
        If polarForm is True, the 'signal' is plotted in polar form, else
        it is plotted in spherical form. By default polarmform = True.

    Returns:
    -------
    fig:    figure.
        Figure that contains the plot.
    ax:     axis.
        Figure axis.
    '''

    theta, phi, rho = sac.cart2sph(x)
    signal = signal.flatten().real  # Real part of signal
    signalAbs = np.abs(signal)  # Absolute value

    #-Convert to cartesian coordinates
    if polarForm == True:
        X = signalAbs*np.cos(phi.flatten())*np.cos(theta.flatten())
        Y = signalAbs*np.cos(phi.flatten())*np.sin(theta.flatten())
        Z = signalAbs*np.sin(phi.flatten())

    if polarForm == False:
        X = np.cos(phi.flatten())*np.cos(theta.flatten())
        Y = np.cos(phi.flatten())*np.sin(theta.flatten())
        Z = np.sin(phi.flatten())
    #-triangulation and plotting
    hull = ConvexHull(x)
    #--calculating matrix "colors" for the color of the surfaces
    colors = np.mean(signal[hull.simplices], axis=1)

    fig = plt.figure()
    ax = fig.gca(projection='3d', adjustable='box')

    #--Plotting using trisurf---------------------------------------------------
    s = ax.plot_trisurf(X, Y, Z, triangles=hull.simplices,
                        cmap='jet', alpha=1, edgecolor='k')
    # Colores de superficies en funcion del valor de "colors"
    s.set_array(colors)
    ax.set_xlabel('$X$', fontsize=15)
    ax.set_ylabel('$Y$', fontsize=15)
    ax.set_zlabel('$Z$', fontsize=15)

    #--Adding colorbar
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(signal)
    fig.colorbar(m, shrink=0.5, label='Sound pressure')
    ax.tick_params(labelsize=7)
    ax.view_init(elev=15, azim=30)
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    return fig, ax


# Plot SFT coefficients
def plotSnm(Snm, dBscale=False):
    '''
    Plot the values of the SFT coefficients.

    Parameters:
    ----------
    Snm:    like_array.
        Mx1 array; column vector containing the SFT coefficients.
    dBscale:    Boolean.
        If dBscale = True, Y axis is in logarmithmic scale, else, Y axis is in
        lineal scale.
    Returns:
    -------
    fig:    figure.
        Figure that contains the plot.
    ax:     axis.
        Figure axis.
    '''
    N = int(np.sqrt(np.shape(Snm)[0])-1)
    SnmPlot = Snm
    if dBscale:
        SnmPlot = 20*np.log10(np.absolute(Snm))
    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(0, (N+1)**2, 1)
    ax.plot(x, SnmPlot.flatten(), 'o')
    w = np.amax(Snm)-np.amin(Snm)
    # Dashed lines
    if not dBscale:
        ax.vlines(x, np.amin(Snm)-0.08*w, Snm, colors='k', linestyle="dashed")

    plt.xticks(np.floor(x**(0.5))**2+np.floor(x**(0.5)))
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(
        np.floor((np.floor(x**(0.5))**2+np.floor(x**(0.5)))**(0.5))))
    ax.xaxis.set_major_formatter(ticks_x)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=10)
    n = np.arange(0, N+1, 1)  # arrangement with the values of N
    lines = n**2-0.5  # square and subtract 0.5 to match the lines
    for i in lines:  # Plot the lines
        ax.axvline(i, color='red', ls='dotted')
    ax.set_xlabel('n: Order', size=15)
    '''
    if dBscale == True:
        ax.set_ylabel('$20Log(|Snm|)$', size=12)
    if dBscale == False:
        ax.set_ylabel('Snm')
    '''
    #ax.set_title('Snm Values', size=15, style='italic')
    plt.ylim(bottom=np.amin(Snm)-0.08*w)
    return fig, ax
