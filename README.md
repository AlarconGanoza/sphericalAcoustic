# Spherical Acoustic

## 1. About Spherical Acoustic
Spherical Acoustic is a project focused on room impulse response (RIR) interpolation using the regularized spherical Fourier transform (SFT). The algorithms are applicable to initial data measured by spherical arrays of microphones with different distributions. The project is developed in python although the initial data was simulated in matlab. The general diagram of the RIR interpolation is shown below.

<div align="center">
<img src="./images/generalDiagram.png" width="500">
</div>

## 2. Libraries available
Currently there are 2 libraries:
|Library name           | description                                         |
|-----------------------|-----------------------------------------------------|
|<a href="https://github.com/AlarconGanoza/sphericalAcoustic/blob/master/sphericalAcoustics.py">sphericalAcoustics.py</a>  | Functions related to regularized SFT                |
|<a href="https://github.com/AlarconGanoza/sphericalAcoustic/blob/master/tools.py">tools.py</a>| Functions related to the points distribution on the sphere and 3D signal plotting|


## 3. Initial acoustic data available
Some RIRs simulated in Matlab are available in <a href="https://github.com/AlarconGanoza/sphericalAcoustic/tree/master/initialRIR">/initialRIR</a> .
RIRs are obtained using the tool developed in [1].

## 4. Usage
To use the libraries you have to copy both .py files to the work environment and import.

For example, to generate and plot a random distribution of **L** points on a sphere of radius 8 cm.

```python
import sphericalAcoustics as sac
import tools as sat
import matplotlib.pyplot as plt

x = sat.randDist(L=40, r=0.08)
fig = sat.plotDist(x)
plt.show()
```

<div align="center">
<img src="./images/randDistPlot.png" width="300">
</div>

## 5. Interpolation example
An example of RIR interpolation is shown in <a href="https://github.com/AlarconGanoza/sphericalAcoustic/blob/master/RIR_interpolation_example.ipynb">RIR_interpolation_Example.ipynb</a>.

  - Interpolation errors using the regularized and non-regularized SFT are compared.


## 6. References
[1]D. P. Jarret, E. A. P. Habets, M. R. P. Thomas, y P. A. Naylor, «Rigid sphere room impulse response simulation: algorithm and applications», J. Acoust. Soc. Am., vol. 132, n.º 3, pp. 1462-1472, sep. 2012, doi: 10.1121/1.4740497.
https://www.audiolabs-erlangen.de/fau/professor/habets/software/smir-generator

## 7. Author
Julio Alarcón Ganoza
