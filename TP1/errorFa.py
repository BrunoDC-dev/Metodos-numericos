from scipy.interpolate import lagrange, CubicSpline, interp1d,RegularGridInterpolator,RectBivariateSpline, griddata
from math import sin, tanh, exp
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebpts1

def originalFunction(x:int):
    return ((0.3**abs(x)) * (sin(4*x))) - tanh(2*x) + 2


def equispaced(bool:bool, start:int, end:int,n:int)->list: 

    if (bool):
         return np.linspace(start,end,n)
    else:
        x_cheb = chebpts1(n)
        x_scaled_cheb =  0.5 * (end - start) * x_cheb + 0.5 * (end + start)
        return x_scaled_cheb
        


x_original_cords= equispaced(False,-4,4,10)

y_original_cords = []


for x in x_original_cords:
    y_original_cords.append(originalFunction(x))

lagrange = lagrange(x_original_cords,y_original_cords)
cubicSpline= CubicSpline(x_original_cords, y_original_cords)
linealInterpolation = interp1d(x_original_cords,y_original_cords)

x_interpolate = np.arange(min(x_original_cords), max(y_original_cords), 0.1)
y_originalFunction = []
y_lagrange =[]
y_spline = []
y_lineal=[]

for x in x_interpolate:
    y_originalFunction.append(originalFunction(x))
    y_lagrange.append(lagrange(x))
    y_spline.append(cubicSpline(x))
    y_lineal.append(linealInterpolation(x))


Error_lagrange = []
Error_Spline= []
Error_lineal = []

for i in range(len(y_originalFunction)):
    Error_lagrange.append(abs(y_originalFunction[i]- y_lagrange[i]))
    Error_Spline.append(abs(y_originalFunction[i]- y_spline[i]))
    Error_lineal.append(abs(y_originalFunction[i]- y_lineal[i]))



if __name__ == "__main__":

    fig, ax = plt.subplots()
    ax.set_xlabel('Coordenadas interpoladas')
    ax.set_ylabel('Error absoluto')


    ax.plot(x_interpolate, Error_lagrange, linestyle='--', label="Lagrange")
    ax.plot(x_interpolate, Error_Spline, linestyle='--', label="Interpolador cubico")
    ax.plot(x_interpolate, Error_lineal, linestyle='--', label="Interpolador lineal")

    ax.legend(loc='best')


    plt.show()




