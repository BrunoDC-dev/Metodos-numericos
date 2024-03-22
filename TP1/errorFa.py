import scipy.interpolate
import math
import matplotlib.pyplot as plt
import numpy as np

def originalFunction(x):
    return ((0.3**abs(x)) * (math.sin(4*x))) - math.tanh(2*x) + 2


x_original_cords= np.arange(-4, 4.5, 0.5) # equiespaciados

y_original_cords = []


for x in x_original_cords:
    y_original_cords.append(originalFunction(x))

lagrange = scipy.interpolate.lagrange(x_original_cords,y_original_cords)
cubicSpline= scipy.interpolate.CubicSpline(x_original_cords, y_original_cords)

x_interpolate=np.arange(-4, 4.25, 0.25)
y_originalFunction = []
y_lagrange =[]
y_spline = []

for x in x_interpolate:
    y_originalFunction.append(originalFunction(x))
    y_lagrange.append(lagrange(x))
    y_spline.append(cubicSpline(x))




Error_lagrange = []
Error_Spline= []
for i in range(len(y_originalFunction)):
    Error_lagrange.append((abs(y_originalFunction[i]- y_lagrange[i]))/abs(y_originalFunction[i]))
    Error_Spline.append((abs(y_originalFunction[i]- y_spline[i]))/abs(y_originalFunction[i]))




fig, ax = plt.subplots()
ax.set_xlabel('Coordenadas interpoladas')
ax.set_ylabel('Error relativo')


ax.plot(x_interpolate, Error_lagrange, linestyle='--', label="Lagrange")
ax.plot(x_interpolate, Error_Spline, linestyle='--', label="Interpolador cubico")

ax.legend(loc='best')

plt.show()




