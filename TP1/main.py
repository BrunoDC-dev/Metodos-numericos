import numpy as np
from math import exp, sin, tanh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x):
    return (0.3**abs(x))*sin(4*x)-tanh(2*x) +2

def g(x, y):
    return 0.75*exp(-((10*x-2)**2)/4 - ((9*y-2)**2)/4) + 0.65*exp(-((9*x+1)**2)/9 - ((10*y+1)**2)/2) + 0.55*exp(-((9*x-6)**2)/4 - ((9*y-3)**2)/4) - 0.01*exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)




for i in range(5):
    fx_interpolate_values =  np.linspace(-1,1,3*2**i)
   # gx_interpolate_values =  np.linspace(-4,4,4*2**i)
    print(fx_interpolate_values)



# Intervalo para f(x)
x_values = np.linspace(-4, 4, 100)
f_values = [f(x) for x in x_values]

# Gráfico para f(x)
plt.figure(figsize=(8, 6))
plt.plot(x_values, f_values, label='f(x)')
plt.legend()
plt.title('Gráfico de f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
# Intervalo para g(x, y)
x_values = np.linspace(-1, 1, 100)
y_values = np.linspace(-1, 1, 100)
g_values = [[g(x, y) for x in x_values] for y in y_values]





# Gráfico para g(x, y)
x_values, y_values = np.meshgrid(x_values, y_values)
g_values = np.array(g_values)  # Convertir g_values a un array de NumPy
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_values, y_values, g_values, cmap='viridis')
ax.set_title('Gráfico de g(x, y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('g(x, y)')
plt.show()