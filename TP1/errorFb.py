from errorFa import exp,  RegularGridInterpolator,RectBivariateSpline, np, griddata, plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d # no me deja directo del archivo ?
import chaospy as cp
import math

def originalFunction(x1, x2):
    return 0.75 * exp((-((10*x1-2)**2)/4) - ((9*x2 - 2)**2)/4) + \
           0.65 * exp((-((9*x1+1)**2)/9) - ((10*x2 +1 )**2)/2) + \
           0.55 * exp((-((9*x1-6)**2)/4) - ((9*x2 - 3)**2)/4) - \
           0.01 * exp((-((9*x1-7)**2)/4) - ((9*x2 - 3)**2)/4)

def equispaced(bool, n:int=1):

    if bool:
        return [np.linspace(-1, 1, 49),np.linspace(-1, 1, 49)] # 49 puntos equiespaciados
    else:
        distribucion = cp.create_chebyshev_samples(math.sqrt(40),2) #genera secuencia de 49 puntos para x1 y x2
        x1 = distribucion[0, :]
        x2 = distribucion[1, :]

        return [x1,x2]



x1 = equispaced(True)[0]
x2 = equispaced(True)[1]





print(len(x1))
print("--")
print(len(x2))


z = []

for i in range(len(x1)):
    z.append(originalFunction(x1[i],x2[i]))


x1_interpolate = np.linspace(-1, 1, 5)
x2_interpolate = np.linspace(-1, 1, 5)

lineal_interp = interp2d(x1, x2, z, kind='linear')
z_lineal = lineal_interp(x1_interpolate,x2_interpolate)

cubic_interp = interp2d(x1, x2, z, kind='cubic')
z_cubic = cubic_interp(x1_interpolate,x2_interpolate)

quintic_interp = interp2d(x1, x2, z, kind='quintic')
z_quintic = cubic_interp(x1_interpolate,x2_interpolate)



z_original = []
auxGrid = [] # para graficar

for i in x1_interpolate:
    aux = []
    xixw = []
    for w in x2_interpolate:
        xixw.append(originalFunction(i,w))
        aux.append((i,w))
    z_original.append(xixw)
    auxGrid.append(aux)


errorLineal = []
errorCubic = []
errorQuintic = []

for i in range(len(z_original)):
    auxLineal = []
    auxCubic = []
    auxQuintic = []
    for w in range(len(z_original[i])):
        auxLineal.append(abs(z_original[i][w]-z_lineal[i][w]))
        auxCubic.append(abs(z_original[i][w]-z_cubic[i][w]))
        auxQuintic.append(abs(z_original[i][w]-z_quintic[i][w]))
    errorLineal.append(auxLineal)
    errorCubic.append(auxCubic)
    errorQuintic.append(auxQuintic)
   
#print("pts z: ")
#print(z_original[0])
#print("pts z lineal: ")
#print(z_lineal[0])
#print("pts z cubic: ")
#print(z_cubic[0])
#print("pts z quintic: ")
#print(z_quintic[0])
#print("error abs lineal: ")
#print(errorLineal[0])
#print("error abs cubic: ")
#print(errorCubic[0])
#print("error abs lineal: ")
#print(errorQuintic[0])
    


errorLineal = [elemento for sublista in errorLineal for elemento in sublista]
errorCubic = [elemento for sublista in errorCubic for elemento in sublista]
errorQuintic = [elemento for sublista in errorQuintic for elemento in sublista]
z_original = [elemento for sublista in z_original for elemento in sublista]


x1s = []
x2s = []
for i in range(len(auxGrid)):
    for w in range(len(auxGrid[i])):
        x1s.append(auxGrid[i][w][0])
        x2s.append(auxGrid[i][w][1])


fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(hspace=0.5)

# Subgráfico 1
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(x1s, x2s, errorLineal, c='r', marker='o')  # c es el color, marker es el marcador
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
#ax1.set_zlabel('Error absoluto')
ax1.set_title('Interpolacion lineal')
ax1.grid(False)

# Subgráfico 2
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(x1s, x2s, errorCubic, c='g', marker='o')  # c es el color, marker es el marcador
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
#ax2.set_zlabel('Error absoluto')
ax2.set_title('Interpolacion cubica')
ax2.grid(False)

# Subgráfico 3
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(x1s, x2s, errorQuintic, c='b', marker='o')  # c es el color, marker es el marcador
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
#ax3.set_zlabel('Error absoluto')
ax3.set_title('Interpolacion quintica')
ax3.grid(False)

# Ajustar el diseño para evitar superposiciones
plt.tight_layout()

# Mostrar los subgráficos
plt.show()

"""
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Gráfico para Cubic spline
axs[0].plot(z_original, errorCubic, marker='o', linestyle='', label="Cubic spline", color='red')
axs[0].set_xlabel('Z')
axs[0].set_ylabel('Error absoluto')
axs[0].legend()

# Gráfico para Lineal spline
axs[1].plot(z_original, errorLineal, marker='o', linestyle='', label="Lineal spline", color='green')
axs[1].set_xlabel('Z')
axs[1].set_ylabel('Error absoluto')
axs[1].legend()

# Gráfico para Quintic spline
axs[2].plot(z_original, errorQuintic, marker='o', linestyle='', label="Quintic spline", color='blue')
axs[2].set_xlabel('Z')
axs[2].set_ylabel('Error absoluto')
axs[2].legend()

plt.tight_layout()
plt.show()

"""
    









