from errorFa import exp,  RegularGridInterpolator,RectBivariateSpline, np, griddata, plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d 
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
        distribucion = cp.create_chebyshev_samples(math.sqrt(40),2) # 49 chebsyhev
        x1 = distribucion[0, :]
        x2 = distribucion[1, :]

        return [x1,x2]



x1 = equispaced(False)[0]
x2 = equispaced(False)[1]





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
        auxLineal.append(abs(z_original[i][w]-z_lineal[i][w])/abs(z_original[i][w]))
        auxCubic.append(abs(z_original[i][w]-z_cubic[i][w])/abs(z_original[i][w]))
        auxQuintic.append(abs(z_original[i][w]-z_quintic[i][w])/abs(z_original[i][w]))
    errorLineal.append(auxLineal)
    errorCubic.append(auxCubic)
    errorQuintic.append(auxQuintic)
   
    

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
    
if __name__ == "__main__":

    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.5)


    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(x1s, x2s, errorLineal, c='r', marker='o')  
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_title('Interpolacion lineal')
    ax1.grid(False)


    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(x1s, x2s, errorCubic, c='g', marker='o')  
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_title('Interpolacion cubica')
    ax2.grid(False)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(x1s, x2s, errorQuintic, c='b', marker='o')  
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    ax3.set_title('Interpolacion quintica')
    ax3.grid(False)

    plt.tight_layout()
    plt.show()