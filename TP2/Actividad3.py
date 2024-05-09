import numpy as np
import matplotlib.pyplot as plt
from methods import runge_kutta_4_system
def lotka_volterra( t,X, r, alpha, beta, q):
    """Modelo de Predador-Presa de Lotka-Volterra"""
    N, P = X
    dN_dt = r * N - alpha * N * P
    dP_dt = beta * N * P - q * P
    return np.array([dN_dt, dP_dt])

def lotka_volterra_extendido( t,X, r, alpha, beta, q, K):
    """Modelo de Predador-Presa de Lotka-Volterra Extendido (LVE)"""
    N, P = X
    dN_dt = r * N * (1 - N / K) - alpha * N * P
    dP_dt = beta * N * P - q * P
    return np.array([dN_dt, dP_dt])



# Parámetros
r = 1  # Tasa de crecimiento de la presa
alpha = 0.1  # Eficiencia de captura
beta = 0.1 # Eficiencia de conversión de presas en predadores
q = 0.5  # Tasa de mortalidad de los predadores
K = 100  # Capacidad de carga (sólo para LVE)

# Tiempo
t = np.linspace(0, 20, 200)

# Condiciones iniciales para el modelo de Lotka-Volterra
N0 = 50
P0 = 10

# Condiciones iniciales para el modelo de Lotka-Volterra Extendido
N0_ext = 50
P0_ext = 10

# Soluciones numéricas
X0 = np.array([float(N0), float(P0)])
X = runge_kutta_4_system(lotka_volterra, t, X0, args=(r, alpha, beta, q))
N, P = X.T

X0_ext = np.array([float(N0_ext), float(P0_ext)])
X_ext = runge_kutta_4_system(lotka_volterra_extendido, t, X0_ext, args=(r, alpha, beta, q, K))
N_ext, P_ext = X_ext.T



#dN_dt, dN2_dt = lotka_volterra(0, [N1_gird, N2_grid], r1, r2, K1, K2, alpha12, alpha21)
#HACER STREAM PLOT
# Gráficas

plt.figure(figsize=(10, 6))
plt.subplot(3, 3, 1)
plt.plot(t, N, label='Presa')
plt.plot(t, P, label='Predador')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.title('Modelo de Lotka-Volterra')

plt.subplot(3, 3, 2)
plt.plot(N, P)
plt.xlabel('Presa')
plt.ylabel('Predador')
plt.legend()
plt.title('Modelo de Lotka-Volterra')




plt.subplot(3, 3, 3)
plt.plot(t, N_ext, label='Presa (LVE)')
plt.plot(t, P_ext, label='Predador (LVE)')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.title('Modelo de Lotka-Volterra Extendido')

plt.subplot(3, 3, 4)
plt.plot(N_ext, P_ext)
plt.xlabel('Presa')
plt.ylabel('Predador')
plt.legend()
plt.title('Modelo de Lotka-Volterra Extendido')

plt.show()

# Crear una cuadrícula de puntos
N_range = np.linspace(0, 100, 20)
P_range = np.linspace(0, 100, 20)
N_grid, P_grid = np.meshgrid(N_range, P_range)

# Calcular las derivadas en la cuadrícula de puntos
dN_dt_grid, dP_dt_grid = lotka_volterra(None, [N_grid, P_grid], r, alpha, beta, q)
dN_dt_grid_ext, dP_dt_grid_ext = lotka_volterra_extendido(None, [N_grid, P_grid], r, alpha, beta, q, K)

# Crear los stream plots
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.streamplot(N_grid, P_grid, dN_dt_grid, dP_dt_grid, color='r')
plt.title('Stream plot - Modelo de Lotka-Volterra')
plt.xlabel('Presa')
plt.ylabel('Predador')

plt.subplot(2, 2, 2)    
plt.streamplot(N_grid, P_grid, dN_dt_grid_ext, dP_dt_grid_ext, color='b')
plt.title('Stream plot - Modelo de Lotka-Volterra Extendido')
plt.xlabel('Presa')
plt.ylabel('Predador')

plt.show()