import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lotka_volterra(X, t, r, alpha, beta, q):
    """Modelo de Predador-Presa de Lotka-Volterra"""
    N, P = X
    dN_dt = r * N - alpha * N * P
    dP_dt = beta * N * P - q * P
    return [dN_dt, dP_dt]

def lotka_volterra_extendido(X, t, r, alpha, beta, q, K):
    """Modelo de Predador-Presa de Lotka-Volterra Extendido (LVE)"""
    N, P = X
    dN_dt = r * N * (1 - N / K) - alpha * N * P
    dP_dt = beta * N * P - q * P
    return [dN_dt, dP_dt]

# Parámetros
r = 1.0  # Tasa de crecimiento de la presa
alpha = 0.2  # Eficiencia de captura
beta = 0.5  # Eficiencia de conversión de presas en predadores
q = 0.3  # Tasa de mortalidad de los predadores
K = 100  # Capacidad de carga (sólo para LVE)

# Condiciones iniciales
N0 = 50
P0 = 10

# Tiempo
t = np.linspace(0, 50, 1000)

# Soluciones numéricas
X0 = [N0, P0]
X_lv = odeint(lotka_volterra, X0, t, args=(r, alpha, beta, q))
X_lve = odeint(lotka_volterra_extendido, X0, t, args=(r, alpha, beta, q, K))

N_lv, P_lv = X_lv.T
N_lve, P_lve = X_lve.T

# Gráficas
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, N_lv, label='Presa')
plt.plot(t, P_lv, label='Predador')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Modelo de Lotka-Volterra')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(N_lv, P_lv)
plt.xlabel('Población de Presa')
plt.ylabel('Población de Predador')
plt.title('Plano de Fase - Modelo de Lotka-Volterra')

plt.subplot(2, 2, 3)
plt.plot(t, N_lve, label='Presa')
plt.plot(t, P_lve, label='Predador')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Modelo de Lotka-Volterra Extendido (LVE)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(N_lve, P_lve)
plt.xlabel('Población de Presa')
plt.ylabel('Población de Predador')
plt.title('Plano de Fase - Modelo de Lotka-Volterra Extendido (LVE)')

plt.tight_layout()
plt.show()