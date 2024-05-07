import numpy as np
import matplotlib.pyplot as plt
from methods import runge_kutta_4_system

def lotka_volterra(t, X, r1, r2, K1, K2, alpha12, alpha21):
    """Sistema de ecuaciones de Lotka-Volterra para competencia"""
    N1, N2 = X
    dN1_dt = r1 * N1 * (1 - N1 / K1 - alpha12 * N2 / K1)
    dN2_dt = r2 * N2 * (1 - N2 / K2 - alpha21 * N1 / K2)
    return np.array([dN1_dt, dN2_dt])

# Parámetros
r1, r2 = 0.5, 0.6  # Tasas de crecimiento
K1, K2 = 100, 80   # Capacidades de carga
alpha12, alpha21 = 0.3, 0.4  # Coeficientes de competencia
N1_0, N2_0 = 10, 15  # Condiciones iniciales

# Tiempo
t = np.linspace(0, 50, 1000)

# Solución numérica
X0 = np.array([float(N1_0), float(N2_0)])
X = runge_kutta_4_system(lotka_volterra, t, X0, args=(r1, r2, K1, K2, alpha12, alpha21))
N1, N2 = X.T

# Gráficas
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, N1, label='Especie 1')
plt.plot(t, N2, label='Especie 2')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(N1, N2)
plt.xlabel('Población Especie 1')
plt.ylabel('Población Especie 2')

plt.tight_layout()
plt.show()