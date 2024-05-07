import numpy as np
import matplotlib.pyplot as plt
from methods import euler_method,runge_kutta_4

# Parámetros
N0 = 10  # Condición inicial
r = 0.1  # Tasa de crecimiento
K = 100  # Capacidad de carga
# Tiempo
t = np.linspace(0, 150, 1500)
h = t[1] - t[0]  # Paso de tiempoh = t[1] - t[0]  # Paso de tiempo
n = len(t)  # Número de puntos
def modelo_exponencial(N0, t , r=r,):
    """Solución exacta del modelo exponencial"""
    return N0 * np.exp(r * t)

def diferencial_exponencial(N, t , r=r,):
    """Ecuación diferencial del modelo exponencial"""
    return r * N

def modelo_logistico(N0, t,  r=r,  K=K):
    """Solución exacta del modelo logístico"""
    return (N0*K*np.exp(r*t))/((K-N0) + N0*np.exp(r*t))

def diferencial_logistico(N, t,  r=r,):

    """Ecuación diferencial del modelo logístico"""
    return r * N * ((K - N) / K)

def exponential_derivative(N,  t, r=r):
    return r * N * np.exp(r * t)

def logistic_derivative(N,t,r=r, K=K,):
    return ( N * K * r * np.exp(r * t) * (K - N * np.exp(r * t))) / ((K - N + N * np.exp(r * t)) ** 2)


# Soluciones exactas
N_exp = modelo_exponencial(N0,  t, r)
N_log = modelo_logistico(N0,  t, r, K)

N_euler_exp = euler_method(diferencial_exponencial, t, N0)
N_rk_exp = runge_kutta_4(diferencial_exponencial, t, N0)
N_euler_log = euler_method(diferencial_logistico, t, N0)
N_rk_log = runge_kutta_4(diferencial_logistico, t, N0)


N_euler_derivate_exp = euler_method(exponential_derivative, t, N0)
N_rk_derivate_exp = runge_kutta_4(exponential_derivative, t, N0)
N_euler_derivate_log = euler_method(logistic_derivative, t, N0)
N_rk_derivate_log = runge_kutta_4(logistic_derivative, t, N0)


derivate_exp_simple = exponential_derivative(N0, t)
derivate_log_simple = logistic_derivative(N0, t)


# Gráficas
plt.figure(figsize=(10, 6))

# # Modelo exponencial
# plt.subplot(2, 2, 1)
# plt.plot(t, N_exp, label='Exacta')
# plt.plot(t, N_euler_exp, label='Euler')
# plt.plot(t, N_rk_exp, label='Runge-Kutta')
# plt.xlabel('Tiempo')
# plt.ylabel('Población')
# plt.title('Modelo exponencial')
# plt.legend()



# plt.subplot(2, 2, 1)
# plt.plot(t, N_exp, label='Exacta')
# plt.plot(t, N_log, label='Exacta')
# plt.xlabel('Tiempo')
# plt.ylabel('Población')
# plt.title('Modelo exponencial')
# plt.legend()

# plt.subplot(2, 2, 1)
# plt.plot(t,derivate_exp_simple, label='Derivada')
# plt.plot(t,derivate_log_simple, label='Derivada')
# plt.xlabel('Tiempo')
# plt.ylabel('Población')
# plt.title('Modelo exponencial')
# plt.legend()



# Modelo exponencial
plt.subplot(2, 2, 1)
plt.plot(t, N_exp, label='Exacta')
plt.plot(t, N_euler_exp, label='Euler')
plt.plot(t, N_rk_exp, label='Runge-Kutta')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Modelo exponencial')
plt.legend()

# Modelo logístico
plt.subplot(2, 2, 2)
plt.plot(t, N_log, label='Exacta')
plt.plot(t, N_euler_log, label='Euler')
#plt.plot(t, N_rk_log, label='Runge-Kutta')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Modelo logístico')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, N_exp, label='Exacta')
plt.plot(t, N_log, label='Exacta')
plt.plot(t, N_euler_exp, label='Euler')
plt.plot(t,N_euler_log , label='Euler')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Modelos exponencial y logístico con método de Euler')
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(t, N_exp, label='Exacta')
plt.plot(t, N_log, label='Exacta')
plt.plot(t, N_rk_exp, label='Runge-Kutta')
plt.plot(t, N_rk_log, label='Runge-Kutta')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Modelos exponencial y logístico con método de Runge-Kutta')
plt.legend()

plt.tight_layout()
plt.show()