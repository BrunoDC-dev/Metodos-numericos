import numpy as np
import matplotlib.pyplot as plt
from methods import euler_method,runge_kutta_4

# Parámetros
N0 =10  # Condición inicial
r = 0.1  # Tasa de crecimiento
K = 50  # Capacidad de carga
# Tiempo
t = np.linspace(0, 10, 500)
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


#exacta grafiquen el tamaño poblacional en función del tiempo (N vs t) 

plt.subplot(3,3 ,1)
plt.plot(t, N_exp, label="Modelo exponencial exacta")
plt.xlabel("Tiempo")
plt.ylabel("Población")
plt.legend()
plt.title("Modelo exponencial Nvst")

plt.subplot(3,3 ,2)
plt.plot(t, N_rk_exp, label="Modelo exponencial RK4")
plt.xlabel("Tiempo")
plt.ylabel("Población")
plt.legend()
plt.title("Modelo exponencial RK4")

plt.subplot(3,3 ,3)
plt.plot(t, N_log, label="Modelo logístico exacta")
plt.xlabel("Tiempo")
plt.ylabel("Población")
plt.legend()
plt.title("Modelo logístico Nvst")

plt.subplot(3,3 ,4)
plt.plot(t, N_rk_log, label="Modelo logístico RK4")
plt.xlabel("Tiempo")
plt.ylabel("Población")
plt.legend()



#la variación poblacional en función dN/dt vs N ). 

plt.subplot(3,3 ,5)
plt.plot(N_exp, N_rk_exp, label="derivada exponencial")
plt.xlabel("tasa de crecimiento")
plt.ylabel("Población")
plt.legend()
plt.title("Modelo exponencial dN/dt vs N")

plt.subplot(3,3 ,6)
plt.plot(N_log,N_rk_log, label="derivada logística")
plt.xlabel("tasa de crecimiento")
plt.ylabel("Población")
plt.legend()
plt.title("Modelo logístico dN/dt vs N")


plt.subplot(3,3 ,7)
plt.plot(t,N_exp, label="Modelo exponencial")
plt.plot(t,N_log, label="Modelo logístico")
plt.xlabel("Tiempo")
plt.ylabel("Población")
plt.legend()
plt.title("Modelo exponencial vs logístico")


# Define the population sizes and their rates of change
N_values = np.linspace(0, 100, 500)
t_values = np.linspace(0, 100, 500)

# Create a grid of points
N, t = np.meshgrid(N_values, t_values)

# Calculate the rates of change at each point in the grid
U_exp = np.ones_like(N)
V_exp = diferencial_exponencial(N, t)
U_log = np.ones_like(N)
V_log = diferencial_logistico(N, t)

# Create the stream plot for the exponential model
plt.subplot(3, 3, 8)
plt.streamplot(N, t, U_exp, V_exp, color='r')
plt.xlabel('Población')
plt.ylabel('Tiempo')
plt.title('Diagrama de fases - Modelo exponencial')

# Create the stream plot for the logistic model
plt.subplot(3, 3, 9)
plt.streamplot(N, t, U_log, V_log, color='b')
plt.xlabel('Población')
plt.ylabel('Tiempo')
plt.title('Diagrama de fases - Modelo logístico')


plt.tight_layout()
plt.show()