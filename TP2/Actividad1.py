import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros
N0 =10  # Condición inicial
r = 0.1 # Tasa de crecimiento
K = 50  # Capacidad de carga
# Tiempo
t = np.linspace(0, 80,800)
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

def diferencial_logistico(N, t,  r=r, K=K):

    """Ecuación diferencial del modelo logístico"""
    return r * N * ((K - N) / K)



def euler_method(f, t, y0):
    y = np.zeros(len(t))
    y[0] = y0
    for n in range(0, len(t) - 1):
        y[n+1] = y[n] + f(y[n], t[n])*(t[n+1] - t[n])
    return y

# 4th order Runge-Kutta method
def runge_kutta_4(f, t, y0):
    y = np.zeros(len(t))
    y[0] = y0
    for n in range(0, len(t) - 1):
        h = t[n+1] - t[n]
        k1 = h * f(y[n], t[n])
        k2 = h * f(y[n] + 0.5 * k1, t[n] + 0.5 * h)
        k3 = h * f(y[n] + 0.5 * k2, t[n] + 0.5 * h)
        k4 = h * f(y[n] + k3, t[n] + h)
        y[n+1] = y[n] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y

def runge_kutta_2(f, t, y0):
    y = np.zeros(len(t))
    y[0] = y0
    for n in range(0, len(t) - 1):
        h = t[n+1] - t[n]
        k1 = h * f(y[n], t[n])
        k2 = h * f(y[n] + k1, t[n] + h)
        y[n+1] = y[n] + 0.5 * (k1 + k2)
    return y





# Soluciones exactas
N_exp = modelo_exponencial(N0,  t, r)
N_log = modelo_logistico(N0,  t, r, K)

N_euler_exp = euler_method(diferencial_exponencial, t, N0)
N_euler_log = euler_method(diferencial_logistico, t, N0)



N_rk2_exp = runge_kutta_2(diferencial_exponencial, t, N0)
N_rk2_log = runge_kutta_2(diferencial_logistico, t, N0)

N_rk4_exp = runge_kutta_4(diferencial_exponencial, t, N0)
N_rk4_log = runge_kutta_4(diferencial_logistico, t, N0)






# Cálculo de errores para cada método
error_euler_exp = np.abs(N_exp - N_euler_exp)
error_euler_log = np.abs(N_log - N_euler_log)


error_rk2_exp = np.abs(N_exp - N_rk2_exp)
error_rk2_log = np.abs(N_log - N_rk2_log)

error_rk4_exp = np.abs(N_exp - N_rk4_exp)
error_rk4_log = np.abs(N_log - N_rk4_log)

# Set a style
plt.style.use('seaborn-darkgrid')

# Create a figure for the exponential model
fig_exp, axs_exp = plt.subplots(1, 3, figsize=(18, 6))

# Modelo exponencial
axs_exp[0].plot(t, N_exp, label='Exacta')
axs_exp[0].plot(t, N_euler_exp, label='Euler')
axs_exp[0].set_xlabel('tiempo')
axs_exp[0].set_ylabel('Población')
axs_exp[0].set_title('Modelo exponencial - Euler')
axs_exp[0].legend()

axs_exp[1].plot(t, N_exp, label='Exacta')
axs_exp[1].plot(t, N_rk2_exp, label='Runge-Kutta 2')
axs_exp[1].set_xlabel('tiempo')
axs_exp[1].set_ylabel('Población')
axs_exp[1].set_title('Modelo exponencial - Runge-Kutta 2')
axs_exp[1].legend()

axs_exp[2].plot(t, N_exp, label='Exacta')
axs_exp[2].plot(t, N_rk4_exp, label='Runge-Kutta')
axs_exp[2].set_xlabel('tiempo')
axs_exp[2].set_ylabel('Población')
axs_exp[2].set_title('Modelo exponencial - Runge-Kutta 4')
axs_exp[2].legend()

plt.tight_layout()
plt.show()

# Create a figure for the logistic model
fig_log, axs_log = plt.subplots(1, 3, figsize=(18, 6))

# Modelo logístico
axs_log[0].plot(t, N_log, label='Exacta')
axs_log[0].plot(t, N_euler_log, label='Euler')
axs_log[0].set_xlabel('tiempo')
axs_log[0].set_ylabel('Población')
axs_log[0].set_title('Modelo logístico - Euler')
axs_log[0].legend()

axs_log[1].plot(t, N_log, label='Exacta')
axs_log[1].plot(t, N_rk2_log, label='Runge-Kutta 2')
axs_log[1].set_xlabel('tiempo')
axs_log[1].set_ylabel('Población')
axs_log[1].set_title('Modelo logístico - Runge-Kutta 2')
axs_log[1].legend()

axs_log[2].plot(t, N_log, label='Exacta')
axs_log[2].plot(t, N_rk4_log, label='Runge-Kutta')
axs_log[2].set_xlabel('tiempo')
axs_log[2].set_ylabel('Población')
axs_log[2].set_title('Modelo logístico - Runge-Kutta 4')
axs_log[2].legend()

plt.tight_layout()
plt.show()

# Create a figure for the comparison and derivative plots
fig_comp, axs_comp = plt.subplots(2, 2, figsize=(12, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.1)

# exacta vs modelo 
axs_comp[0, 0].plot(t,N_exp, label='Exponencial')
axs_comp[0, 0].set_xlabel('tiempo')
axs_comp[0, 0].set_ylabel('Población')
axs_comp[0, 0].set_title(' Modelo exponencial')
axs_comp[0, 0].legend()

axs_comp[0, 1].plot(t, N_log, label='Logistico')
axs_comp[0, 1].set_xlabel('tiempo')
axs_comp[0, 1].set_ylabel('Población')
axs_comp[0, 1].set_title(' Modelo logístico')
axs_comp[0, 1].legend()

# Derivadas DN/Dt vs N
axs_comp[1, 0].plot(N_exp,diferencial_exponencial(N_exp, t), label='Exponencial')
axs_comp[1, 0].set_xlabel('Población')
axs_comp[1, 0].set_ylabel('dN/dt')
axs_comp[1, 0].set_title('Derivadas DN/Dt vs N Modelo exponencial')
axs_comp[1, 0].legend()

axs_comp[1, 1].plot(N_log,diferencial_logistico(N_log, t), label='Logístico')
axs_comp[1, 1].set_xlabel('Población')
axs_comp[1, 1].set_ylabel('dN/dt')
axs_comp[1, 1].set_title('Derivadas DN/Dt vs N Modelo logístico')
axs_comp[1, 1].legend()

plt.show()
# Set a style
plt.style.use('seaborn-darkgrid')

# Create a figure
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the errors for the exponential model
axs[0].semilogy(t, error_euler_exp, label='Euler', linewidth=2)
axs[0].semilogy(t, error_rk2_exp, label='Runge-Kutta 2', linewidth=2)
axs[0].semilogy(t, error_rk4_exp, label='Runge-Kutta 4', linewidth=2)
axs[0].set_xlabel('Tiempo', fontsize=14)
axs[0].set_ylabel('Error', fontsize=14)
axs[0].set_title('Error Modelo Exponencial', fontsize=16)
axs[0].legend(fontsize=12)

# Plot the errors for the logistic model
axs[1].semilogy(t, error_euler_log, label='Euler', linewidth=2)
axs[1].semilogy(t, error_rk2_log, label='Runge-Kutta 2', linewidth=2)
axs[1].semilogy(t, error_rk4_log, label='Runge-Kutta 4', linewidth=2)
axs[1].set_xlabel('Tiempo', fontsize=14)
axs[1].set_ylabel('Error', fontsize=14)
axs[1].set_title('Error Modelo Logístico', fontsize=16)
axs[1].legend(fontsize=12)

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()


r_values = [0.75, 0.5 ,0.25,0.1,0.05, 0 ,  -0.05,  -0.2 ]
# Crear una figura
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Iterar sobre los valores de r
for r in r_values:
    # Calcular los resultados
    N_exp = modelo_exponencial(N0,  t, r)
    N_log = modelo_logistico(N0,  t, r, K)


    # Graficar los resultados de la función exponencial en el primer subgráfico
    axs[0 , 0].semilogy(t, N_exp, label=f'r = {r}')
    axs[0 , 0].set_xlabel('Tiempo')
    axs[0 , 0].set_ylabel('Población')
    axs[0 , 0].set_title('Modelo Exponencial')
    axs[0, 0].legend()

    # Graficar los resultados de la función logística en el segundo subgráfico
    axs[1, 0].plot(t, N_log, label=f'r = {r}')
    axs[1 , 0].set_xlabel('Tiempo')
    axs[1, 0].set_ylabel('Población')
    axs[1, 0].set_title('Modelo Logístico')
    axs[1,0 ].legend()


    #graficar derviadas

    axs[0, 1].plot(N_exp , diferencial_exponencial(N_exp , t , r) , label=f'r = {r}')
    axs[0, 1].set_xlabel('Población')
    axs[0, 1].set_ylabel('dN/dt')
    axs[0, 1].set_title('Derivada Modelo Exponencial')
    axs[0, 1].legend()
    axs[0, 1].set_xlim([-1, 100])  # Reemplaza min_value y max_value con los límites que deseas para el eje x
    axs[0, 1].set_ylim([-5,100])

    axs[1, 1].plot(N_log , diferencial_logistico(N_log , t , r) , label=f'r = {r}')
    axs[1, 1].set_xlabel('Población')
    axs[1, 1].set_ylabel('dN/dt')
    axs[1, 1].set_title('Derivada Modelo Logístico')
    axs[1, 1].legend()

# Mostrar la figura
plt.tight_layout()

plt.show()

K_values = [100, 80, 60, 50, 40, 20, 10]  # Valores de K a explorar

# Crear una figura
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Iterar sobre los valores de K
for K in K_values:
    # Calcular los resultados del modelo logístico
    N_log = modelo_logistico(N0, t, r=0.1, K=K)

    # Graficar los resultados del modelo logístico
    axs[0].plot(t, N_log, label=f'K = {K}')
    axs[0].set_xlabel('Tiempo')
    axs[0].set_ylabel('Población')
    axs[0].set_title('Modelo Logístico')
    axs[0].legend()

    # Graficar las derivadas del modelo logístico
    axs[1].plot(N_log, diferencial_logistico(N_log, t, r=0.1, K=K), label=f'K = {K}')
    axs[1].set_xlabel('Población')
    axs[1].set_ylabel('dN/dt')
    axs[1].set_title('Derivada Modelo Logístico')
    axs[1].legend()

# Ajustar el layout
plt.tight_layout()

# Mostrar la figura
plt.show()