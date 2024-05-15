import numpy as np
import matplotlib.pyplot as plt
from methods import runge_kutta_4_system

def lotka_volterra(t, X, r1, r2, K1, K2, alpha12, alpha21):
    """Sistema de ecuaciones de Lotka-Volterra para competencia"""
    N1, N2 = X
    dN1_dt = r1 * N1 * ((K1 -N1 - alpha12 * N2) / K1)
    dN2_dt = r2 * N2 * ((K2-N2 - alpha21*N1) / K2)
    return np.array([dN1_dt, dN2_dt])

paremter_dict = [
    {
        'r1': 0.4,
        'r2': 0.8,
        'K1': 80,
        'K2': 100,
        'alpha12': 2,
        'alpha21': 0.6,
        'N1_0': 10,
        'N2_0': 15
    },
    {
        'r1': 0.4,
        'r2': 0.8,
        'K1': 80,
        'K2': 100,
        'alpha12': 0.6,
        'alpha21': 2,
        'N1_0': 10,
        'N2_0': 15},
    {
        'r1': 0.4,
        'r2': 0.6,
        'K1': 80,
        'K2': 100,
        'alpha12': 0.7,
        'alpha21': 0.6,
        'N1_0': 10,
        'N2_0': 15
    },
    {
        'r1': 0.3,
        'r2': 0.5,
        'K1': 80,
        'K2': 100,
        'alpha12': 0.9,
        'alpha21': 1.6,
        'N1_0': 10,
        'N2_0': 15
    }
]


# Tiempo
t = np.linspace(0, 50, 1000)# Iterar sobre cada conjunto de parámetros en el array
for params in paremter_dict:
    # Extraer los parámetros
    r1, r2 = params['r1'], params['r2']
    K1, K2 = params['K1'], params['K2']
    alpha12, alpha21 = params['alpha12'], params['alpha21']
    N1_0, N2_0 = params['N1_0'], params['N2_0']

    # Calcular la solución numérica
    X0 = np.array([float(N1_0), float(N2_0)])
    X = runge_kutta_4_system(lotka_volterra, t, X0, args=(r1, r2, K1, K2, alpha12, alpha21))
    N1, N2 = X.T

    # Calcular las isoclinas
    N1_value = np.linspace(0,K1,100)
    N2_value = np.linspace(0,K2,100)
    N1_isocline = (K1-N1_value)/alpha12
    N2_isocline = (K2-N2_value)/alpha21

    # Crear las gráficas

    plt.style.use('seaborn-darkgrid')

    fig , axs = plt.subplots(1,2, figsize=(18,4))

    axs[0].plot(t, N1, label='Especie 1'  , linewidth=2)
    axs[0].plot(t, N2, label='Especie 2', linewidth=2)
    axs[0].set_xlabel('Tiempo' , fontsize=14)
    axs[0].set_ylabel('Población', fontsize=14)
    axs[0].set_title('Modelo de Lotka-Volterra', fontsize=16)
    axs[0].legend()
    
    axs[1].plot(N1_value, N1_isocline, 'r--',label='Iscolina especie 1', linewidth=2)
    axs[1].plot(N2_isocline,N2_value, 'b--', label='Isoclina especie 2', linewidth=2)
    axs[1].legend()
    axs[1].set_xlabel('Población Especie 1', fontsize=14)
    axs[1].set_ylabel('Población Especie 2', fontsize=14)
    axs[1].set_title('Isoclinas', fontsize=16)  

    plt.tight_layout()
    plt.show()