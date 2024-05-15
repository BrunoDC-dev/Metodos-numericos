import numpy as np
import matplotlib.pyplot as plt
from methods import runge_kutta_4_system , runge_kutta_4
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
r = 7 # Tasa de crecimiento de la presa
alpha = 0.1  # Eficiencia de captura
beta = 0.05 # Eficiencia de conversión de presas en predadores
q = 0.5  # Tasa de mortalidad de los predadores
K = 50  # Capacidad de carga (sólo para LVE)


parameter_dict = [
    {
        'r': 7,
        'alpha': 0.1,
        'beta': 0.05,
        'q': 0.5,
        'K': 50,
        'N0': 15,
        'P0': 10

    },
    {
        'r': 1,
        'alpha': 0.4,
        'beta': 0.2,
        'q': 0.5,
        'K': 50,
        'N0': 15,
        'P0': 10
    },
    {
        'r': 1,
        'alpha': 0.1,
        'beta': 0.05,
        'q': 0.5,
        'K': 50,
        'N0': 15,
        'P0': 10
    },

]

# Tiempo
t = np.linspace(0, 80, 800)

plt.style.use('seaborn-darkgrid')

# Iterar sobre cada conjunto de parámetros en el diccionario
for params in parameter_dict:
    # Extraer los parámetros
    r = params['r']
    alpha = params['alpha']
    beta = params['beta']
    q = params['q']
    K = params['K']
    N0 = params['N0']
    P0 = params['P0']

    # Condiciones iniciales
    X0 = np.array([float(N0), float(P0)])

    # Soluciones numéricas
    X = runge_kutta_4_system(lotka_volterra, t, X0, args=(r, alpha, beta, q))
    N, P = X.T

    X0_ext = np.array([float(N0), float(P0)])
    X_ext = runge_kutta_4_system(lotka_volterra_extendido, t, X0_ext, args=(r, alpha, beta, q, K))
    N_ext, P_ext = X_ext.T

    def NO_isocline(N0 ,t=t , r=r, alpha=alpha, K=K):
        return r / alpha

    def P_isocline(P0 ,t=t , q=q, beta=beta, K=K):
        return q/beta

    def NO_ext_isocline(N0_ext ,t=t , r=r, alpha=alpha, K=K):
        return (r / alpha )* (1- (N0_ext/K))

    # Crear las gráficas
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    axs[0, 0].plot(t, N, label='Presa', linewidth=2)
    axs[0, 0].plot(t, P, label='Predador', linewidth=2)
    axs[0, 0].set_xlabel('Tiempo', fontsize=14)
    axs[0, 0].set_ylabel('Población', fontsize=14)
    axs[0, 0].set_title('Modelo de Lotka-Volterra', fontsize=16)
    axs[0, 0].legend()

    axs[0, 1].plot(N, P, label='Presa vs Predador', linewidth=2)
    axs[0, 1].set_xlabel('Presa', fontsize=14)
    axs[0, 1].set_ylabel('Predador', fontsize=14)
    axs[0, 1].legend()
    axs[0, 1].set_title('Modelo de Lotka-Volterra', fontsize=16)

    axs[1, 0].plot(t, N_ext, label='Presa (LVE)', linewidth=2)
    axs[1, 0].plot(t, P_ext, label='Predador (LVE)', linewidth=2)
    axs[1, 0].set_xlabel('Tiempo', fontsize=14)
    axs[1, 0].set_ylabel('Población', fontsize=14)
    axs[1, 0].set_title('Modelo de Lotka-Volterra Extendido', fontsize=16)
    axs[1, 0].legend()

    axs[1, 1].plot(N_ext, P_ext, label='Presa vs Predador (LVE)', linewidth=2)
    axs[1, 1].set_xlabel('Presa', fontsize=14)
    axs[1, 1].set_ylabel('Predador', fontsize=14)
    axs[1, 1].legend()
    axs[1, 1].set_title('Modelo de Lotka-Volterra Extendido', fontsize=16)

    plt.tight_layout()
    plt.show()

    N_range = np.linspace(0, 50, 50)
    P_range = np.linspace(0, 50, 50)
    NO_isocline_value = np.array([NO_isocline(t) for t in N_range])
    P_isocline_value = np.array([P_isocline(t) for t in P_range])
    NO_ext_isocline_value = np.array([NO_ext_isocline(t) for t in N_range])

    # Graficar isoclinas para el modelo de Lotka-Volterra
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    axs[0].plot(N_range, NO_isocline_value, 'r--', label='Isoclina dN/dt = 0', linewidth=2)
    axs[0].plot(P_isocline_value, P_range, 'g--', label='Isoclina dP/dt = 0', linewidth=2)
    axs[0].set_xlabel('Presa', fontsize=14)
    axs[0].set_ylabel('Predador', fontsize=14)
    axs[0].set_title('Isoclinas - Modelo de Lotka-Volterra', fontsize=16)
    axs[0].legend()

    axs[1].plot(N_range, NO_ext_isocline_value, 'r--', label='Isoclina dN/dt = 0', linewidth=2)
    axs[1].plot(P_isocline_value, P_range, 'g--', label='Isoclina dP/dt = 0', linewidth=2)
    axs[1].set_xlabel('Presa', fontsize=14)
    axs[1].set_ylabel('Predador', fontsize=14)
    axs[1].set_title('Isoclinas - Modelo de Lotka-Volterra Extendido', fontsize=16)
    axs[1].legend()

    plt.tight_layout()
    plt.show()





