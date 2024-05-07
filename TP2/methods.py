import numpy as np

def runge_kutta_4(f, t_values, y0):
    num_steps = len(t_values) - 1
    h = t_values[1] - t_values[0]  # assuming uniform step size
    y_values = np.zeros(num_steps + 1)
    
    y_values[0] = y = y0
    
    for i in range(1, num_steps + 1):
        t = t_values[i-1]
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        
        y_values[i] = y
    
    return y_values

def euler_method(f, t_values, y0):
    num_steps = len(t_values) - 1
    y_values = np.zeros(num_steps+1)
    y_values[0] = y0

    for i in range(1, num_steps+1):
        h = t_values[i] - t_values[i-1]  # calculate step size
        y_values[i] = y_values[i-1] + h * f(t_values[i-1], y_values[i-1])

    return y_values

def runge_kutta_4_system(f, t_values, y0, args=()):
    num_steps = len(t_values) - 1
    h = t_values[1] - t_values[0]  # assuming uniform step size
    y_values = np.zeros((num_steps + 1, len(y0)))
    
    y_values[0] = y = y0
    
    for i in range(1, num_steps + 1):
        t = t_values[i-1]
        k1 = h * f(t, y, *args)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1, *args)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2, *args)
        k4 = h * f(t + h, y + k3, *args)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        
        y_values[i] = y
    
    return y_values