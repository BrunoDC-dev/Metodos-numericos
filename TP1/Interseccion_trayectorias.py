import pandas as pd
import numpy as np 
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Load data
mediciones1_data = pd.read_csv('mnyo_mediciones.csv', header=None, sep='\s+')
ground_truth_data = pd.read_csv('mnyo_ground_truth.csv', header=None, sep='\s+')
mediciones2_data = pd.read_csv('mnyo_mediciones2.csv', header=None, sep='\s+')

# Create an array of equispaced time points
time_points_mediciones1 = np.arange(1, mediciones1_data.shape[0] + 1)
time_points_mediciones2= np.arange(1, mediciones2_data.shape[0] + 1)
time_points_ground_truth = np.arange(1, ground_truth_data.shape[0] + 1)

# Interpolate the positions using cubic splines
mediciones1_interpolator_x = CubicSpline(time_points_mediciones1, mediciones1_data.iloc[:, 0])
mediciones1_interpolator_y = CubicSpline(time_points_mediciones1, mediciones1_data.iloc[:, 1])

mediciones2_interpolator_x = CubicSpline(time_points_mediciones2, mediciones2_data.iloc[:, 0])
mediciones2_interpolator_y = CubicSpline(time_points_mediciones2, mediciones2_data.iloc[:, 1])

ground_truth_interpolator_x = CubicSpline(time_points_ground_truth, ground_truth_data.iloc[:, 0])
ground_truth_interpolator_y = CubicSpline(time_points_ground_truth, ground_truth_data.iloc[:, 1])

# Interpolate the positions at the common time points
common_time_points_mediciones1 = np.linspace(time_points_mediciones1.min(), time_points_mediciones1.max(), num=1000)
m1x1_interpolated = mediciones1_interpolator_x(common_time_points_mediciones1)
m1x2_interpolated = mediciones1_interpolator_y(common_time_points_mediciones1)

common_time_points_mediciones2 = np.linspace(time_points_mediciones2.min(), time_points_mediciones2.max(), num=1000)
m2x1_interpolated = mediciones2_interpolator_x(common_time_points_mediciones2)
m2x2_interpolated = mediciones2_interpolator_y(common_time_points_mediciones2)

common_time_points_ground_truth = np.linspace(time_points_ground_truth.min(), time_points_ground_truth.max(), num=1000)

ground_truth_x1_interpolated = ground_truth_interpolator_x(common_time_points_ground_truth)
ground_truth_x2_interpolated = ground_truth_interpolator_y(common_time_points_ground_truth)

# Derivatives
mediciones1_interpolator_x_derivative = mediciones1_interpolator_x.derivative()
mediciones1_interpolator_y_derivative = mediciones1_interpolator_y.derivative()

mediciones2_interpolator_x_derivative = mediciones2_interpolator_x.derivative()
mediciones2_interpolator_y_derivative = mediciones2_interpolator_y.derivative()

ground_truth_interpolator_x_derivative = ground_truth_interpolator_x.derivative()
ground_truth_interpolator_y_derivative = ground_truth_interpolator_y.derivative()

# Intersection functions
def intersection_mediciones1_mediciones2_x(s1, s2):
    return mediciones1_interpolator_x(s1) - mediciones2_interpolator_x(s2)

def intersection_mediciones1_mediciones2_y(s1, s2):
    return mediciones1_interpolator_y(s1) - mediciones2_interpolator_y(s2)

def intersection_ground_truth_mediciones2_x(s1, s2):
    return ground_truth_interpolator_x(s1) - mediciones2_interpolator_x(s2)

def intersection_ground_truth_mediciones2_y(s1, s2):
    return ground_truth_interpolator_y(s1) - mediciones2_interpolator_y(s2)

# Newton's method for finding intersection points
def newton_method_intersection(fx1, fx2, s1, s2, f1x1d, f1x2d ,f2x1d, f2x2d, max_iterations=1000, tol=0.005):
    for i in range(max_iterations):
        # Evaluate the function and its Jacobian at s1, s2
        f_values = np.array([fx1(s1, s2), fx2(s1, s2)])
        jacobian = np.array([[f1x1d(s1), -f2x1d(s2)],
                             [f1x2d(s1), -f2x2d(s2)]])

        # Solve the linear system to get the increment in s1, s2
        delta_s = np.linalg.solve(jacobian, f_values)

        # Update s1, s2
        s1 -= delta_s[0]
        s2 -= delta_s[1]

        # Check convergence
        if np.linalg.norm(delta_s) < tol:
            print("Converged to the desired tolerance.")
            break

    else:
        print("Maximum number of iterations reached.")

    return s1, s2

# Find intersection points
s1, s2 = newton_method_intersection(intersection_mediciones1_mediciones2_x, intersection_mediciones1_mediciones2_y, 1, 1, mediciones1_interpolator_x_derivative, mediciones1_interpolator_y_derivative, mediciones2_interpolator_x_derivative, mediciones2_interpolator_y_derivative)
x1, y1 = mediciones1_interpolator_x(s1), mediciones1_interpolator_y(s1)
x2, y2 = mediciones2_interpolator_x(s2), mediciones2_interpolator_y(s2)

print("Interseccion entre Mediciones 1 y Mediciones 2")
print('s1 =', s1, 's2 =', s2)
print('x1 =', x1, 'y1 =', y1)
print('x2 =', x2, 'y2 =', y2)

t1, t2 = newton_method_intersection(intersection_ground_truth_mediciones2_x, intersection_ground_truth_mediciones2_y, 15, 3, ground_truth_interpolator_x_derivative, ground_truth_interpolator_y_derivative, mediciones2_interpolator_x_derivative, mediciones2_interpolator_y_derivative)
gx1, gy1 = ground_truth_interpolator_x(t1), ground_truth_interpolator_y(t1)
mx2, my2 = mediciones2_interpolator_x(t2), mediciones2_interpolator_y(t2)

print("Interseccion entre Ground Truth y Mediciones 2")
print('t1 =', t1, 't2 =', t2)
print('x1 =', gx1, 'y1 =', gy1)
print('x2 =', mx2, 'y2 =', my2)

# Calculate absolute error of the norm
norm_abs_error= np.abs(np.sqrt(x1**2 + y1**2) - np.sqrt(gx1**2 + gy1**2))
print('Error absoluto de la norma:', norm_abs_error)

# Plotting
plt.rcParams.update({'font.size':18})

plt.figure(figsize=(12, 8))
plt.style.use('seaborn-darkgrid')  
plt.plot(m1x1_interpolated, m1x2_interpolated, label='Trayectoria Interpolada 1')
plt.plot(m2x1_interpolated, m2x2_interpolated, label='Trayectoria Interpolada 2')
plt.plot(ground_truth_data.iloc[:, 0].to_numpy(), ground_truth_data.iloc[:, 1].to_numpy(), label='Ground truth')

plt.scatter(mediciones1_data.iloc[:, 0], mediciones1_data.iloc[:, 1], color='blue', s=20, label='Mediciones 1')
plt.scatter(mediciones2_data.iloc[:, 0], mediciones2_data.iloc[:, 1], color='purple', s=20, label='Mediciones 2')

plt.scatter(x1, y1, color='red', s=100, label='Punto de Interseccion')
plt.text(0.0, 0.95, f"(Tn, T'n) = ({s1:.2f}, {s2:.2f})\nPunto de intersección = ({x1:.2f}, {y1:.2f})", 
         transform=plt.gca().transAxes, verticalalignment='top')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()