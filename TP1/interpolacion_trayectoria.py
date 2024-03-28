import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Load data
mediciones1_data = pd.read_csv('mnyo_mediciones.csv', header=None, sep='\s+')
ground_truth_data = pd.read_csv('mnyo_ground_truth.csv', header=None, sep='\s+')


# Create an array of equispaced time points
time_points_mediciones1 = np.linspace(0, 9, mediciones1_data.shape[0])
time_points_ground_truth = np.linspace(0, 9, ground_truth_data.shape[0])

# Interpolate the positions using cubic splines
mediciones1_interpolator_x = CubicSpline(time_points_mediciones1, mediciones1_data.iloc[:, 0])
mediciones1_interpolator_y = CubicSpline(time_points_mediciones1, mediciones1_data.iloc[:, 1])

ground_truth_interpolator_x = CubicSpline(time_points_ground_truth, ground_truth_data.iloc[:, 0])
ground_truth_interpolator_y = CubicSpline(time_points_ground_truth, ground_truth_data.iloc[:, 1])

# Interpolate the positions at the common time points
m1x1_interpolated = mediciones1_interpolator_x(time_points_ground_truth)
m1x2_interpolated = mediciones1_interpolator_y(time_points_ground_truth)

# Calculate the absolute error
absolute_error = np.abs(np.sqrt(m1x1_interpolated**2 + m1x2_interpolated**2) - 
                        np.sqrt(ground_truth_data.iloc[:, 0]**2 + ground_truth_data.iloc[:, 1]**2))

# Calculate the average absolute error
average_error = np.mean(absolute_error)

# Print the average absolute error
print('Error promedio absoluto:', average_error)

# Plot the absolute error
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-darkgrid')
plt.plot(time_points_ground_truth, absolute_error.to_numpy(), label='Error absoluto de la norma')
plt.xlabel('Tiempo')
plt.ylabel('Error absoluto de la norma')
plt.legend()
plt.show()

# Plot the interpolated trajectory and the ground truth
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-darkgrid')
plt.plot(m1x1_interpolated, m1x2_interpolated, label='Trayectoria interpolada')
plt.plot(ground_truth_data.iloc[:, 0].to_numpy(), ground_truth_data.iloc[:, 1].to_numpy(), label='Ground truth')
plt.scatter(mediciones1_data.iloc[:, 0], mediciones1_data.iloc[:, 1], color='blue', s=10, label='Mediciones 1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()