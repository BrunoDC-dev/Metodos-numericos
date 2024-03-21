import sympy as sp
import math
from scipy.interpolate import CubicSpline

def lagrange_interpolation_function(points):
    x = sp.symbols('x')
    L = 0
    for i in range(len(points)):
        term = points[i][1]
        for j in range(len(points)):
            if j != i:
                term *= (x - points[j][0]) / (points[i][0] - points[j][0])
        L += term
    return sp.simplify(L)


def lagrange_interpolation_evaluation(points, x):
    result = 0
    for i in range(len(points)):
        term = points[i][1]
        for j in range(len(points)):
            if j != i:
                term = term * (x - points[j][0]) / (points[i][0] - points[j][0])
        result += term
    return result

def linear_interpolation(x1, y1, x2, y2, x):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)



def cubic_spline_interpolation(x_points, y_points, x):
    cs = CubicSpline(x_points, y_points)
    return cs(x)
