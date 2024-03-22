from errorFa import exp, equispaced, RegularGridInterpolator,RectBivariateSpline, np

def originalFunction(x1:int, x2:int):
    return 0.75*exp( (-((10*x1-2)**2)/4)-((9*x2 - 2)**2)/4 ) + 0.65*exp( (-((9*x1+1)**2)/9)-((10*x2 +1 )**2)/2 )+0.55*exp( (-((9*x1-6)**2)/4)-((9*x2 - 3)**2)/4 )-0.01*exp( (-((9*x1-7)**2)/4)-((9*x2 - 3)**2)/4 )

x1_original_cords = equispaced(True,-1,1,10)
x2_original_cords = equispaced(True,-1,1,10)

z_original_cords = []

for i in range(len(x1_original_cords)):
    z_original_cords.append(originalFunction(x1_original_cords[i],x2_original_cords[i]))

linearInterpolation = RegularGridInterpolator((x1_original_cords,x2_original_cords),z_original_cords,method ="linear")
splinesInterpolation = RectBivariateSpline(x1_original_cords,x2_original_cords,z_original_cords)

x1_interpolate = np.arange(min(x1_original_cords+x2_original_cords) , max((x1_original_cords+x2_original_cords)), 20)
x2_interpolate = np.arange(min(x1_original_cords+x2_original_cords) , max((x1_original_cords+x2_original_cords)), 20)

z_original_function=[]
z_linear = []
z_splines = []

for i in range(len(x1_interpolate)):
    z_original_cords.append(originalFunction(x1_interpolate[i],x2_interpolate))
    z_linear.append(linearInterpolation((x1_interpolate,x2_interpolate)))
    z_splines.append(splinesInterpolation(x1_interpolate[i],x2_interpolate[i]))

print(z_original_cords)
print(z_linear)
print(z_splines)






