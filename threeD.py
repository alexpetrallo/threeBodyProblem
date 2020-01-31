# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 01:14:14 2020

@author: kpurc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 23:51:51 2020

@author: kpurc
"""


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


G = 1
M1= 2 #kg
M2= 1 #kg
M= M1+M2
a = 1 #m
q=M2/M1

#dimensionless
def phi1(x,y):
    r=(x_1**2+y**2)**.5                #one might be x/a
    return -((a/r)+(q/(((r/a)**2)-((2*x)/a)+((1)**2))**.5)+(((q+1)*r**2)/(2*a**2)))
            #a/r might be a/x

x_1 = np.array(np.linspace(-2, 2.5, 100))
y_1 = np.array(np.linspace(-2, 2.5, 100))
#z_1 = np.array(np.linspace(-2, 2.5, 100))

x_3d, y_3d = np.meshgrid(x_1, y_1)

ph1=phi1(x_3d, y_3d)

n=M2/(M1+M2)
L1= a*(1-((n/3)**.33))
L2= a*(1+((n/3)**.33))
L3=-a*(1-(((5*n)/12)**.33))


def force(x): #Force in 1-D - this is actually the magnitude of the force
    r1 = np.sqrt((x**2))
    r2 = np.sqrt(((x-a)**2))
    #return -np.abs(((-2)/((1+q)*(x**2)))+((2*q)/((1+q)*((1-x)**2)))+(2*(x-((q)/(1+q)))))
    #return -np.abs(((-2)/((1+q)*(x**2)))+((-2*q)/((1+q)*((x)**2)))+(2*(x-((q)/(1+q)))))
    #eqtn force
    return -np.abs(-((2*x)/((1+q)*((r1)**(3/2))))-(((2*q)*(x-a))/((1+q)*((r2)**(3/2))))+(2*(x-((q)/(1+q)))))
force1=force(x_1)

def force2x(x,y): #x- component of the force in 2D
    r1 = np.sqrt((x**2)+(y**2))
    r2 = np.sqrt(((x-a)**2)+(y**2))
    #return -np.abs(((-2)/((1+q)*(x**2)))+((2*q)/((1+q)*((1-x)**2)))+(2*(x-((q)/(1+q)))))
    #return -np.abs(((-2)/((1+q)*(x**2)))+((-2*q)/((1+q)*((x)**2)))+(2*(x-((q)/(1+q)))))
    return -(-((2*x)/((1+q)*((r1)**(3/2))))-(((2*q)*(x-a))/((1+q)*((r2)**(3/2))))+(2*(x-((q)/(1+q)))))

def force2y(x,y): #y- component of the force in 2D
    r1 = np.sqrt((x**2)+(y**2))
    r2 = np.sqrt(((x-a)**2)+(y**2))
    return -(-((2*y)/((1+q)*((r1)**(3/2))))-((2*q*y)/((1+q)*((r2)**(3/2))))+(2*y))
x_2d, y_2d = np.meshgrid(x_1, y_1)
force2_x=force2x(x_2d, y_2d)
force2_y=force2y(x_2d, y_2d)


force_mag = np.sqrt(force2_x**2+force2_y**2)

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.contour3D(x_1, y_1, ph1, 100, cmap='twilight')
ax.contour3D(x_1, y_1, ph1, 100, cmap = "twilight")
ax.set_xlabel('Dimensionless X Position')
ax.set_ylabel('Dimensionless Y Position')
ax.set_zlabel('Potential')
ax.set_title("3D Contour Plot of the Potential")
plt.show()
