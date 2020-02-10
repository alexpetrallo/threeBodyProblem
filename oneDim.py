#one dimension plot

import matplotlib.pyplot as plt
import math
import numpy as np
import scipy as sci
from mpl_toolkits import mplot3d

G = 6.67430e-11
M1= 1.989e+30*2 #kg
M2= 1.989e+30 #kg
M= M1+M2
m3= 10 #kg
a = 1 #m
q=M2/M1

r1_vec=[0,0,0] #ratio of the aplha cen distance to each other
r2_vec=[a,0,0] #




#dimensionless
def phi1(x):
    r=(x_1**2)**.5                #one might be x/a
    return -((a/r)+(q/(((r/a)**2)-((2*x)/a)+((1)**2))**.5)+(((q+1)*r**2)/(2*a**2)))
            #a/r might be a/x
def force(x):
    return -np.gradient(x)

'''
def phi2(x,y):
    return (-G*M1/a)*((a/((x**2+abs(y)**2)**0.5))
            + (q*a/((((abs(x)-a)**2+abs(y)**2))**0.5))
            + (((q+1)*(abs(x)**2 + abs(y)**2)/(2*a**2))))


def phi3(x,y,z):
    r=(x_1**2+y_1**2+z_1)**.5
    return (-G*M1/a)*((a/((abs(x)**2+abs(y)**2+abs(z)**2))**.5)
            +(q*a/(((abs(x)-a)**.5**2+abs(y)**2+abs(z)**2)))
            +(((q+1)*(abs(x)**2 + abs(y)**2 +abs(z)**2))/(2*a**2)))
'''


x_1 = np.array(np.linspace(-1.0025, 2, 500))
y_1 = np.array(np.linspace(-1.0025, 2, 500))
z_1 = np.array(np.linspace(-1.0025, 2, 500))
#xy_1 = np.transpose(np.vstack((x_1,y_1)))
#z_1 = np.array(np.linspace(-1.0025, 2, 60))
#x_3d, y_3d, z_3d = np.meshgrid(x_1, y_1, z_1)
#x_2d, y_2d = np.meshgrid(x_1, y_1)

phi1=phi1(x_1)
#ph1=phi1(x_3d, y_3d, z_3d)
force1=force(phi1)

#phi2=phi2(x_2d,y_2d)
#phi3=phi3(x_1,y_1,z_1)

size=np.linspace(-1.0025, 2, 500)



plt.plot(size, phi1, label="phi")
plt.plot(size, force1, label="force1")
#plt.scatter(phi2[0], phi2[1], label="phiY and phiX")
plt.legend(loc="lower left",fontsize=14)
plt.xlabel('X units (1D) or PhiX (2D)')
plt.ylabel('PhiX (1D) or PhiY(2D)')
plt.show()
