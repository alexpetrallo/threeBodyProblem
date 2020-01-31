#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:33:42 2020

@author: aliciaeglin

Name: Alicia Eglin
Homework #X
Due date
File name
Comments:
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

G = 6.67430e-11
M1= 1.989e+30*5.15#kg
M2= 1.989e+30 #kg
M= M1+M2
m3= 10 #kg
a = 1 #m
q=M2/M1

#1-Dimension
def phi1(x):
    r1 = np.sqrt(x**2)
    r2 = np.sqrt((x-a)**2)
    return -(((2/(1+q))*(1/r1))+(((2*q)/(1+q))*(1/r2))+((x-(q/(1+q)))**2))

#def force(x):
    #return -np.gradient(x)

x_1 = np.array(np.linspace(-2, 2, 500))
#x_1 = np.array(np.linspace(0, 1, 60))
y_1 = np.array(np.linspace(-2, 2, 500))
z_1 = np.array(np.linspace(-2, 2, 500))

phi1=phi1(x_1)
#ph1=phi1(x_3d, y_3d, z_3d)
#force1=force(phi1)
size=np.linspace(-2, 2, 500)

fig = plt.figure()
plt.plot(size, phi1, label= "Potential")
#plt.scatter(phi2[0], phi2[1], label="phiY and phiX")
plt.legend(loc="lower left",fontsize=14)
plt.title('Dimensionless Potential and Force in 1-D')
plt.xlabel('Dimensionless X Position')
plt.ylabel('Phi(X) and F(x)')
#plt.text()
plt.xlim(-2, 2)
plt.ylim(-30,5)
plt.text(0.6292585170340681, 0, 'L1')
plt.text(1.4068136272545089, 0, 'L2')
plt.text(-0.7094188376753507, 0, 'L3')

def force(x): #Force in 1-D - this is actually the magnitude of the force
    r1 = np.sqrt((x**2))
    r2 = np.sqrt(((x-a)**2))
    #return -np.abs(((-2)/((1+q)*(x**2)))+((2*q)/((1+q)*((1-x)**2)))+(2*(x-((q)/(1+q)))))
    #return -np.abs(((-2)/((1+q)*(x**2)))+((-2*q)/((1+q)*((x)**2)))+(2*(x-((q)/(1+q)))))
    return -np.abs(-((2*x)/((1+q)*((r1)**(3/2))))-(((2*q)*(x-a))/((1+q)*((r2)**(3/2))))+(2*(x-((q)/(1+q)))))
force1=force(x_1)
plt.plot(size, force1, label="Force")
plt.legend(loc="lower left",fontsize=14)

#2-Dimension
def phi2(x, y):
    r1 = np.sqrt((x**2)+(y**2))
    r2 = np.sqrt(((x-a)**2)+(y**2))
    return  -(((2/(1+q))*(1/r1))+(((2*q)/(1+q))*(1/r2))+((x-(q/(1+q)))**2)+(y**2))

x_2d, y_2d = np.meshgrid(x_1, y_1)


phi2=phi2(x_2d,y_2d)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1)
levels = np.linspace(-10, 5 ,100)
ct = plt.contour(x_2d, y_2d, phi2, levels=levels)
cbar = plt.colorbar(ct)
cbar.ax.get_yaxis().labelpad = 18
cbar.ax.set_ylabel('Potential', rotation = 270)
plt.xlim(-2, 2)
ax.set_title('Dimensionless Potential and Force in 2-D')
ax.set_xlabel(' Dimensionless X Position ')
ax.set_ylabel(' Dimensionless Y Position')


#plt.text(0.4929859719438876, 0.8697394789579156, 'L4')
#plt.text(0.4929859719438876, -0.8697394789579156, 'L5')
plt.show()

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
'''
def force_x(x):
    #return -np.abs(((-2)/((1+q)*(x**2)))+((2*q)/((1+q)*((1-x)**2)))+(2*(x-((q)/(1+q)))))
    #return -np.abs(((-2)/((1+q)*(x**2)))+((-2*q)/((1+q)*((x)**2)))+(2*(x-((q)/(1+q)))))
    return (-((2*x)/((1+q)*((x**2)**(3/2))))-(((2*q)*(x-a))/((1+q)*(((a-x)**2)**(3/2))))+(2*(x-((q)/(1+q)))))

def force_y(y):
    return (-((2*y)/((1+q)*((y**2)**(3/2))))-((2*q*y)/((1+q)*((y**2)**(3/2))))+(2*y))
'''
force2_x=force2x(x_2d, y_2d)
force2_y=force2y(x_2d, y_2d)
#fig = plt.figure()
force_mag = np.sqrt(force2_x**2+force2_y**2) #magnitude of the force in 2d
'''
dummy = force1
minList = []
locationList = []
for i in range(3):
    min = dummy.max()
    minList = np.append(minList, min)
    location = np.where(dummy == min)
    locationList = np.append(locationList, location)
    dummy = dummy[dummy != min]
print(locationList)
print(minList)
'''


#force2 = force2(x_2d,y_2d)
#fig = plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Force Plot in Two Dimensions")
plt.streamplot(x_1, y_1, force2_x, force2_y, color = 'black')

#plt.plot(force2)
#xs = np.linspace(-2, 2, 11)
#ys = np.linspace(-2, 2, 11)

#for x in x_1:
#plt.quiver(x_2d, y_2d, force2_x, force2_y)
#[np.ones(len(y_1))*x
#plt.quiver(force2_x, force2_y, [force2])
#plt.plot(force2_x, force2_y)
#plt.show()

#force2 = np.vstack((force2_x, force2_y))
#force2 = np.transpose(force2)


fig = plt.figure()
cf = plt.contour(x_1, y_1, force_mag, levels=levels)
cbar = plt.colorbar(ct)
cbar.ax.get_yaxis().labelpad = 18
cbar.ax.set_ylabel('Force', rotation = 270)
plt.title('Magnitude of Force in 2-D')
plt.title('Lagrange Points on Plot of Force' )
plt.xlabel(' Dimensionless X Position ')
plt.ylabel(' Dimensionless Y Position')
plt.text(0.6292585170340681, 0, 'L1')
plt.scatter(0.6292585170340681, 0, color = 'k')
plt.text(1.4068136272545089, 0, 'L2')
plt.scatter(1.4068136272545089, 0, color = 'black')
plt.text(-0.7094188376753507, 0, 'L3')
plt.scatter(-0.7094188376753507, 0,  color = 'black')
plt.text(0.4929859719438876, 0.8697394789579156, 'L4')
plt.scatter(0.4929859719438876, 0.8697394789579156,  color = 'black')
plt.text(0.4929859719438876, -0.8697394789579156, 'L5')
plt.scatter(0.4929859719438876, -0.8697394789579156,  color = 'black')

plt.show()

#3_Dimension
'''
def phi3(x, y, z):
    r1 = np.sqrt((x**2)+(y**2)+(z**2))
    r2 = np.sqrt(((x-a)**2)+(y**2)+(z**2))
    #r1 = np.sqrt((x**2)+(y**2)+(z**2))
     #r2 = np.sqrt(((x-a)**2)+(y**2)+(z**2))
    return - -(((2/(1+q))*(1/r1))+(((2*q)/(1+q))*(1/r2))+((x-(q/(1+q)))**2)+(y**2))

x_3d, y_3d, z_3d = np.meshgrid(x_1, y_1, z_1)
phi3 = phi3(x_3d, y_3d, z_3d)

n=M2/(M1+M2)
L1= a*(1-((n/3)**.33))
L2= a*(1+((n/3)**.33))
L3=-a*(1-(((5*n)/12)**.33))



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x_1, y_1, ph3, 100, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("3D Countour Plot")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(x_3d, y_3d, z_3d, c=phi3, cmap=plt.hot())
fig.colorbar(img)
plt.show()
'''
def threebody(t, y):
    # y[0] = x, y[1] = y, y[2] = z, y[3] = vx, y[4] = vy, y[5] = vz
    #mu = ((1.488*(10**-34))*(1.989*(10**30)))
    #mu = 1.488e-34 * 1.989e30

    r1 = np.sqrt(y[0]**2 + y[1]**2)
    r2 = np.sqrt(((y[0]-a)**2)+(y[1]**2))

    #ydot = np.empty((4,))
    #ydot[:] = np.nan
    ydot = np.empty(4)

    ydot[0] = y[2]
    ydot[1] = y[3]
   # ydot[2] = y[5]
    ydot[2] = -(-((2*y[0])/((1+q)*((r1)**(3/2))))-(((2*q)*(y[0]-a))/((1+q)*((r2)**(3/2))))+(2*(y[0]-((q)/(1+q)))))
    ydot[3] = -(-((2*y[1])/((1+q)*((r1)**(3/2))))-((2*q*y[1])/((1+q)*((r2)**(3/2))))+(2*y[1]))
    #ydot[5] = -np.abs(-((2*y[2])/((1+q)*((y[2]**2)**(3/2))))-((2*q*y[2])/((1+q)*((y[2]**2)**(3/2))))

    return ydot

Y0 = np.array([0.4929859719438876, 0.8697394789579156 , 0 , 0])

integrator_rk45 = ode.solve_ivp(fun=threebody, t_span=(0,(50)), y0=Y0, method='RK45')#, first_step=1.)#, first_step=1., max_step=1., atol=0, rtol=1e-6)
data_rk45 = np.transpose(integrator_rk45.y)
x_data_rk45 = data_rk45[:,0]
y_data_rk45 = data_rk45[:,1]

#print(data)

# plt.plot(x_data_rk45, y_data_rk45, label='RK45')
# plt.scatter(0.4929859719438876, 0.8697394789579156, color='black')
# plt.scatter(0.4929859719438876, -0.8697394789579156, color='black')
# plt.ylim(0, 1.5)
# plt.show()

'''
ax = fig.add_subplot(111)
ax.set_aspect(1)
x_pos_norm = x_data_rk45 / a
y_pos_norm = y_data_rk45 / a

ax.plot(x_pos_norm, y_pos_norm)
ax.plot(0,0, 'ko')
ax.plot(a, 0, 'ro')
qm = '{}'.format(q)
ax.set_title('Orbit of Tertiary Body q = ' + qm )
ax.set_xlabel('X Position [AU]')
ax.set_ylabel('Y Position [AU]')
'''
eta = M2/(M)
L1 = a*(1-(eta/3)**(1/3))
plt.scatter(L1, 0, label = 'L1 (equation)')
L2 = a*(1+(eta/3)**(1/3))
plt.scatter(L2, 0, label = 'L2 (equation)')
L3 = -a*(1-((5*eta)/12))
plt.scatter(L3, 0, label = 'L3 (equation)')
L4x = a*((1/2)*((M1-M2)/(M)))
L4y = a*np.sqrt(3)/2
plt.scatter(L4x, L4y, label = 'L4 (equation)')
L5x = a*((1/2)*((M1-M2)/(M)))
L5y = a*-np.sqrt(3)/2
plt.scatter(L5x, L5y, label = 'L5 (equation)')
# plt.show()
