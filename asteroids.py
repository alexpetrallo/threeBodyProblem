#will plot the Greek and Trojan asteroids around Jupiter L4 and L5 pts
import numpy as np
import scipy.integrate as ode
import matplotlib.pyplot as plt
#from pyodeint import integrate_adaptive

G = 6.67430e-11
M1= 1.989e+30*1000 #kg
M2= 1.989e+30 #kg
M= M1+M2
m3= 10 #kg
a = 1 #m
q=M2/M1

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

Y0 = np.array([0.16666666666666666, 0.8660254037844386 , .05, 0])

#
'''
t0 = 0
tend = 27492
dt0=1e-8
atol=1e-8
rtol=1e-8

tout, yout, info = integrate_adaptive(twoBody, y0=Y0, t0, tend, nsteps=1000, dt0, atol, rtol, method='bulirsh_stoer')
#tout, yout, info = integrate_adaptive(twoBody,, y0, t0, tend, dt0, atol, rtol,method='bs', nsteps=1000)

'''



#integrate with really small step to make the orbit more accurate
integrator_rk45 = ode.solve_ivp(fun=threebody, t_span=(0,(100)), y0=Y0, method='RK45', first_step=1., max_step=.005)## atol=0, rtol=1e-6)
data_rk45 = np.transpose(integrator_rk45.y)
x_data_rk45 = data_rk45[:,0]
y_data_rk45 = data_rk45[:,1]
#print(data)
''''
plt.plot(x_data_rk45, y_data_rk45, label='RK45')
plt.scatter(0.4929859719438876, 0.8697394789579156, color='black')
plt.scatter(0.4929859719438876, -0.8697394789579156, color='black')
plt.ylim(0, 1.5)
'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1)
x_pos_norm = x_data_rk45 / a
y_pos_norm = y_data_rk45 / a

ax.plot(x_pos_norm, y_pos_norm)
ax.plot(0,0, 'ko', label = "Sun")
ax.plot(a, 0, 'ro', label = "Jupiter")
qm = '{}'.format(q)
ax.set_title('Orbit of Tertiary Body q = ' + qm )
ax.set_xlabel('X Position [AU]')
ax.set_ylabel('Y Position [AU]')
plt.legend()
eta = M2/(M)
L1 = a*(1-(eta/3)**(1/3))
plt.scatter(L1, 0, color = 'b')
L2 = a*(1+(eta/3)**(1/3))
plt.scatter(L2, 0, color = 'b')
L3 = -a*(1-((5*eta)/12))
plt.scatter(L3, 0, color = 'b')
L4x = a*((1/2)*((M1-M2)/(M)))
L4y = a*np.sqrt(3)/2
plt.scatter(L4x, L4y, color = 'b')
L5x = a*((1/2)*((M1-M2)/(M)))
L5y = a*-np.sqrt(3)/2
plt.scatter(L5x, L5y, color = 'b')
# plt.xlim(-1, 2)
plt.show()
