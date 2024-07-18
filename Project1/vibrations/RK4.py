import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

def F(t):
	F = np.array([0.0,0.0])
	if t <= 20:   #apply force for a set amount of time
		F[0] = F0 * np.cos(omega*t)
	else:
		F[0] = 0.0
	return F

def G(y,t):
	return A_inv.dot( F(t) - B.dot(y) )

def RK4_step(y, t, dt):
    k1 = G(y,t)
    k2 = G(y + 0.5*k1*dt, t + 0.5*dt)
    k3 = G(y + 0.5*k2*dt, t + 0.5*dt)
    k4 = G(y + k3*dt, t + dt)
    
    #return dt * G(y,t)  #RK1 SDOF integrator
    return dt * (k1 + 2*k2 + 2*k3 + k4)/6
	
# variables
m = 2.0
c = 0.0    #critial damping = 2 * sqrt(m*k) = 4.0
k = 2.0

F0 = 0.0
delta_t = 0.1
omega = 1.0
time = np.arange(0.0, 40.0, delta_t)

#intial state
y = np.array([0,1])     #[velocity, displacement]

A = np.array([[m,0],[0,1]])
B = np.array([[c,k],[-1,0]])
A_inv = inv(A)


Y = []
force = []

#time stepping solution
for t in time:
	
	y = y + RK4_step(y, t, delta_t)
	Y.append(y[1])
	force.append(F(t)[0])

	KE = 0.5 * m * y[0]**2
	PE = 0.5 * k * y[1]**2

	if t % 1 <= 0.01:
		print('Total Energy:', KE+PE)

#plotting results

plt.plot(time, Y)
plt.plot(time, force)
plt.grid(True)
plt.legend(['Displacement', 'Force'], loc='lower right')
plt.show()
print('Critical Damping', np.sqrt((-c**2 + 4*m*k / (2*m))))
print('Natural Frequency', np.sqrt(k/m))


