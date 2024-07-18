import numpy as np
from numpy import sin, cos
from scipy.integrate import odeint
from matplotlib import pyplot as plt

# define the equations
def eq(y0, t):
    theta, x = y0 
    f = [x, -(g/l) * sin(theta)]
    return f

def plot_results(time, theta1, theta2):
    plt.plot(time, theta1[:,0])
    plt.plot(time, theta2)

    s = '(i_angle = ' + str(i_angle) + ' degrees)'
    plt.title('Simple Pendulum: ' + s)
    plt.xlabel('time (s)')
    plt.ylabel('angle (rad)')
    plt.grid(True)
    plt.legend(['nonlinear', 'linear'], loc='lower right')
    plt.show()

# prarameters
g = 9.81
l = 1.0
time = np.arange(0, 10.0, 0.025)

# initial conditons
i_angle = 130.0
theta0 = np.radians(i_angle)
x0 = np.radians(0.0)

# solution for nonlinear problem
theta1 = odeint(eq, [theta0, x0], time)

# solution to linear problem
w = np.sqrt(g/l)
theta2 = [theta0 * cos(w*t) for t in time]

# plotting results
plot_results(time, theta1, theta2)
