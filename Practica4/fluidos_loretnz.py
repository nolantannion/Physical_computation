import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Parametros y condiciones
x0, y0, z0 = 0,1,1.2

# Integracion
t0 = 0
tf = 100
teval  = np.linspace(t0,tf,10000)
tspan = (t0,tf)

sigma = 10
rho = 28
beta = 8/3

def sistema(t, y):
    dx = sigma * (y[1] - y[0])
    dy = y[0] * (rho - y[2]) - y[1]
    dz = y[0] * y[1] - beta * y[2]
    return [dx, dy, dz]

sol = solve_ivp(sistema,t_span=tspan, t_eval= teval, y0 = [x0,y0,z0], method='RK45')
print(sol.t)

x = sol.y[0]
y = sol.y[1]
z = sol.y[2]


ax = plt.figure().add_subplot(projection = '3d')
ax.plot(x,y,z, lw = 0.5)


plt.show()


