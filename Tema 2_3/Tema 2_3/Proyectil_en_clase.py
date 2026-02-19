import numpy as np
import matplotlib.pyplot as plt


# Constantes y etc
g = 9.81
t0 = 0
B2 = 1e-5
m = 10

dt = 1e-3


# Euler con rozamiento 
def euler_roz(t0,vars,y0):

    # Vars = [x,y,vx,vy]
    dvars = []

    dvars[0] = vars[2]
    dvars[1] = vars[3]

    v = np.sqrt(vars[2]**2 + vars[3]**2)

    dvars[2] = vars[2] - B2*vars[2]*v/m*dt
    dvars[3] = vars[3] - (g + B2*vars[3]*v /m)*dt

    tm = t0 + dt

    return tm, dvars
