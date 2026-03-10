import numpy as np
import matplotlib.pyplot as plt
from funciones_t5 import euler_cromer, loc_max, loc_min
from matplotlib.animation import FuncAnimation

''' 
Script que lleva a cabo el apartado d, calcula y compara la precesion 
al estar o no jupiter y calcula perihelio y afelio en cada orbita 
'''

# Constantes
Ms = 4*np.pi**2
M_sol = 2e30
Mj = 1.9e27

q = Mj/M_sol
GMj = q*Ms

# Jupiter
Rj = 5.2
Tj = 11.86
omegaj = 2*np.pi/Tj

# Halley
r0 = 0.59
vy0 = 11.47270
Periodo = 75.979
T = 8*Periodo

dt = 0.0005


# Sistema Sol
def sistema(t,estado):

    x,vx,y,vy = estado
    r = np.sqrt(x**2+y**2)

    ax = -Ms*x/r**3
    ay = -Ms*y/r**3

    return np.array([vx,ax,vy,ay])


# Sistema Sol + Jupiter
def sistema_j(t,estado):

    x,vx,y,vy = estado

    r = np.sqrt(x**2+y**2)

    # orbita circular de Jupiter
    xj = Rj*np.cos(omegaj*t + 5*np.pi/4)
    yj = Rj*np.sin(omegaj*t)

    dx = x-xj
    dy = y-yj
    rj = np.sqrt(dx**2+dy**2)

    ax = -Ms*x/r**3 - GMj*dx/rj**3
    ay = -Ms*y/r**3 - GMj*dy/rj**3

    return np.array([vx,ax,vy,ay])


# condiciones iniciales
estado0 = np.array([r0,0,0,vy0])


# integración
t, vars = euler_cromer(sistema,0,T,estado0,dt)
tpr, varspr = euler_cromer(sistema_j,0,T,estado0,dt)


# Trayectoria de jupiter
thj = np.linspace(0,2*np.pi, 100)
xxjr = Rj*np.cos(thj)
yyjr = Rj*np.sin(thj)


# soluciones
x = vars[:,0]
y = vars[:,2]

xpr = varspr[:,0]
ypr = varspr[:,2]




# Calculamos la distancia y angulo del afelio

# Sin precesion
r = np.sqrt(x**2 + y**2)
iM, _ = loc_max(r)
# im, _ = loc_min(r)

ttM = t[iM]/Periodo
# ttm = t[im]/Periodo

theta = np.arctan(y[iM]/x[iM])



# Con precesion
rpr = np.sqrt(xpr**2 + ypr**2)
iMpr, _ = loc_max(rpr)
# impr, _ = loc_min(rpr)

ttpM = tpr[iMpr]/Periodo
# ttpm = tpr[impr]/Periodo
thetapr = np.arctan(ypr[iMpr]/xpr[iMpr])



# Calculamos la velocidad de precesion con un ajuste
p = np.polyfit(ttM, theta, 1)
ppr = np.polyfit(ttpM, thetapr, 1)

print(f'Velocidad de precesion:')
print(f'Sin Jupiter: {p[0]:.3e} (rad/a)')
print(f'Con Jupiter: {ppr[0]:.3e} (rad/a)')

# Representamos la evolucion temporal del angulo y distancia 
fig, ax = plt.subplots(nrows= 1, ncols=2, figsize = (8,6))
ax[0].scatter(ttpM, thetapr, label = 'Precesión')
ax[0].scatter(ttM,theta, label = 'Sin precesión')

ax[1].scatter(ttM, r[iM], label = 'Sin precesión')
ax[1].scatter(ttpM, rpr[iMpr], label = 'Con precesión')

ax[0].set_xlabel('t/T')
ax[0].set_ylabel(r'$\theta$ (rad)')
ax[1].set_xlabel('t/T')
ax[1].set_ylabel(r'r (UA)')

ax[0].title.set_text('Ángulo del afelio')
ax[1].title.set_text('Distancia del afelio')

ax[0].legend()
ax[1].legend()
ax[0].grid()
ax[1].grid()






figura, eje = plt.subplots(nrows=1,ncols=2,figsize  = (8,6))

lx = 0.2*np.abs(np.max(x) - np.min(x))
ly = 0.2*np.abs(np.max(y) - np.min(y))

xl = 0.2*np.abs(np.max(xpr) - np.min(xpr))
yl = 0.2*np.abs(np.max(ypr) - np.min(ypr))

eje[0].set_xlim(np.min(x)-lx , np.max(x)+lx)
eje[0].set_ylim(np.min(ypr)-ly , np.max(ypr)+ly)
eje[0].axis('equal')
eje[0].grid()
eje[0].legend()

eje[1].set_xlim(np.min(xpr)-xl , np.max(xpr)+xl)
#eje[1].set_ylim(-lx , +lx)
eje[1].set_ylim(np.min(ypr)-yl , np.max(ypr)+yl)
eje[1].axis('equal')
eje[1].grid()
eje[1].legend()

eje[0].plot(x,y,color = 'b', label = 'Trayectoria Halley')
eje[0].plot(xxjr,yyjr, color = 'green', label = 'Jupiter')
eje[0].scatter(x[iM], y[iM], c = 'r')

eje[1].plot(xpr,ypr,color = 'b', label = 'Trayectoria Halley')
eje[1].plot(xxjr,yyjr, color = 'green', label = 'Jupiter')
eje[1].scatter(xpr[iMpr], ypr[iMpr], c = 'r')


eje[0].set_title('Fuerza Central')
eje[1].set_title('Fuerza No Central')

eje[0].legend(bbox_to_anchor = (1.05,1))
eje[1].legend(bbox_to_anchor = (1.05,1))


# Descomentar para almacenar las imagenes con el titulo indicado
# fig.savefig('th_r.png', dpi = 500)
# figura.savefig('Prec_comp.png', dpi = 500)

plt.show()