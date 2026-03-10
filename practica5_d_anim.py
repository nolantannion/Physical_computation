import numpy as np
import matplotlib.pyplot as plt
from funciones_t5 import euler_cromer
from matplotlib.animation import FuncAnimation

'''
Script con el calculo de la orbita incluyendo jupiter
 y animacion del movimiento
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
T = 8*75.979

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
    # Añadimos un desfase para que no comiencen todos alineados en la region
    # de mayor error
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





# animación
fig, ax = plt.subplots()

# Sol
ax.scatter(0,0,color="y",label="Sol")

# objetos animados
linea, = ax.plot([],[],lw=1,color="b")
cometa, = ax.plot([],[],'ro',label="Halley")
jupiter, = ax.plot([],[],'go',label="Jupiter")
cj, = ax.plot([],[], color = 'g')


xl = 0.2*np.abs(np.max(xpr) - np.min(xpr))
yl = 0.2*np.abs(np.max(ypr) - np.min(ypr))

ax.set_xlim(np.min(xpr)-xl , np.max(xpr)+xl)
ax.set_ylim(np.min(ypr)-yl , np.max(ypr)+yl)

ax.grid()
ax.set_aspect("equal")
ax.legend()



def init():

    linea.set_data([],[])
    cometa.set_data([],[])
    jupiter.set_data([],[])
    cj.set_data([xxjr],[yyjr])
    ax.plot()

    return linea,cometa,jupiter


def update(i):

    # trayectoria del cometa
    linea.set_data(xpr[:i],ypr[:i])
    cometa.set_data([xpr[i]],[ypr[i]])

    # posición de Jupiter
    xj = Rj*np.cos(omegaj*tpr[i])
    yj = Rj*np.sin(omegaj*tpr[i])

    jupiter.set_data([xj],[yj])

    ax.set_title(f"t = {tpr[i]:.2f} años")

    return linea,cometa,jupiter


ani = FuncAnimation(
    fig,
    update,
    frames=range(0,len(xpr),500),
    init_func=init,
    interval=10,
    blit=True
)

ani.save('Precesion_orbita.mp4', writer='ffmpeg')

plt.show()