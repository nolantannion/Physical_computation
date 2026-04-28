import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Practica6.funciones_t6 import laplace2D

D = 1
deltax = 1
deltat = deltax**2 / (4*D)   # estabilidad en 2D

NX = 50
NY = 50
#NT = 200

tf = 200.0

NT = int(200/deltat)

x = np.arange(-NX, NX+1, deltax)
y = np.arange(-NY, NY+1, deltax)
t = np.arange(0, (NT + 1)*deltat, deltat)

# densidad: (x, y, t)
rho = np.zeros((2*NX+1, 2*NY+1, NT+1))

# condición inicial: delta en el centro
rho[NX, NY, 0] = 1

# evolución temporal
for n in range(NT):
    rho[1:-1,1:-1,n+1] = rho[1:-1,1:-1,n] + (D*deltat/deltax**2)*(
        rho[2:,1:-1,n] + rho[:-2,1:-1,n] +
        rho[1:-1,2:,n] + rho[1:-1,:-2,n] -
        4*rho[1:-1,1:-1,n]
    )

tsol, rho = laplace2D(rho, tf = tf, dt = deltat)


# graficas y animacion
figura, eje = plt.subplots(nrows=1,ncols=3)
eje[0].imshow(rho[:,:,0])
eje[1].imshow(rho[:,:,int(NT/4)])
eje[2].imshow(rho[:,:,NT])

eje[0].set_xlabel('X')
eje[0].set_ylabel('Y')
eje[0].set_title(f't:0s')

eje[1].set_xlabel('X')
eje[1].set_ylabel('Y')
eje[1].set_title(f't:{t[int(NT/4)]}s')

eje[2].set_xlabel('X')
eje[2].set_ylabel('Y')
eje[2].set_title(f't:{t[NT]}s')



fig, ax = plt.subplots()

s = ax.imshow(rho[:,:,0],
              extent=[-NX,NX,-NY,NY],
              origin='lower',
              cmap='hot')

def init():
    s.set_array(rho[:,:,0])
    return (s,)

def update(frame):
    s.set_array(rho[:,:,frame])
    return (s,)

anim = FuncAnimation(fig,
                     update,
                     init_func=init,
                     frames=np.arange(0,NT, 5),
                     interval=30,
                     blit=False)

anim.save('Evolucion_rho_2d1.mp4', writer='ffmpeg')

plt.show()