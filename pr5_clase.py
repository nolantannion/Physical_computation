import numpy as np
from funciones_t5 import Body
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros
h, w = 50, 50
N = 25

# Crear cuerpos
bodies = []

for _ in range(N):

    m = np.random.uniform(50,100)

    # x = np.random.choice([-1,1])*np.random.uniform(0.3*w, w)
    # y = np.random.choice([-1,1])*np.random.uniform(0.3*h, h)

    # Creamos una distribucion en forma de disco
    rr = np.random.uniform(0.4*w, w)
    th = np.random.uniform(0, 2*np.pi)

    x = rr*np.cos(th)
    y = rr*np.sin(th)

    r = np.array([x,y])

    # v = np.random.uniform(-10,10,2)
    vv = np.random.uniform(-30,-50)
    vx = -vv*np.sin(th)
    vy = vv*np.cos(th)

    v = np.array([vx,vy])

    # Defino velocidad puramente angular


    bodies.append(Body(m,r,v))

# masa central
b_centro = Body(1e5, np.array([0,0]), np.array([0,0]))

# Integración temporal
dt = 0.01
t0, tf = 0, 20
steps = int((tf-t0)/dt)

# Almacenamos las trayectorias en funcion del tiempo
traj = np.zeros((steps,N,2))
choque = [] # Array para almacenar los indices en caso de choque

# Calculamos la evolucion temporal
for k in range(steps):

    # Reiniciamos las fuerzas
    for b in bodies:
        b.zerof()

    # Calculamos f para cada cuerpo. Crecimiento del calculo O(N**2)
    for i in range(N):
        # Calculamos la fuerza ejercida por la masa central
        bodies[i].calcularf(b_centro)

        for j in range(i+1,N):
            bodies[i].calcularf(bodies[j])
            bodies[j].calcularf(bodies[i])

    # Calculamos el siguiente punto y comprobamos si hay colision con el centro
    for i,b in enumerate(bodies):
        b.step(dt)
        # Comprobamos la colision
        if np.sqrt((b.r[0] - b_centro.r[0])**2 + (b.r[1] - b_centro.r[1])**2) < 1.5:
            choque.append(i)
            
    # Almacenamos, si ha habido colision el punto no evoluciona mas
    for i,b in enumerate(bodies):
        traj[k,i] = b.r
        if i in choque:
            traj[k,i] = np.array([0.0,0.0])
        

# Animación
fig, ax = plt.subplots()

ax.set_xlim(-1.5*w,1.5*w)
ax.set_ylim(-1.5*h,1.5*h)

# partículas
scatters = [
    ax.scatter(traj[0,i,0], traj[0,i,1], s=bodies[i].m/5)
    for i in range(N)
]

# masa central
center = ax.scatter(0,0,color='black',s=100)

# Fotograma inicial
def init():
    return scatters + [center]

# Funcion de actualizacion
def update(frame):

    for i in range(N):
        scatters[i].set_offsets(traj[frame,i])

    return scatters + [center]

# Animacion con sus parametros
anim = FuncAnimation(
    fig,
    update,
    frames=steps,
    init_func=init,
    interval=30,
    blit=True
)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Distribución alrededor de un cuerpo masivo ')
anim.save('Orbita_central.mp4', writer='ffmpeg')
plt.show()