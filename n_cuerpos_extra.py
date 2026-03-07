import numpy as np
from funciones_t5 import Body
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros
h, w = 50, 50
N = 2

# Crear cuerpos
bodies = []

ml, mt = 10,200
rt = np.array([h/2,0.0])
vt = np.array([0.0,2.0])

rl = np.array([h/2 + 5.0, 5.0])
vl = np.sqrt(2)/2* np.array([3.0,3.0])

tierra  = Body(mt,rt,vt)
luna = Body(ml,rl,vl)


bodies.append(tierra)
bodies.append(luna)
# masa central
b_centro = Body(1e-6, np.array([0,0]), np.array([0,0]))

# Integración temporal
dt = 0.01
t0, tf = 0, 20
steps = int((tf-t0)/dt)

traj = np.zeros((steps,N,2))

for k in range(steps):

    for b in bodies:
        b.zerof()

    for i in range(N):

        bodies[i].calcularf(b_centro)

        for j in range(i+1,N):
            bodies[i].calcularf(bodies[j])
            bodies[j].calcularf(bodies[i])

    for b in bodies:
        b.step(dt)

    for i,b in enumerate(bodies):
        traj[k,i] = b.r

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

def init():
    return scatters + [center]

def update(frame):

    for i in range(N):
        scatters[i].set_offsets(traj[frame,i])

    return scatters + [center]

anim = FuncAnimation(
    fig,
    update,
    frames=steps,
    init_func=init,
    interval=5,
    blit=True
)

plt.show()