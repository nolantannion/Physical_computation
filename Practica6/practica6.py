'''
Script para resolver los distintos apartados de la practica 6, incluyendo
una simulacion de las evoluciones temporales y la evolucion temporal de los perfiles laterales.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from funciones_t6 import laplace2D, random_walk_2d

############################################
# CONSTANTES ARRAYS Y VARAIABLES
############################################


# Semilla para numeros aleatorios
np.random.seed(1)

# Parametros caminantes
N = 1000 # numero de pasos
nw = 1000 # Numero de caminantes
dr = 1  # Distancia de paso radial

dist = np.zeros([nw,N+1], dtype= float)
angulo = np.zeros([nw,N+1], dtype= float)

# Parametros difusion
D = 1
deltax = 1
deltat = deltax**2 / (4*D) /2  # estabilidad en 2D / 2

# Numero de puntos en x e y, tiempo final y puntos para ts
NX = 50
NY = 50

tf = 100.0
NT = int(tf/deltat)

t = np.arange(0, (NT + 1)*deltat, deltat)


# densidad: (x, y, t)
rho0 = np.zeros((2*NX+1, 2*NY+1))



############################################
# EVOLUCIONES TEMPORALES
############################################

# condición inicial: delta en el centro
rho0[NX, NY] = 1

# Calculamos la difusion
ts, rho = laplace2D(A= rho0, tf= tf, dt = deltat)


# Calculamos el random walk
xw, yw = random_walk_2d(nw= nw, n = N)





############################################
# APARTADO a
############################################

# Calculamos el cuadrado de los desplazamientos y el desplazamiento cuadratico promedio
x2, y2 = np.sum(xw**2, 0), np.sum(yw**2, 0)

r2 = (x2 + y2)/nw
  

x_fit = np.linspace(0,N,N+1)
coef, pcov= np.polyfit(x_fit, r2, 1, cov=True)

err = np.sqrt(np.diag(pcov))

ajuste  = coef[0]*x_fit + coef[1]

fig4, ax4 = plt.subplots()

ax4.plot(x_fit,r2, label = 'Desplazamiento cuadrático promedio')
ax4.plot(x_fit,ajuste, label = f'y = {coef[0]:.2f}x + {coef[1]:.1f}')

ax4.legend()
ax4.grid()

print(f'C de proporcionalidad: {coef[0]:.5f}, err: {err[0]:.5f}')


ax4.set_xlabel('Pasos')
ax4.set_ylabel(r'$\langle r^2 \rangle$')
ax4.set_title(f'{nw} Caminantes ')


plt.show()



############################################
# APARTADO b
############################################

# Animacion de comparacion
fig, ax = plt.subplots(1, 2, figsize=(10,5))

# Random walk (scatter)
scat = ax[0].scatter([], [], s=1, alpha=0.6)
ax[0].set_xlim(-NX, NX)
ax[0].set_ylim(-NY, NY)

ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)

ax[0].set_title(f"Random Walk")
ax[1].set_title(f"Difusión")

ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

ax[1].set_xlabel('x')
ax[1].set_ylabel('y')



# Difusión (densidad)
img = ax[1].imshow(rho[:, :, 0],
                   extent=[-NX, NX, -NY, NY],
                   origin='lower',
                   vmin=0,
                    vmax=np.max(rho) / 100
                   )

cbar = fig.colorbar(img, ax=ax[1])
cbar.set_label("Densidad")


# Funciond de actualizacion de la animacion
def update(frame):
    fr = int(frame * NT / N)
    scat.set_offsets(np.c_[xw[:, frame], yw[:, frame]]) # Ajustamos los datos juntandolos en una matriz con c_
    img.set_data(rho[:, :, fr])


    #fig.suptitle(f'Simulación: t = {frame}')
    return (scat, img)

# Animacion
anim = FuncAnimation(fig, update,
                     frames=N,
                     interval=50, 
                     blit = True)

# Se pausa hasta completar la animacion o cerrarla
plt.show()



# Evolucion de los perfiles laterales
fig2, ax2 = plt.subplots(1, 2, figsize=(10,6))

# Ejes espaciales
x_axis = np.arange(-NX, NX+1)
y_axis = np.arange(-NY, NY+1)

# Líneas iniciales
line_x, = ax2[0].plot(y_axis, rho[NX, :, 0])
line_y, = ax2[1].plot(x_axis, rho[:, NY, 0])

ax2[0].set_title("Corte x = 0")
ax2[1].set_title("Corte y = 0")

ax2[0].set_xlabel("y")
ax2[1].set_xlabel("x")

ax2[0].set_ylabel("Densidad")

ax2[0].set_title(f"Corte x=0 ")
ax2[1].set_title(f"Corte y=0")
fig2.suptitle('Perfiles laterales')

# Funcion para la animacion de los perfiles laterales
def update_profiles(frame):
    # Cortes
    profile_x = rho[NX, :, frame]   # x = 0
    profile_y = rho[:, NY, frame]   # y = 0

    line_x.set_ydata(profile_x)
    line_y.set_ydata(profile_y)


    fig2.suptitle(f'Perfiles laterales: t={t[frame]})')

    # Hacemos que los ejes se ajusten con cada paso de t
    ax2[0].relim()
    ax2[0].autoscale_view()

    ax2[1].relim()
    ax2[1].autoscale_view()

    return (line_x, line_y)


anim2 = FuncAnimation(fig2, update_profiles,
                      frames=NT,
                      interval=50)

plt.show()





############################################
# APARTADO c
############################################

# Calculamos la entropia usando un histograma para en 2 dimensiones
S = np.zeros(N+1)

for i in range(N+1):
    H, _, _ = np.histogram2d(
        xw[:, i], yw[:, i],
        bins=10,
        range=[[-NX, NX], [-NY, NY]]
    )
    
    p = H / nw
    
    # log seguro
    S[i] = -np.sum(p * np.log(p + 1e-12))

fig3, ax3 = plt.subplots()

ax3.plot(S)

ax3.set_xlabel('Pasos')
ax3.set_ylabel('Entropía')
ax3.grid()
ax3.set_title('Entropía en función de los pasos')

plt.show()




