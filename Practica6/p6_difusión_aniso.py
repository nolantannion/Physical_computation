
'''
Script que simula la evolucion los caminantes y la difusion en un sistema anisotropo
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from funciones_t6 import laplace2D_anisot, random_walk_2d_anisot

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


# Numero de puntos en x e y, tiempo final y puntos para ts
NX = 50
NY = 50

tf = 100.0

# densidad: (x, y, t)
rho = np.zeros((2*NX+1, 2*NY+1))



############################################
# EVOLUCIONES TEMPORALES
############################################

# condición inicial: delta en el centro
rho[NX, NY] = 1

# Calculamos la difusion
ts, rho = laplace2D_anisot(A= rho, tf= tf)


NT = len(ts)


# Calculamos el random walk
xw, yw = random_walk_2d_anisot(nw= nw, n = N)



############################################
# APARTADO a
############################################

# Calculamos el cuadrado de los desplazamientos y el desplazamiento cuadratico promedio
x2, y2 = np.sum(xw**2, 0), np.sum(yw**2, 0)

r2 = (x2 + y2)/nw

x2p, y2p = x2/nw, y2/nw


  

x_fit = np.linspace(0,N,N+1)
coef, pcov= np.polyfit(x_fit, r2, 1, cov=True)

err = np.sqrt(np.diag(pcov))

Dx = np.polyfit(x_fit, x2p, 1)
Dy = np.polyfit(x_fit, y2p, 1)

ajuste  = coef[0]*x_fit + coef[1]
ajustex  = Dx[0]*x_fit + Dx[1]
ajustey  = Dy[0]*x_fit + Dy[1]




fig4, ax4 = plt.subplots()

ax4.plot(x_fit,r2, label = r'$\langle r^2 \rangle$')
ax4.plot(x_fit,ajuste, label = f'y = {coef[0]:.2f}x + {coef[1]:.1f}')

ax4.plot(x_fit,x2p, label = r'$\langle x^2 \rangle$')
ax4.plot(x_fit,ajustex, label = f'y = {Dx[0]:.2f}x + {Dx[1]:.1f}')

ax4.plot(x_fit,y2p, label = r'$\langle y^2 \rangle$')
ax4.plot(x_fit,ajustey, label = f'y = {Dy[0]:.2f}x + {Dy[1]:.1f}')

ax4.set_xlabel('Pasos')
ax4.set_ylabel('Desplazamiento cuadratico promedio')

ax4.legend()
ax4.grid()

print(f'D obtenido mediante el ajuste: {coef[0]:.5f}')
print(f'Dx obtenido mediante el ajuste: {Dx[0]:.5f}')
print(f'Dy obtenido mediante el ajuste: {Dy[0]:.5f}')

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
img = ax[1].imshow(rho[:, :, 1],
                   extent=[-NX, NX, -NY, NY],
                   origin='lower',
                   vmin=0,
                    vmax=np.max(rho) / 100, #reducimos el maximo para mejorar visibilidad
                   )

cbar = fig.colorbar(img, ax=ax[1])
cbar.set_label("Densidad")


# Funciond de actualizacion de la animacion
def update(frame):
    fr = int(frame * NT / N)

    scat.set_offsets(np.c_[xw[:, frame], yw[:, frame]])
    img.set_data(rho[:, :, fr])

    return (scat, img)

# Animacion
anim = FuncAnimation(fig, update,
                     frames=N,
                     interval=50, 
                     blit = True)

# Se pausa hasta completar la animacion o cerrarla
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



