'''
Script que simula la evolucion de la ecuacion de difusion y la posiciond de nw caminantes bajo 
una condicion inicial gaussiana
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
nw = 2000 # Numero de caminantes
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

tf = 1000.0
NT = int(tf/deltat)

t = np.arange(0, (NT + 1)*deltat, deltat)


# densidad: (x, y, t)
rho = np.zeros((2*NX+1, 2*NY+1))



############################################
# EVOLUCION TEMPORAL 
############################################



# Gaussiana
X = np.arange(-NX, NX+1)
Y = np.arange(-NY, NY+1)
X, Y = np.meshgrid(X, Y, indexing='ij')
sigma0 = 5

rho[:, :] = np.exp(-(X**2 + Y**2)/(2*sigma0**2))
rho[:, :] /= np.sum(rho[:, :])  # normalizar

# Calculamos la difusion
ts, rho = laplace2D(A= rho, tf= tf, dt = deltat)




# Distribucion inicial gaussiana
xw = np.zeros((nw, N))
yw = np.zeros((nw, N))


xw[:, 0] = np.random.normal(0, sigma0, size=nw)
yw[:, 0] = np.random.normal(0, sigma0, size=nw)



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
# APARTADO a
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



# Difusión (densidad)
img = ax[1].imshow(rho[:, :, 0],
                   extent=[-NX, NX, -NY, NY],
                   origin='lower',
                   vmin=0,
                    vmax=np.max(rho) / 10
                   )

cbar = fig.colorbar(img, ax=ax[1])
cbar.set_label("Densidad")


# Funciond de actualizacion de la animacion
def update(frame):
    fr = int(frame * NT / N)    # ajustamos para mostrar el mismo t
    scat.set_offsets(np.c_[xw[:, frame], yw[:, frame]])
    img.set_data(rho[:, :, fr])


    fig.suptitle(f'Simulación: t = {frame}')
    return scat, img

# Animacion
anim = FuncAnimation(fig, update,
                     frames=N,
                     interval=50, 
                     blit = False)

# Se pausa hasta completar la animacion o cerrarla
plt.show()



# Mostramos la gaussiana en t = 50, utilizado para la grafica del informe
# fig1, ax1 = plt.subplots(1, 2, figsize=(10,5))

# # Random walk (scatter)
# ax1[0].set_xlim(-NX, NX)
# ax1[0].set_ylim(-NY, NY)
# ax1[0].set_box_aspect(1)
# ax1[1].set_box_aspect(1)
# ax1[0].set_title(f"Random Walk")
# ax1[1].set_title(f"Difusión")
# ax1[0].scatter(xw[:,100], yw[:,100], s = 1)
# ax1[1].imshow(rho[:,:,int(50*NT/N)])

# fig1.suptitle('Gaussiana en t = 50')

# cbar = fig1.colorbar(img, ax=ax[1])
# cbar.set_label("Densidad")

# plt.show()
# fig1.savefig('Gaussiana.png', dpi = 500)





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



