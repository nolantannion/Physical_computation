'''
Alba Ruesga Alonso, Sofia Martín Alañón, Nolan Tannion Rodríguez
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import factorized

'''
Resolución de la ecuación de Schrödinger 2D
con un potencial de doble rendija usando Crank-Nicolson.

Se aplican distintos metodos de vectorizacion y operaciones matriciales
mas complejos.

La teoria se basa en la referencia citada en el informe
'''

# ============================================================
# PARAMETROS FISICOS
# ============================================================

hbar = 1.0546e-34
m = 9.109e-31

# ============================================================
# DISCRETIZACION ESPACIAL Y TEMPORAL
# ============================================================

Lx = 1e-8
Ly = 1e-8

Nx = 100
Ny = 100

dx = Lx / Nx
dy = Ly / Ny

dt = 2e-17

# ============================================================
# MALLA ESPACIAL
# ============================================================

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)

X, Y = np.meshgrid(x, y, indexing='ij')

# ============================================================
# FUNCION DE ONDA INICIAL
# ============================================================

sigma = 1.1e-9
k0 = 8e10

# Paquete gaussiano con momento inicial en x
psi = (np.exp(-((X + Lx/4)**2 + Y**2)/(2*sigma**2))* np.exp(1j * -k0 * X))
psi = psi.astype(complex)

# Condiciones de contorno
psi[0,:] = psi[-1,:] = 0
psi[:,0] = psi[:,-1] = 0

# ============================================================
# POTENCIAL DE DOBLE RENDIJA
# ============================================================
V0 = 1e-16

# Barrera inicialmente nula
V = np.zeros((Nx, Ny))

# Posición de la barrera
x_bar = 0

# Grosor de la barrera
grosor = 2e-10

# Rendijas
separacion = 5e-10
anchura = 1.5e-10

# Barrera
barrera = np.abs(X - x_bar) < grosor

# Rendija superior
rendija_sup = ( (Y > separacion/2 - anchura/2) & (Y < separacion/2 + anchura/2))

# Rendija inferior
rendija_inf = ( (Y > -separacion/2 - anchura/2) & (Y < -separacion/2 + anchura/2))

# La barrera existe excepto en las rendijas, usamos los operadores not y and logicos: ~, |
V[barrera & ~(rendija_sup | rendija_inf)] = V0

# Calculo del laplaciano 2D
Ix = eye(Nx)    # Matrices identidad del tamaño correspondiente
Iy = eye(Ny)    # Matrices identidad del tamaño correspondiente

# Segunda derivada en x e y vectorizando para mayor velocidad
ex = np.ones(Nx)
Tx = diags([ex[:-1], -2*ex, ex[:-1]],offsets=[-1,0,1]) / dx**2

ey = np.ones(Ny)
Ty = diags([ey[:-1], -2*ey, ey[:-1]],offsets=[-1,0,1]) / dy**2

# Laplaciano 2D. Asociamos a cada Y  la segunda derivada de x y a cada x segunda derivada de y
# Esto resuelve todos los terminos de la discretizacion.
# Se usa la funcion kron que en este caso lleva a cabo la relacion que busacamos
Laplacian = kron(Iy, Tx) + kron(Ty, Ix)

# ============================================================
# Matrices para Crank-Nicolson
# ============================================================
Ntot = Nx * Ny  # numero de puntos totales
I = eye(Ntot)   # diagonal con 

# Potencial como matriz diagonal ya que solo influye el potencial 
# de cada termino sin elementos cruzados
# se define de esta manera para disminuir el tiempo de calculo
V_mat = diags(V.ravel())

# Coeficientes
alpha = 1j * hbar * dt / (4 * m)
beta = 1j * dt / (2 * hbar)

# Matrices CN
A = I + alpha * Laplacian + beta * V_mat
B = I - alpha * Laplacian - beta * V_mat

# Formatos eficientes para el calculo
A = A.tocsc()   # hace mas eficiente factorized
B = B.tocsr()   # hace mas eficiente la multiplicacion con @

# Factorizacion LU. Resuelve el sistema dado por la matriz A
# Devulve una funcion que toma como input el termino independiente de la ecuacion a resolver
solve_A = factorized(A)



# ============================================================
# PASO TEMPORAL
# ============================================================
def paso_CN(psi):
    # Convertimos en un vector para usar el potencial diagonal construido antes
    psi_vec = psi.ravel()

    # Lado derecho
    rhs = B @ psi_vec

    # Resolver sistema lineal
    psi_new = solve_A(rhs)

    # Volver a matriz 2D
    psi_new = psi_new.reshape((Nx, Ny))

    # Condiciones de contorno en los bordes
    psi_new[0,:] = 0
    psi_new[-1,:] = 0
    psi_new[:,0] = 0
    psi_new[:,-1] = 0

    return psi_new

# ============================================================
# EVOLUCION TEMPORAL
# ============================================================

pasos = 2000 

psit = np.zeros((Nx, Ny, pasos), dtype=complex)
psit[:,:,0] = psi

for n in range(1, pasos):
    psit[:,:,n] = paso_CN(psit[:,:,n-1])

# ============================================================
# ANIMACION
# ============================================================
fig, ax = plt.subplots()

img = ax.imshow( np.abs(psit[:,:,0]).T**2, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='equal')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"$|\psi(x,y,t)|^2$")

plt.colorbar(img, ax=ax)

# ============================================================
# FUNCION DE ANIMACION
# ============================================================

def update(frame):
    img.set_data(np.abs(psit[:,:,frame]).T**2)

    return (img,)

# ============================================================
# EJECUCION
# ============================================================
ani = FuncAnimation( fig, update, frames=range(0, pasos, 5), interval=10, blit=True)
plt.show()