import numpy as np
import matplotlib.pyplot as plt

'''
Resolucion de la ecuacion de laplace en un contorno irregular.
El contorno se compone de la recta y = 2x, y un arco de dos circunferencias de distintos radios.
La frontera del plano esta fijada a V = 0 y las de las esferas a distinto potencial no nulo.
La frontera y = 0 es oscilante,  V(x,y=0) = sin(2*pi*x) 
Se resuelve usando jacobi para facilitar la implementacion de las fronteras usando un array auxiliar.
'''

# Malla 
N = 200 # puntos de la malla
L = 1.0 # longitud del lado del dominio
x = np.linspace(0, L, N)    
y = np.linspace(0, L, N)    
X, Y = np.meshgrid(x, y)    # malla espacial cuadrada
h = x[1] - x[0] # paso espacial

# Anillo + plano 
r = np.sqrt(X**2 + Y**2)
r_int, r_ext = 0.3, 1.0

anillo = (r > r_int) & (r < r_ext)

# Plano incluyendo y=0
plano = (X >= 0) & (Y >= 0) & (X > (1/2)*Y)

mask = anillo & plano

# Inicializacion 
u = np.zeros((N, N))

# Condiciones de contorno 
# bordes radiales
u[r >= r_ext] = 0.5
u[r <= r_int] = 1.0

# Oscilacion en y=0
u[0, :] = np.sin(2*np.pi*x)

# resto del plano fuera del dominio
u[~plano] = 0.0

# Resolucion con jacobi
tol = 1e-6
delta = 1.0

while delta > tol:

    u2 = u.copy()

    # actualización interior
    u2[1:-1, 1:-1] = 0.25 * (
        u[2:,1:-1] + u[:-2,1:-1] +
        u[1:-1,2:] + u[1:-1,:-2]
    )

    # aplicar máscara
    u2 = np.where(mask, u2, u)

    # reimponemos las condiciones de contorno
    u2[r >= r_ext] = 0.5
    u2[r <= r_int] = 1.0
    u2[0, :] = np.sin(2*np.pi*x) 
    u2[~plano] = 0.0

    delta = np.max(np.abs(u2 - u))
    u = u2

print(f'Convergencia alcanzada: delta = {delta:.2e}')

# Campo electrico
dV_dy, dV_dx = np.gradient(u, h, h)

Ex = -dV_dx
Ey = -dV_dy

E_mag = np.sqrt(Ex**2 + Ey**2)
E_max = np.percentile(E_mag, 90)   

# Limitamos los outliers para una visualizacion mas limpia
factor = np.minimum(1, E_max / (E_mag + 1e-12))
Ex_clip = Ex * factor
Ey_clip = Ey * factor


mode_c = np.sqrt(Ex_clip**2 + Ey_clip**2)

# Visualizacion 
plt.figure(figsize=(6,5))

plt.contourf(X, Y, u, 50)
plt.colorbar(label='Potencial')

step = 5
plt.quiver(X[::step, ::step], Y[::step, ::step],
           Ex_clip[::step, ::step], Ey_clip[::step, ::step],
           color='k', scale=100)

# fronteras
frontera = mask.astype(int)
plt.contour(X, Y, frontera, levels=[0.5], colors='red', linewidths=1.5)
plt.hlines(0, 0, 1, lw=4, color='r')
plt.plot([0,0.5],[0,1], color='r')

plt.title(r'V(x,y) y $\vec{E}$')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([0,L,0,L])
plt.savefig('Modv.png', dpi = 500)
plt.show()

# Modulo de E 
plt.figure(figsize=(6,5))
plt.contourf(X, Y, mode_c)
plt.colorbar(label='|E|')

plt.xlabel('x')
plt.ylabel('y')
plt.title('|E|(x,y)')
plt.savefig('Mode.png', dpi = 500)

plt.show()