import numpy as np
import matplotlib.pyplot as plt
from funciones_t9 import gs_cil_2d

# Parametros físicos
r0 = 0.01   
rf = 0.04  
z0 = 0
zf = 20 
Vi = 2.5    
Ve = 0.0  

nr = 100
dr = (rf - r0)/(nr - 1)

# hacemos dz = dr y ajustamos nz
dz = dr
nz = int((zf - z0)/dz) + 1

r = np.linspace(r0, rf, nr)
z = np.linspace(z0, zf, nz)

# Inicializacion 
V = np.zeros((nr, nz))

# Condiciones de contorno 
V[0, :] = Vi
V[-1, :] = Ve
V[:, 0] = 0.0
V[:, -1] = 0.0

# No tenemos termino inhomogeneo
rho = np.zeros_like(V)

# Iteracion 
tol = 1e-3
delta = 1.0

while delta > tol:
    V, delta = gs_cil_2d(V, rho=rho, r_min=r0, a=dr)

    # Volvemos a imponer los contornos
    V[0, :] = Vi
    V[-1, :] = Ve
    V[:, 0] = 0.0
    V[:, -1] = 0.0

print(f"Convergencia: {delta:.2e}")

# Representacion 
plt.figure(figsize=(6,5))

plt.imshow(
    V.T,
    extent=[r0, rf, z0, zf],
    origin='lower',
    aspect='auto'
)

plt.axvline(r0, color='red', linestyle='--', label='Ri')
plt.axvline(rf, color='red', linestyle='--', label='Re')

plt.colorbar(label="V (V)")
plt.xlabel("r (m)")
plt.ylabel("z (m)")
plt.title("Condensador cilíndrico finito")
plt.legend()
plt.savefig('Cond_finito.png', dpi = 500)

plt.show()