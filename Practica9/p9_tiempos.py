import numpy as np
import matplotlib.pyplot as plt
from funciones_t9 import gs_cil, jacobi_cil, gss_cil_1d
import time

# Parametros fisicos
r0 = 0.01   
rf = 0.04   
Vi = 2.5    
Ve = 0.0    
NN = np.linspace(100 ,20000 ,100, dtype= int)

# Arrays de tiempos (float)
tg = np.zeros(len(NN))
tj = np.zeros(len(NN))
tgs = np.zeros(len(NN))


# Medimos la compilacion JIT para añadir al tiempo de ejecucion 
N_test = 20
r_test = np.linspace(r0, rf, N_test)
a_test = r_test[1] - r_test[0]

V_test = np.zeros(N_test)
V_test[0] = Vi
V_test[-1] = Ve


# GS
t0 = time.time()
gs_cil(V_test.copy(), a_test, r0)
t_jit_gs = time.time() - t0


# GS sobre relajacion
t0 = time.time()
gss_cil_1d(V_test.copy(), a_test, r0)
t_jit_sor = time.time() - t0


print(f"JIT GS:     {t_jit_gs:.6f} s")
print(f"JIT GS Sobre Rel.:    {t_jit_sor:.6f} s")


tol = 1e-3

for k, N in enumerate(NN):

    # Malla
    r = np.linspace(r0, rf, N)
    h = r[1] - r[0]

    # Inicialización
    Vj = np.zeros(N)
    Vg = np.zeros(N)
    Vgs = np.zeros(N)

    # Condiciones de contorno
    for V in [Vj, Vg, Vgs]:
        V[0] = Vi
        V[-1] = Ve

    rho = np.zeros_like(Vj)


    # Jacobi
    deltaj = 1.0
    t0 = time.time()
    while deltaj > tol:
        Vj, deltaj = jacobi_cil(Vj, h, r0)
    tj[k] = time.time() - t0

    # Gauss-Seidel 
    deltag = 1.0
    t0 = time.time()
    while deltag > tol:
        Vg, deltag = gs_cil(Vg, h, r0)
    tg[k] = time.time() - t0

    # Gauss-Seidel con sobrerrelajacion
    deltags = 1.0
    t0 = time.time()
    while deltags > tol:
        Vgs, deltags = gss_cil_1d(Vgs, h, r0,  omega=0.8) 
    tgs[k] = time.time() - t0


# Representacion 
plt.figure()
plt.plot(NN[1:-1], tj[1:-1], label="Jacobi")
plt.plot(NN[1:-1], tg[1:-1], label="Gauss-Seidel")
plt.plot(NN[1:-1], tgs[1:-1], label="GS - sobre rel.")

plt.xlabel("N")
plt.ylabel("Tiempo (s)")
plt.legend()
plt.grid()

plt.savefig('tiempos.png', dpi = 500)

plt.show()