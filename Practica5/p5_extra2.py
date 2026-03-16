import numpy as np
from funciones_t5 import Body
import matplotlib.pyplot as plt

'''
Script para representar el efecto de la resonancia en las orbitas por la influencia de jupiter.

Se representa la desviacion radial maxima 
'''


# Constantes
G = 1.0
Msol = 1000.0
Mj = 1.0

# órbita de Júpiter
rj = 25.0
vj = np.sqrt(G*Msol/rj)

# semiejes mayores a explorar
a_vals = np.linspace(10.0,22.0,200)

dt = 0.02
tf = 200.0
steps = int(tf/dt)

variacion = []

sol = Body(Msol, np.array([0.0,0.0], dtype=float), np.array([0.0,0.0], dtype=float))



for a0 in a_vals:
    jupiter = Body( Mj, np.array([rj,0.0], dtype=float), np.array([0.0,vj], dtype=float))

    v = np.sqrt(G*Msol/a0)

    asteroid = Body( 1e-6, np.array([a0,0.0], dtype=float), np.array([0.0,v], dtype=float))

    bodies = [jupiter, asteroid]

    r_hist = []

    for k in range(steps):

        for b in bodies:
            b.zerof()

        # fuerza del Sol
        for b in bodies:
            b.calcularf(sol)

        # interacción con Júpiter
        bodies[0].calcularf(bodies[1])
        bodies[1].calcularf(bodies[0])

        for b in bodies:
            b.step(dt)

        r = np.linalg.norm(asteroid.r)
        r_hist.append(r)

    r_hist = np.array(r_hist, dtype=float)

    variacion.append(np.max(r_hist) - np.min(r_hist))


variacion = np.array(variacion, dtype=float)

plt.figure()

plt.scatter(a_vals, variacion, s=10)


plt.xlabel("Semieje mayor inicial (Ua)")
plt.ylabel("Variación radial (Ua)")
plt.grid()
plt.title("Resonancias según periodo")

# plt.savefig('Resonancia_periodo.png', dpi = 500)

plt.show()