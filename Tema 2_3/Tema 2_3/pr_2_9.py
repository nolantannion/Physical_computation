import math as m
import matplotlib.pyplot as plt
import numpy as np

# -------------------
# Constantes
# -------------------
g = 9.81         # gravedad de la tierra
H = 1e4          # altura característica atmósfera
B2m = 4e-5       # coeficiente arrastre para un
rho0 = 1         # densidad típica del aire a altura 0

# Integración
h = 1e-3
t0 = 0

# Valores iniciales
x0, y_init = 0, 0
v = 1000

# Funciones para calcular las densidades según los distintos modelos
def rho_const(y):
    return rho0

def rho_y_iso(y):
    return rho0 * np.exp(-y / H)   # signo corregido

def rho_y_ad(y):
    a = 6.5e-3
    alfa = 2.5
    T0 = 300       # Kelvin (más razonable)

    factor = 1 - a*y/T0
    if factor <= 0:
        return 0   # evita valores no físicos

    return rho0 * factor**alfa

# Array con las funciones para los coeficientes y las leyendas
coef = [rho_const, rho_y_ad, rho_y_iso]
leyenda = ['rho_const', 'rho_y_ad', 'rho_y_iso']


# Sistema dinámico
# Definimos vars = [x, vx, y, vy]
def sistema(vars, B):
    modv = np.sqrt(vars[1]**2 + vars[3]**2)

    dvars = np.zeros_like(vars)

    dvars[0] = vars[1]
    dvars[2] = vars[3]

    dvars[1] = -B * modv * vars[1]
    dvars[3] = -g - B * modv * vars[3]

    return dvars

# Función para calcular por euler el siguiente paso
def euler(t, vars0, B):
    tp = t + h
    varsp = vars0 + h * sistema(vars0, B)
    return tp, varsp


# Cálculo para cada rho
for i, rho_model in enumerate(coef):

    t = [t0]
    x, y = [x0], [y_init]

    # velocidad inicial correcta
    theta = np.pi / 4
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)

    variables = np.array([x0, vx, y_init, vy], dtype=float)

    indice = 0

    while y[indice] >= 0:

        # coeficiente drag dependiente de densidad
        B = rho_model(y[indice]) / rho0 * B2m

        tsol, varsol = euler(t[indice], variables, B)

        t.append(tsol)
        variables = varsol

        x.append(varsol[0])
        y.append(varsol[2])

        indice += 1

    # Calculamos el alcance usando el penúltimo punto y una aproximación lineal para el incremento
    alcance = x[indice - 1] + (y[indice]-y[indice-1])/(x[indice]-x[indice-1])*h/2
    print(f'Alcance: {alcance} (m)')

    plt.plot(x, y, label=leyenda[i])

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.grid()
plt.show()
