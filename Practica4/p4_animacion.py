import numpy as np
import matplotlib.pyplot as plt
from funciones_t4 import euler_cromerr
from matplotlib.animation import FuncAnimation

'''
Simulación de la evolución del sistema para distintos rozamientos
'''

# Variables y constantes
g = 9.81    # Aceleracion de la gravedad
l = 9.81   # Longitud del pendulo
roz = 1/2   # Coeficiente de rozamiento
omega_forz = 2/3  # Frecuencia de la fuerza impulsora
Af = 1.2  # Amplitud de la fuerza impulsora
fi = 0  # Desfase de la fuerza impulsora

# Parametros del sistema con rozamiento
parametros = [Af,omega_forz, roz]

def sistema(t,vars, params):
    '''
    INPUT
    - t: tiempo 
    - vars: variables a derivar

    OUTPUT
    - dvars: derivadas de las variables contendias en vars
    '''
    # Creamos un array del mismo tamaño que las variables
    dvars = np.zeros_like(vars)

    # Derivadas de cada variable
    dvars[0] = vars[1]
    dvars[1] = -g/l * np.sin(vars[0]) - params[2]*vars[1] + params[0]*np.sin(params[1]*t)

    return dvars


# Parametros de integracion
t0 = 0
tmax = 50
dt = 1e-2


# Condiciones iniciales del pendulo 1
theta01 = 0.20
omega01 = 0.0
estado1 = [theta01, omega01]

t1, sol1 = euler_cromerr(sistema=sistema, t0 = t0, tf = tmax, estado0= estado1, params= parametros, h = dt)

theta1 = sol1[:,0]

# Array con los 
rozamientos = np.linspace(0.3,0.9,6)
leyendas = []

# Array que almacenara las soluciones
thetasol = []

# Pendulo 2 y representacion
for rozm in rozamientos:

    theta02 = theta01 + 1e-4
    omega02 = 0.0
    estado2 = [theta02, omega02]
    parametros[2] = rozm
    t2, sol2 = euler_cromerr(sistema=sistema, t0 = t0, tf = tmax, estado0= estado2, params= parametros, h = dt)

    theta2 = sol2[:,0]

    thetasol.append(theta2)
    leyendas.append(f'q = {rozm:.3f}')

thetasol.append(theta1) # Añadimos el primer pendulo
leyendas.append(f'ref  q={roz}')
thetasol = np.array(thetasol)   # Transformamos a un np array para trabajar con mas facilidad

# Transformamos a coordenadas cartesianas
x = l* np.sin(thetasol)
y = -l*np.cos(thetasol)

# Figura para la animacion
fig, ax = plt.subplots(figsize = (10,6))
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_aspect('equal')

# Representamos los pendulos como segmentos con borde redondeado
lines = [ax.plot([], [], 'o-', label = leyendas[_])[0] for _ in range(len(rozamientos))]
lines.append(ax.plot([], [], 'o-', color= 'black', label = leyendas[-1])[0])  # Primer pendulo graficado de color negro

# Situamos la leyenda fuera del cuadro de la grafica para mejor visibilidad
ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
# Valor de la fuerza en funcion del tiempo
Valorf = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                  color='red', fontsize=9, va='top')

# Grafica inicial
def init():
    for line in lines:
        line.set_data([], [])
    return lines


# Funcion para actualizar la animacion
def animate(frame):
    for i, line in enumerate(lines):
        line.set_data([0, x[i, frame]], [0, y[i, frame]])

    # Valor de la fuerza en este instante
    F_t    = Af * np.sin(omega_forz * t1[frame])
    # Ajustamos el texto de la fuerza y el titulo
    Valorf.set_text(f'F(t) = {F_t:.2f}')
    ax.set_title(f'Evolucion de los pendulos. t = {t1[frame]:.2f}')

    return lines + [Valorf]

# Llamamos a la funcion para la animacion con los parametros correspondientes
ani = FuncAnimation(
    fig,
    animate,
    frames=len(t1),
    init_func=init,
    interval=10,
    blit=False
)

# Almacenamos la simulacion (mucho mas lento que visualizarla solo)
ani.save('Pendulossim.mp4', writer='ffmpeg')
plt.show()