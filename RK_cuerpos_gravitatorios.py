import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation
import time 
from funciones_t5 import euler_cromer
# Medimos el tiempo de ejecución del programa
start_t = time.time()

# Definimos las constantes del problema
n = 3
m1 = 100
m2 = 100
m3 = 100
G = 2.5

# Definir la distancia entre partículas (lado del triángulo)
R = 3.0  # Longitud del lado del triángulo equilátero

# Posiciones iniciales en los vértices del triángulo equilátero
x1, y1 = R, 0
x2, y2 = -R/2, np.sqrt(3)/2 * R
x3, y3 = -R/2, -np.sqrt(3)/2 * R

# Velocidad orbital requerida para estabilidad
V = np.sqrt(G * (10 + 10 + 10) / R) / np.sqrt(3)

# Velocidades iniciales perpendiculares al radio vector
vx1, vy1 = 0, V
vx2, vy2 = -V * np.cos(np.pi / 6), -V * np.sin(np.pi / 6)
vx3, vy3 = V * np.cos(np.pi / 6), -V * np.sin(np.pi / 6)

# Vector de condiciones iniciales
y0 = [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]


# EDO a resolver
def edo(t, y):
    # y = [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3]
    dx1, dy1, dx2, dy2, dx3, dy3 = y[6], y[7], y[8], y[9], y[10], y[11]

    # Cálculo de distancias entre partículas
    r12_squared = (y[0] - y[2])**2 + (y[1] - y[3])**2
    r13_squared = (y[0] - y[4])**2 + (y[1] - y[5])**2
    r23_squared = (y[2] - y[4])**2 + (y[3] - y[5])**2
    r12 = r12_squared**(3/2)
    r13 = r13_squared**(3/2)
    r23 = r23_squared**(3/2)

    # Evitar divisiones por cero en caso de colisión
    if r12 == 0: r12 = 1e-6
    if r13 == 0: r13 = 1e-6
    if r23 == 0: r23 = 1e-6

    # Aceleraciones por atracción gravitacional
    dvx1 = -G * (m2 * (y[0] - y[2]) / r12 + m3 * (y[0] - y[4]) / r13)
    dvy1 = -G * (m2 * (y[1] - y[3]) / r12 + m3 * (y[1] - y[5]) / r13)
    dvx2 = -G * (m1 * (y[2] - y[0]) / r12 + m3 * (y[2] - y[4]) / r23)
    dvy2 = -G * (m1 * (y[3] - y[1]) / r12 + m3 * (y[3] - y[5]) / r23)
    dvx3 = -G * (m1 * (y[4] - y[0]) / r13 + m2 * (y[4] - y[2]) / r23)
    dvy3 = -G * (m1 * (y[5] - y[1]) / r13 + m2 * (y[5] - y[3]) / r23)

    return [dx1, dy1, dx2, dy2, dx3, dy3, dvx1, dvy1, dvx2, dvy2, dvx3, dvy3]

# Condiciones iniciales
# y0 = [1, 0, -0.5, np.sqrt(3)/2, -0.5, -np.sqrt(3)/2, 
#       0, 1, -0.866, -0.5, 0.866, -0.5]

tspan = (0, 15)
teval = np.linspace(0, 15, 2000)

# Resolver la EDO
sol = solve_ivp(edo, tspan, y0, method='DOP853', t_eval=teval)
# t, sol = euler_cromer(edo, t0 = t0, tf = tf, estado0= y0, h= dt)



# Extraer soluciones
t = sol.t
x1, y1 = sol.y[0], sol.y[1]
x2, y2 = sol.y[2], sol.y[3]
x3, y3 = sol.y[4], sol.y[5]

# Calcular el centro de masa en cada instante
X_CM = (m1 * x1 + m2 * x2 + m3 * x3) / (m1 + m2 + m3)
Y_CM = (m1 * y1 + m2 * y2 + m3 * y3) / (m1 + m2 + m3)

# Transformar las posiciones al marco del centro de masa
x1_cm, y1_cm = x1 - X_CM, y1 - Y_CM
x2_cm, y2_cm = x2 - X_CM, y2 - Y_CM
x3_cm, y3_cm = x3 - X_CM, y3 - Y_CM

# Crear la figura
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title(f'Movimiento de {n} masas')
ax.set_xlabel('Posición en x')
ax.set_ylabel('Posición en y')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.grid()

# Inicializar las partículas sin la línea de unión
part1 = ax.scatter([], [], color='blue', label="Partícula 1")
part2 = ax.scatter([], [], color='red', label="Partícula 2")
part3 = ax.scatter([], [], color='green', label="Partícula 3")

# Arrays para el trail de las posiciones 
xx1, yy1 = [], []
xx2, yy2 = [], []
xx3, yy3 = [], []

# Iniciamos una estela para cada cuerpo
duracion = 0.7 
framesps = len(teval)/(teval[-1] - teval[0])
max = int(duracion*framesps)

line1, = ax.plot([],[], color = 'blue', alpha = 0.5)
line2, = ax.plot([],[], color = 'red', alpha = 0.5)
line3, = ax.plot([],[], color = 'green', alpha = 0.5)

def init():
    part1.set_offsets([[x1_cm[0], y1_cm[0]]])
    part2.set_offsets([[x2_cm[0], y2_cm[0]]])
    if m3 != 0: part3.set_offsets([[x3_cm[0], y3_cm[0]]])

    line1.set_data([],[])
    line2.set_data([],[])
    line3.set_data([],[])
    
    return part1, part2, part3, line1, line2, line3

def update(frame):
    part1.set_offsets([[x1_cm[frame], y1_cm[frame]]])
    part2.set_offsets([[x2_cm[frame], y2_cm[frame]]])
    if m3 != 0: part3.set_offsets([[x3_cm[frame], y3_cm[frame]]])

    if len(xx1) > max:
        xx1.pop(0)
        yy1.pop(0)
        xx2.pop(0)
        yy2.pop(0)
        xx3.pop(0)
        yy3.pop(0)

    xx1.append(x1_cm[frame])
    xx2.append(x2_cm[frame])
    xx3.append(x3_cm[frame])
    yy1.append(y1_cm[frame])
    yy2.append(y2_cm[frame])
    yy3.append(y3_cm[frame])

    line1.set_data(xx1,yy1)
    line2.set_data(xx2,yy2)
    if m3 != 0: line3.set_data(xx3,yy3)

    return part1, part2, part3, line1, line2, line3

# Configurar animación
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, interval=15, blit=True)

# Medimos el tiempo que ha tardado y representamos 
end_t = time.time()

tiempo_ejecucion = end_t - start_t

print(f'El tiempo de ejecucion del programa es de {tiempo_ejecucion} segundos')


ani.save(f'{n} Cuerpos.mp4', writer = 'ffmpeg')

plt.legend()
plt.show()