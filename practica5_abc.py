import numpy as np
import matplotlib.pyplot as plt
from funciones_t5 import euler_cromer, vis_viva
from matplotlib.animation import FuncAnimation

'''
Resolucion de los apartados a y b. Incluye una animacion de la órbita
'''


# Constantes
G = 1
Ms = 4*np.pi**2
Mj = 1.9e27         # (kg) aprox 1000 veces menos que el sol
Mh = 2.2e14             # (kg)

Frmhj = Mh/(Mj *1000) # Masa del cometa en masas solares

# Datos del cometa Halley
r0 = 0.59   # perihelio (UA)
vy0 = 11.47270  # v en perihelio (UA/año)
Periodo = 75.979    # (a)
T = 1*Periodo   # Tiempo a integrar (a)

# Datos de pluton
rp_pl = 29.7    # Radio perihelio
a_pl = (29.7 + 49.3)/2  # Semieje mayor
vy_pl = vis_viva(Ms, rp_pl, a_pl)
T_pl = 248.3    # Periodo de pluton en años


dt = 0.0005


def sistema(t,estado):

    x,vx,y,vy = estado

    r = np.sqrt(x**2 + y**2)

    ax = -Ms*x/r**3
    ay = -Ms*y/r**3

    return np.array([vx,ax,vy,ay])


# condición inicial
estado0 = np.array([r0,0,0,vy0])
estado0_pl = np.array([29.7, 0, 0, vy_pl])


# integración
t,vars = euler_cromer(sistema,0,T,estado0,dt)
tp, varsp = euler_cromer(sistema, 0,T_pl, estado0_pl)

# Almacenamos las soluciones 
x = vars[:,0]
y = vars[:,2]

xp = varsp[:,0]
yp = varsp[:,2]

vx = vars[:,1]
vy = vars[:,3]

vxp = varsp[:,1]
vyp = varsp[:,3]

r = np.sqrt(x**2 + y**2)
v = np.sqrt(vx**2 + vy**2)

rp = np.sqrt(xp**2 + yp**2)
vp = np.sqrt(vxp**2 + vyp**2)

# Mostramos en pantalla el radio del afelio, velocidad maxima
# y la distancia entre la posicion inicial tras una orbita
print('Cometa Halley: ')
print("Afelio =",np.max(r),"UA")
print('|V| max =', np.max(v),"UA/año \n")

print('Pluton: ')
print("Afelio =",np.max(rp),"UA")
print('|V| max =', np.max(vp),"UA/año \n")



figura, eje = plt.subplots(figsize = (6,6))
eje.plot(x,y, color = 'b', label = 'Cometa')
eje.scatter(x[-1],y[-1], color = 'b')

eje.scatter(xp[-1], yp[-1], label = 'Pluton', color = 'r')
eje.plot(xp,yp, color = 'r')

eje.scatter(0,0, color = 'y', s = 25, label = 'Sol')
eje.axis('equal')
eje.set_title('Trayectorias')
eje.legend()

eje.set_xlabel('x (UA)')
eje.set_ylabel('y (UA)')





# Calculo y representacion de energias y momento angular
Ech = 1/2*v**2
Eph = -Ms/r

Eth = Ech + Eph # Por unidad de masa

L = x*vy - y*vx   # Momento angular en el plano por unidad de masa

VarE = Eth[0] - Eth[-1]
print('La variacion de energia en el sistema tras una orbita: ', VarE )

figura1, eje1 = plt.subplots(nrows=1,ncols=3, figsize = (10,6))
eje1[0].plot(t,Ech, lw = 0.8, label = 'Energía cinética')
eje1[0].plot(t,Eph, lw = 0.8, label = 'Energía potencial')
eje1[1].plot(t,Eth, lw = 0.8, label = 'Energía Total')
eje1[2].plot(t,L, lw = 0.8, label = 'Momento angular')

eje1[0].legend()
eje1[1].legend()
eje1[2].legend()


# Animacion de la orbita
fig, ax = plt.subplots()

ax.scatter(0,0,color="y",label="Sol")

# Objetos para la animacion
linea, = ax.plot([],[],lw=1)
punto, = ax.plot([],[],'ro')

xl = 0.2*np.abs(np.max(x) - np.min(x))
yl = 0.2*np.abs(np.max(y) - np.min(y))


ax.set_xlim(np.min(x) - xl ,np.max(x) + xl)
ax.set_ylim(np.min(y) - yl,np.max(y) + yl)
ax.grid()
ax.set_aspect("equal")

# Funcion para actualizar cada fotograma. movemos el planeta y añadimos los 
# datos a la trayectoria
def update(i):

    linea.set_data(x[:i], y[:i])
    punto.set_data([x[i]], [y[i]])
    ax.set_title(f'Órbita del cometa. t = {t[i]:.3f} años')

    return linea, punto


ani = FuncAnimation(fig,update,frames=range(0, len(x), 300),interval=10)

# Opcionalmente almacenamos las figuras y animacion
# figura.savefig('Orbitas.png', dpi = 500)
# figura1.savefig('E y M.png', dpi = 500)
# ani.save('Orbita_halley.mp4', writer= 'ffmpeg')
plt.show()