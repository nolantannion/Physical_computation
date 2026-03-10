import numpy as np
import matplotlib.pyplot as plt
from funciones_t5 import verlet, euler_cromer, rk4
from matplotlib.animation import FuncAnimation

'''
Comparacion de la energias y momentos angulares con distintos integradores
'''

# Constantes
Ms = 4*np.pi**2
M_sol = 2e30
Mj = 1.9e27

q = Mj/M_sol
GMj = q*Ms

# Jupiter
Rj = 5.2
Tj = 11.86
omegaj = 2*np.pi/Tj

# Halley
r0 = 0.59
vy0 = 11.47270
Periodo = 75.979
T = 1*Periodo
# T = 10 * Periodo # opcional para un mayor duracion


dt = 0.0005


# Sistema Sol
def sistema(t,estado):

    x,vx,y,vy = estado
    r = np.sqrt(x**2+y**2)

    ax = -Ms*x/r**3
    ay = -Ms*y/r**3

    return np.array([vx,ax,vy,ay])


# condiciones iniciales
estado0 = np.array([r0,0,0,vy0])


# Array con listas de pareja nombre / funcion
integradores = [("Euler-Cromer", euler_cromer),("Verlet", verlet),("RK4", rk4)]


figura1, eje1 = plt.subplots(nrows=1,ncols=2, figsize=(10,6))
fig, eje = plt.subplots( figsize = (8,6))
figura1.subplots_adjust( wspace= 0.3)

print("Variacion de energia por unidad de masa")
# Bucle para recorrer el array con los nombres y funciones
# sin almacenar los resultados
for nombre, integrador in integradores:

    t, vars = integrador(sistema,0,T,estado0,dt)

    x = vars[:,0]
    y = vars[:,2]
    vx = vars[:,1]
    vy = vars[:,3]

    r = np.sqrt(x**2 + y**2)
    v = np.sqrt(vx**2 + vy**2)

    Ec = 0.5*v**2
    Ep = -Ms/r
    Et = Ec + Ep

    L = x*vy - y*vx

    VarE = Et[0] - Et[-1]

    print(f'{nombre}: {VarE:.3e}')

    eje1[0].plot(t, Ec, lw=0.8, label=f'Ec {nombre}')
    eje1[0].plot(t, Ep, lw=0.8, label=f'Ep {nombre}')

    eje1[1].plot(t, L, lw=0.8, label=nombre)

    eje1[0].set_xlabel('tiempo (años)')
    eje1[1].set_xlabel('tiempo (años)')
    
    eje1[0].set_ylabel(r'Energía por ud de masa ($Ua^2/años^2$)')
    eje1[1].set_ylabel(r'Momento angular por ud de masa (años)')
    

    eje.scatter(t,Et, label =f'{nombre}', s = 0.1)
    

eje.set_ylabel(r'E total por ud de masa ($Ua^2/años^2$)')
eje.set_xlabel('tiempo (años)')
eje.set_title('Comparación de Energía total')
eje.legend(loc = 'upper center')

eje1[0].legend()
eje1[1].legend()

eje.legend()
figura1.suptitle('Comparación de magnitudes según el integrador')


# Opcionalmente descomentar para guardar
figura1.savefig('Energia_momento1.png', dpi = 500)
fig.savefig('Energias1.png', dpi = 500)

plt.show()
