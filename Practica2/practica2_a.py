import numpy as np
import matplotlib.pyplot as plt 


# Constantes 
b = 1
a = 10
v0 = [0,15]

# Creamos un array que pueda contener otros arrays con distintos tamaños
tsol = []
vsol = []

# Intervalo temporal (en segundos)

t0 = 0 
tf = 20
dt = [1,0.5,0.1,1e-2]

# Función que define la dinámica del sistema (susceptible a modificación)
def sistema(v):
    
    dv = a -b*v
    
    return dv

# Función que aplica euler
def euler(dt,t0,tf,v0):  
    N = int((tf - t0)/dt)
    t = np.linspace(t0, tf, N+1)
    v = np.zeros(N+1)
    
    v[0] = v0
    
    for n in range(0,N):
        v[n+1] = v[n] + sistema(v[n])*dt
        
    return t, v



# Calculamos las solucionesy representamos
fig, ax = plt.subplots()    
for i,dtt in enumerate(dt):
    
    t, v = euler(dtt,t0,tf,v0[0])
    
    tsol.append(t)
    vsol.append(v)
    
    # Representación de la trayectoria frente al tiempo
    ax.plot(t,v, label = f'dt: {dtt}')

print(f'Parámetros: a: {a}, b: {b}, v0: {v0}')

# La aceleración se hace 0 una vez a = bv
vlim = a/b
print(f'La velocidad límite es: {vlim} m/s')

# Ajustes de la gráfica 
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Velocidad (m/s)')
ax.set_title('Velocidad frente al tiempo ($v_0 < vlim)$')
ax.axis('tight')
ax.legend()
ax.grid()
plt.show()


# Graficamos lo que sucede si v0 > vlim

figura, eje = plt.subplots()    
for i,dtt in enumerate(dt):
    
    t, v = euler(dtt,t0,tf,v0[1])
    
    tsol.append(t)
    vsol.append(v)
    
    # Representación de la trayectoria frente al tiempo
    eje.plot(t,v, label = f'dt: {dtt}')
    
    
# Ajustes de la gráfica 
eje.set_xlabel('Tiempo (s)')
eje.set_ylabel('Velocidad (m/s)')
eje.set_title(r'Velocidad frente al tiempo ($v_0 > vlim)$')
eje.axis('tight')
eje.legend()
eje.grid()

plt.show()






