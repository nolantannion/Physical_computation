import numpy as np
import matplotlib.pyplot as plt 


# Consultams si desea usar valores personalizados
usuario = input('Desea introducir valores manualmente? (Si/No) ').lower()

if usuario == 'si':
    a = float(input('Valor de a: '))
    
    b = float(input('Valor de b: '))
    
    v0 = float(input('Valor de v0: '))
    
elif usuario == 'no':

    b = 1
    a = 10
    v0 = 0
    
else:
    print('Respuesta no válida, usando valores predefinidos')
    b = 1
    a = 10
    v0 = 0

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
    t = np.arange(t0, tf, dt)
    v = np.zeros(len(t))
    
    v[0] = v0
    
    for n in range(0,len(t) -1):
        v[n+1] = v[n] + sistema(v[n])*dt
        
    return t, v



# Calculamos las soluciones y representamos
fig, ax = plt.subplots()    
for i,dtt in enumerate(dt):
    
    t, v = euler(dtt,t0,tf,v0)
    
    tsol.append(t)
    vsol.append(v)
    
    # Representación de la trayectoria frente al tiempo
    ax.plot(t,v, label = f'dt: {dtt}')

print(f'Parámetros: a: {a}, b: {b}, v0: {v0}')


# Ajustes de la gráfica 
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Velocidad (m/s)')
ax.axis('tight')
ax.legend()
ax.set_title('Velocidad frente al tiempo')
ax.grid()
plt.show()



