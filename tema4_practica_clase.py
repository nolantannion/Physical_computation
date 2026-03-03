import numpy as np
import matplotlib.pyplot as plt
from funciones_t4 import euler_cromerr
'''
Caso con resonancia en la fuerza impulsora y rozamiento bajo
'''

# Variables y constantes
g = 9.81    # Aceleracion de la gravedad
l = 9.8   # Longitud del pendulo
roz = 1/2   # Coeficiente de rozamiento
omega_forz = 2/3  # Frecuencia de la fuerza impulsora
Af = 4  # Amplitud de la fuerza impulsora
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
tmax = 100
# Se puede variar dt pero se incluye como un parametro fijo en la funcion
# con valor 1e-3
# dt = 1e-3

# Condiciones iniciales del pendulo 1
theta01 = 0.20
omega01 = 0.0
estado1 = [theta01, omega01]

# Condiciones inciales para el pendulo 2
theta02 = 0.3
omega02 = 0.0
estado2 = [theta02, omega02]


# Solucion del sistema con las condiciones iniciales 
t1, sol1 = euler_cromerr(sistema=sistema, t0 = t0, tf = tmax, estado0= estado1, params= parametros)
t2, sol2 = euler_cromerr(sistema=sistema, t0 = t0, tf = tmax, estado0= estado2, params= parametros)



# Almacenamos los resultados de los angulos en sus propias variables
theta1 = sol1[:,0]
theta2 = sol2[:,0]

# Normalizamos los angulos entre 0 y 2pi usando division de parte entera
theta1norm = np.mod(theta1, 2*np.pi)
theta2norm = np.mod(theta2, 2*np.pi)



# Calculamos el coeficiente de lyapunov
diff = np.log(abs(theta2norm - theta1norm))

# Ajuste lineal para hallar el coeficiente
# recortando un 20 porciento de los datos
N =int( 0.2* len(t1))
p = np.polyfit(t1[N:],diff[N:],1)

if p[0] > 0:
    print('Es caotico: ', p[0])

else:
    print('No es caotico: ', p[0])




# Representamos

# Panel de figuras
fig, ax = plt.subplots(nrows=1, ncols=2, figsize= (10,6))
ax[0].plot(t1,sol1[:,0], label = 'Péndulo 1')
ax[0].plot(t1,sol2[:,0], label = 'Péndulo 2')

ax[1].plot(t1,diff, label = 'Diferencia angular')


ax[0].grid()
ax[0].legend()
ax[0].set_xlabel('tiempo (s)')
ax[0].set_ylabel(r'$\theta$ (rad)')
ax[0].set_title('Evolución angular ')


ax[1].grid()
ax[1].legend()
ax[1].set_xlabel('tiempo (s)')
ax[1].set_ylabel(r'$log(\Delta \theta)$')
ax[1].set_title(r'Evolución de $log(\Delta \theta )$')


plt.show()