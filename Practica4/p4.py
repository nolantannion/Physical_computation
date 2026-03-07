import numpy as np
import matplotlib.pyplot as plt
from funciones_t4 import euler_cromerr
'''
Estudiamos el coeficiente de Lyapunov para distintos rozamientos
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
tmax = 100
dt = 1e-2


# Condiciones iniciales del pendulo 1
theta01 = 0.20
omega01 = 0.0
estado1 = [theta01, omega01]

# Calculamos el pendulo de referencia para usar t1 
t1, sol1 = euler_cromerr(sistema=sistema, t0 = t0, tf = tmax, estado0= estado1, params= parametros, h = dt)


# Inicializar grafica y parametros
fig, ax = plt.subplots( figsize= (8,8))

ax.grid()
ax.set_xlabel('tiempo (s)')
ax.set_ylabel(r'$log(\Delta \theta)$')
ax.set_title(r'Evolución de $log(\Delta \theta )$')


rozamientos = np.linspace(0.3,0.9,4)

# Panel de 4 figuras con la variacion angular
figura, eje = plt.subplots( nrows= 2, ncols= 2, figsize = (7,7))
figura.subplots_adjust(wspace=0.4, hspace=0.4)
i,j = 0,0   # Indices para las graficas

# Calculo de los pendulos para cada rozamineto
for rozm in rozamientos:

    theta02 = theta01 - 1e-2
    omega02 = 0.0
    estado2 = [theta02, omega02]
    parametros[2] = rozm
    t1p, sol1p = euler_cromerr(sistema=sistema, t0 = t0, tf = tmax, estado0= estado1, params= parametros, h = dt)
    t2, sol2 = euler_cromerr(sistema=sistema, t0 = t0, tf = tmax, estado0= estado2, params= parametros, h = dt)

    theta2 = sol2[:,0]
    theta1p = sol1p[:,0]

    # Normalizamos entre +- pi
    theta2norm = (theta2 + np.pi) % (2*np.pi) - np.pi   #Normalizacion
    theta1pnorm = (theta1p + np.pi) % (2*np.pi) - np.pi   #Normalizacion



    # Calculamos el logaritmo de la diferencia angular con 
    # un suavizado por si acaso fueran exactamente iguales
    diff = np.abs(theta2norm-theta1pnorm) + 1e-12
    diff = np.log(diff)
        
    # Evolucion del logaritmo
    ax.scatter(t1,diff, s = 0.4, label = f'r: {rozm:.2f}' )
    ax.legend()


    # Ajuste lineal para hallar el coeficiente
    # recortando un el transitorio y la saturacion
    N =int(0.2* len(t1))
    tly = np.float32(t1[N:2*N])
    diffly = diff[N:2*N]

    # Mostramos el coeficiente calculado para cada rozamiento
    print(f'roz: {rozm:.3f}')
    p = np.polyfit(tly,diffly,1)

    if p[0] > 0:
        print('Es caotico: ', p[0], '\n')

    else:
        print('No es caotico: ', p[0], '\n')

    # Ajustamos los indices para las graficas
    if i>1:
        i = 0
        j = 1
    
    # Representamos con un scatter para mayor claridad
    eje[i,j].plot(tly, diffly, lw = 0.4, label = f'r: {rozm:.2f}')
    # Representamos el ajuste del coeficiente de Lyapunov
    yly = p[0]*tly + p[1]
    eje[i,j].plot(tly, yly, color = 'r', lw = 0.5, label= rf'$\lambda $ = {p[0]:.1e}')
    eje[i,j].legend()
    eje[i,j].set_xlabel('t (s)')
    eje[i,j].set_ylabel(r'$\log (\Delta \theta) $')


    i += 1

figura.suptitle(r'$\log (\Delta \theta)$ frente al tiempo')

# Almacenamos las figuras
fig.savefig('Coeficientes.png', dpi = 500)
figura.savefig('Coeficientes1.png', dpi = 500)

plt.show()

