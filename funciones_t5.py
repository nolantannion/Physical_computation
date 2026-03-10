import numpy as np

'''
Script con las funciones necesarias para el tema 5
Incluye 3 distintos integradores, funciones para calcular minimos y maximos locales
y una clase para facilitar la simulacion de mas cuerpos 
'''


def vis_viva(M, r,a):
    '''
    Funcion para calcular la velocidad en un punto de la orbita de kepler
    INPUT
    M: masa que ejerce la fuerza
    r: distancia 
    a: semieje mayor
    
    RETURN
    v: modulo de v en ese punto
    '''
    G = 1
    return 2*np.pi* np.sqrt( (2/r - 1/a) )

# Buscador de maximo local
def loc_max(s):
    '''
    INPUT
    s: array del que queremos hallar el maximo

    RETURN
    ind: array con los indices de los maximos
    M: array con los valores de los maximos locales 

    '''
    ind = np.where((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]))[0] + 1
    M = s[ind]
    return ind, M


# Buscador de mínimo local
def loc_min(s):
    '''
    INPUT
    s: array del que queremos hallar los minimos

    RETURN
    i: array con los indices de los minimos
    M: array con los valores de los minimos locales 

    '''
    # Ind incluye +1 porque comparamos con un array que comienza 1 posicion por delante
    ind = np.where((s[1:-1] < s[:-2]) & (s[1:-1] < s[2:]))[0] + 1
    M = s[ind]
    return ind, M




# Clase
class Body:
    ''' Clase para simplificar la simulación de n cuerpos utilizada en la ampliacion '''
    
    G = 1

    def __init__(self, m, r, v):
        self.m = m
        self.r = r
        self.v = v
        self.f = np.zeros(2)

    def calcularf(self, b):
        rad = self.r - b.r
        R = np.linalg.norm(rad) + 1e-3
        self.f += -Body.G * self.m * b.m / R**3 * rad

    def zerof(self):
        self.f = np.zeros(2)

    def aceleracion(self):
        ''' a = F/m '''
        return self.f / self.m

    def step(self, dt):
        ''' Un paso de Euler-Cromer '''
        
        # actualizar velocidad
        self.v += self.aceleracion() * dt
        
        # actualizar posición con la velocidad nueva
        self.r += self.v * dt


# Integracion de Euler - Cromer
def euler_cromer(sistema, t0, tf, estado0, h = 1e-3):
    """
    ## INPUTS:
    - sistema: funcion que define las derivadas de cada variable
    - h: longitud del paso para integrar
    - t0: tiempo inicial
    - tf: tiempo maximo de simulación
    - estado0: estado inicial del sistema 

    ## RETURN:
    - vars: np.array de dimensiones (N,2)
    - t: array de longitud N
    
    """

    N = int(abs(tf - t0) / h)
    Nv = len(estado0)
    
    vars = np.zeros((N, Nv))
    t = np.zeros(N)
    vars[0] = estado0
    t[0] = t0

    for i in range(N - 1):
        deriv = sistema(t[i],vars[i])

        vars[i+1] = vars[i]

        # velocidades - índices impares
        vars[i+1, 1::2] = vars[i, 1::2] + h * deriv[1::2]

        # posiciones - índices pares (usan v nueva)
        vars[i+1, 0::2] = vars[i, 0::2] + h * vars[i+1, 1::2]

        t[i+1] = t[i] + h

    return t, vars

# Integrador verlet
def verlet(sistema, t0, tf, estado0, h=1e-3):

    N = int(abs(tf - t0) / h)
    Nv = len(estado0)

    vars = np.zeros((N, Nv))
    t = np.zeros(N)

    vars[0] = estado0
    t[0] = t0

    # aceleracion inicial
    deriv = sistema(t0, vars[0])
    a = deriv[1::2]

    for i in range(N-1):

        x = vars[i,0::2]
        v = vars[i,1::2]

        # posiciones
        x_new = x + v*h + 0.5*a*h**2

        vars[i+1,0::2] = x_new

        # aceleracion en nuevo punto
        temp = vars[i].copy() # array con el paso intermedio 
        temp[0::2] = x_new
        deriv_new = sistema(t[i], temp)
        a_new = deriv_new[1::2]

        # velocidades
        v_new = v + 0.5*(a + a_new)*h
        vars[i+1,1::2] = v_new

        a = a_new
        t[i+1] = t[i] + h

    return t, vars

# Integrador runge kutta de 4 orden
def rk4(sistema, t0, tf, estado0, h=1e-3):
    """
    Runge-Kutta de orden 4

    INPUT
    sistema: funcion que devuelve derivadas
    t0: tiempo inicial
    tf: tiempo final
    estado0: estado inicial
    h: paso temporal

    RETURN
    t: array tiempos
    vars: estados
    """

    N = int(abs(tf - t0) / h)
    Nv = len(estado0)

    vars = np.zeros((N, Nv))
    t = np.zeros(N)

    vars[0] = estado0
    t[0] = t0

    for i in range(N-1):

        y = vars[i]
        ti = t[i]

        k1 = sistema(ti, y)
        k2 = sistema(ti + h/2, y + h*k1/2)
        k3 = sistema(ti + h/2, y + h*k2/2)
        k4 = sistema(ti + h, y + h*k3)

        vars[i+1] = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = ti + h

    return t, vars
