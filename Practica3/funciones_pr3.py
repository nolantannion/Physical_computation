import numpy as np

rho0 = 1.27     # Densidad del aire estándar a la altura del mar
H = 1e4          # altura característica atmósfera


# Funciones para calcular las densidades según los distintos modelos
def rho_const(y):
    return rho0

def rho_iso(y):
    return rho0 * np.exp(-y / H)   

def rho_ad(y):
    
    # Parámetros característicos del aire
    a = 6.5e-3
    alfa = 2.5
    T0 = 300       # Temperatura incial Kelvin 

    factor = 1 - a*y/T0
    if factor <= 0:
        return 0   # evita valores no físicos

    return rho0 * factor**alfa



# Coeficiente de arrastre como función de v en cada dirección
def arrastre_completo(var):
    '''
    Calcula el arrastre en todas las direcciones
    INPUT 
    -var array con todas las variables 

    RETURNS
    - C: coeficiente de arrastre 
    '''
    
    C = [1,1,1]
    if abs(var[1]) > 14:
        C[0] = 14/abs(var[1])

    if abs(var[3]) > 14:
        C[1] = 14/abs(var[3])
        
    if abs(var[5]) > 14:
        C[2] = 14/abs(var[5])

    return C

# Arrastre que solo tiene en cuenta la variable x
def arrastrex(var):
    '''
    Calcula el arrastre en la direccion x
    INPUT 
    -var array que contiene todas las variables 

    RETURNS
    - C: coeficiente de arrastre 
    '''
    C = [1,0,0]
    if abs(var[1]) > 14:
        C[0] = 14/abs(var[1])

    return C



    
# Funcion que integra por euler
def euler(sistema,t0,tf,h,estado0, signo, rhof):
    """
    INPUTS:
    - sistema: funcion que define las derivadas de cada variable
    - h: longitud del paso para integrar
    - t0: tiempo inicial
    - tf: tiempo maximo de simulación
    - estado0: estado inicial del sistema
    - signo: signo que describe el sentido de la rotación
    - rho: funcion que calcula la densidad del aire a la altura dada

    RETURN:
    - vars: np.array de dimensiones (N,6)
    - t: array de longitud N
    
    """
    N = int(abs(tf-t0) / h)
    vars = np.zeros((N, 6))
    vars[0] = estado0
    t = [t0]

    for i in range(N - 1):
        rho = rhof(vars[i, 2])  # densidad en función de y
        vars[i+1] = vars[i] + sistema(vars[i], rho, signo) * h
        t.append(t[i] + h)

        # impacto con el suelo e interpolación
        if vars[i+1, 2] < 0:
            interp = (vars[i+1] + vars[i])/2
            vars[i+1] = interp
            t[i+1] = t[i] + h/2
            return t,vars[:i+2]

    return t, vars


def rk4(sistema, t0, tf, h, estado0, signo, rhof):
    '''
    INPUTS:
    sistema : funcion que define la dinamica del sistema
    t0      : tiempo inicial
    tf      : tiempo final
    h       : paso de integración
    estado0 : array con el estado inicial [x, vx, y, vy, z, vz]
    
    ADICIONAL (para los casos de proyectil)
    signo   : signo del giro (+1 slice, -1 hook)
    rhof    : metodo para calcular la densidad del aire en funcion de la altura

    RETURNS:
    t    : array de tiempos
    vars : array con la evolución del sistema,
           vars[i] = [x, vx, y, vy, z, vz] en t[i]
    '''
    
    N = int(abs((tf - t0)) / h)

    vars = np.zeros((N, 6))
    t = np.zeros(N)

    vars[0] = estado0
    t[0] = t0

    for i in range(N - 1):
        y = vars[i]

        rho1 = rhof(y[2])
        k1 = sistema(y, rho1, signo)

        rho2 = rhof(y[2] + 0.5*h*k1[2])
        k2 = sistema(y + 0.5*h*k1, rho2, signo)

        rho3 = rhof(y[2] + 0.5*h*k2[2])
        k3 = sistema(y + 0.5*h*k2, rho3, signo)

        rho4 = rhof(y[2] + h*k3[2])
        k4 = sistema(y + h*k3, rho4, signo)

        vars[i+1] = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = t[i] + h

        # impacto con el suelo (interpolación lineal)
        if vars[i+1, 2] <= 0:
            y1 = vars[i, 2]
            y2 = vars[i+1, 2]

            # Parametro para la interpolacion 
            interp =  (y2 - y1)/2 

            vars[i+1] = vars[i] + interp *h/2
            t[i+1] = t[i] + h/2

            return t[:i+2], vars[:i+2]

    return t, vars
