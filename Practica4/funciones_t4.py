import numpy as np

# Inntegracion por euler
def euler(sistema,t0,tf,h,estado0):
    """
    INPUTS:
    - sistema: funcion que define las derivadas de cada variable
    - h: longitud del paso para integrar
    - t0: tiempo inicial
    - tf: tiempo maximo de simulación
    - estado0: estado inicial del sistema

    RETURN:
    - vars: np.array de dimensiones (N,2)
    - t: array de longitud N
    
    """
    N = int(abs(tf-t0) / h)
    vars = np.zeros((N, 2))
    vars[0] = estado0
    t = [t0]

    for i in range(N - 1):
        vars[i+1] = vars[i] + sistema(t[i],vars[i]) * h
        t.append(t[i] + h)

    return t, vars


# Integracion de Euler - Cromer
def euler_cromer(sistema, t0, tf, h, estado0):
    """
    INPUTS:
    - sistema: funcion que define las derivadas de cada variable
    - h: longitud del paso para integrar
    - t0: tiempo inicial
    - tf: tiempo maximo de simulación
    - estado0: estado inicial del sistema

    RETURN:
    - vars: np.array de dimensiones (N,2)
    - t: array de longitud N
    
    """

    N = int(abs(tf - t0) / h)
    vars = np.zeros((N, 2))
    vars[0] = estado0
    t = [t0]

    for i in range(N - 1):
        deriv = sistema(t[i],vars[i])

        vars[i+1] = vars[i]

        # velocidades - índices impares
        vars[i+1, 1::2] = vars[i, 1::2] + h * deriv[1::2]

        # posiciones - índices pares (usan v nueva)
        vars[i+1, 0::2] = vars[i, 0::2] + h * vars[i+1, 1::2]

        t.append(t[i] + h)

    return t, vars


# Inntegracion por euler
def eulerr(sistema,t0,tf,h,estado0, roz):
    """
    INPUTS:
    - sistema: funcion que define las derivadas de cada variable
    - h: longitud del paso para integrar
    - t0: tiempo inicial
    - tf: tiempo maximo de simulación
    - estado0: estado inicial del sistema
    - roz: rozamiento del sistema

    RETURN:
    - vars: np.array de dimensiones (N,2)
    - t: array de longitud N
    
    """
    N = int(abs(tf-t0) / h)
    vars = np.zeros((N, 2))
    vars[0] = estado0
    t = [t0]

    for i in range(N - 1):
        vars[i+1] = vars[i] + sistema(t[i],vars[i], roz) * h
        t.append(t[i] + h)

    return t, vars


# Integracion de Euler - Cromer
def euler_cromerr(sistema, t0, tf, estado0, params, h = 1e-3):
    """
    ## INPUTS:
    - sistema: funcion que define las derivadas de cada variable
    - h: longitud del paso para integrar
    - t0: tiempo inicial
    - tf: tiempo maximo de simulación
    - estado0: estado inicial del sistema
    - params: array que contiene la amplitud y frecuencia de la fuerza impulsora y el rozamiento 

    ## RETURN:
    - vars: np.array de dimensiones (N,2)
    - t: array de longitud N
    
    """

    N = int(abs(tf - t0) / h)
    vars = np.zeros((N, 2))
    vars[0] = estado0
    t = [t0]

    for i in range(N - 1):
        deriv = sistema(t[i],vars[i], params)

        vars[i+1] = vars[i]

        # velocidades - índices impares
        vars[i+1, 1::2] = vars[i, 1::2] + h * deriv[1::2]

        # posiciones - índices pares (usan v nueva)
        vars[i+1, 0::2] = vars[i, 0::2] + h * vars[i+1, 1::2]

        t.append(t[i] + h)

    return t, vars

