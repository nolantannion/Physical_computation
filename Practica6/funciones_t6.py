import numpy as np
import random as rd


def laplace2D(A, tf, t0 = 0, dt=None, dx=1.0, dy=1.0, D = 1):
    """
    Resuelve el lalpaciano de la ecuacion de difusion en 2D y calcula el siguiente paso temporal  

    Parámetros:
    - T: array 2D [x,y] con condiciones iniciales y/o de frontera
    - dt: paso temporal, opcional, sino se calcula por convergencia
    - dx, dy: pasos espaciales
    - D: parametro de la difusión de laplace

    Devuelve:
    - t: array con el tiempo
    - Sol: array de igual tamaño a T con las soluciones
    """

    if dt is None:
        dt = 1/ (2*D* (1/dx**2 + 1/dy**2))   # estabilidad en 2D
    
    nx, ny = np.shape(A)

    NT = int(np.abs(tf-t0)/dt)


    t = np.arange(t0,tf, dt)
    T = np.zeros([nx,ny,NT])
    T[:,:,0] = A


    for n in range(NT-1):
        T[1:-1,1:-1,n+1] = T[1:-1,1:-1,n] + (D*dt/dx**2)*(
            T[2:,1:-1,n] + T[:-2,1:-1,n] +
            T[1:-1,2:,n] + T[1:-1,:-2,n] -
            4*T[1:-1,1:-1,n]
        )

    return t, T


def random_walk_2d(nw, n, dr=1.0, seed=None):
    """
    Random walk 2D con paso de módulo dr (por defecto 1).

    Params:
    - nw (int) : numero de caminantes
    - n (int)   : número de pasos
    - dr      : tamaño del paso (default=1)
    - seed      : semilla opcional

    Returns:
        x, y, dist  con dimensiones (nw,n+1)
    """
    if seed is not None:
        np.random.seed(seed)

    # Inicialización (incluye estado inicial y final)
    x = np.zeros((nw, n + 1))
    y = np.zeros((nw, n + 1))

    for i in range(1, n + 1):
        # Angulos aleatorios
        angles = np.deg2rad(np.random.randint(0, 359, size=nw))

        dx =  dr * np.cos(angles)
        dy =  dr * np.sin(angles)

        x[:, i] = x[:, i-1] + dx
        y[:, i] = y[:, i-1] + dy

    return x, y


def laplace2D_anisot(A, tf, t0 = 0, dt=None, dx=1.0, dy=1.0, D = 1):
    """
    Resuelve el lalpaciano de la ecuacion de difusion en 2D con coeficientes anisotropos

    Parámetros:
    - T: array 2D [x,y] con condiciones iniciales y/o de frontera
    - dt: paso temporal, opcional, sino se calcula por convergencia
    - dx, dy: pasos espaciales
    - D: parametro de la difusión de laplace

    Devuelve:
    - t: array con el tiempo
    - Sol: array de igual tamaño a T con las soluciones
    """

    #T = T.copy()

    Dx, Dy = 0.9534, 0.3

    D = 1 # es la suma cuadratica

    if dt is None:
        # condición de estabilidad anisótropa
        dt = 1 / (2 * (Dx/dx**2 + Dy/dy**2))/2

    NT = int(np.abs(tf-t0)/dt)

    Nx, Ny = np.shape(A)
    T = np.zeros([Nx,Ny,NT])

    T[:,:,0] = A[:,:]

    t = np.arange(t0,tf, dt)


    for n in range(NT-1):
        T[1:-1,1:-1,n+1] = T[1:-1,1:-1,n] + (Dx*dt/dx**2) * (T[2:,1:-1,n] + T[:-2,1:-1,n] - 2*T[1:-1,1:-1,n]) + (Dy*dt/dy**2) * (T[1:-1,2:,n] + T[1:-1,:-2,n] - 2*T[1:-1,1:-1,n])

    return t, T


def random_walk_2d_anisot(nw, n, dr=1.0, Dx=1.0, Dy=1.0, seed=None):
    """
    Random walk 2D anisótropo

    - nw: numero de caminantes
    - n: numero de pasos
    - dr: tamaño de paso
    - Dx, Dy: coeficientes de difusión (anisotropía)
    """

    if seed is not None:
        np.random.seed(seed)

    x = np.zeros((nw, n+1))
    y = np.zeros((nw, n+1))

    # Probabilidad de moverse en x
    p = 0.09

    for i in range(1, n+1):

        # elegir eje para cada caminante
        r = np.random.rand(nw)
        move_x = r < p
        move_y = r >= p

        # dirección aleatoria +-1
        step_x = dr * np.where(np.random.rand(nw) < 0.5, 1, -1)
        step_y = dr * np.where(np.random.rand(nw) < 0.5, 1, -1)

        # actualizar posiciones
        x[:, i] = x[:, i-1] + move_x * step_x
        y[:, i] = y[:, i-1] + move_y * step_y

    return x, y