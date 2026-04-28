import numpy as np

def laplaciano_1d(T, h):
    """
    Calcula el laplaciano 1D de T (solo interior).
    
    T: array 1D (espacio)
    h: paso espacial
    """
    L = np.zeros_like(T)
    
    L[1:-1] = (T[2:] - 2*T[1:-1] + T[:-2]) / h**2
    
    return L