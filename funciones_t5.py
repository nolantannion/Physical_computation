import numpy as np

G = 1

class Body:
    def __init__(self, m, r, v):
        self.m = m
        self.r = r
        self.v = v
        self.f = np.zeros(2)

    # Calcular fuerza
    def computef(self,other, tol = 1e-3):
        R = other.r - self.r
        dist = np.linalg.norm(R) + tol
        F = -G*self.m *other.m / (dist + tol)
        return F
    
    # Reinciar fuerza de la particula
    def resetf(self):
        self.f[:] = 0.0

