import numpy as np


import numpy as np

class Body:
    ''' Clase para simplificar la simulación de n cuerpos '''
    
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