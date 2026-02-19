import numpy as np 
from scipy.constants import hbar

# Datos (en eV y kg)
E = 10
m = 9.11e-31
V = 9


k1 = np.sqrt(2*m*E)/hbar
k2 = np.sqrt(2*m*(E-V))/hbar

T = 4*k1*k2/(k1+k2)**2
R = ((k1-k2)/(k1+k2))**2

print('El coeficiente de transmision es: ', T)
print('El coeficiente de reflexion es: ', R)