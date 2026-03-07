import numpy as np
import matplotlib.pyplot as plt
from funciones_t4 import euler_cromerr
from numpy.fft import rfft, rfftfreq

# Variables y constantes
g          = 9.81
l          = 9.81
omega_forz = 2/3
Af         = 1.2

def sistema(t, vars, params):
    dvars    = np.zeros_like(vars)
    dvars[0] = vars[1]
    dvars[1] = -g/l * np.sin(vars[0]) - params[2]*vars[1] + params[0]*np.sin(params[1]*t)
    return dvars

# Parámetros de integración adecuados para hacer la transformada
T_imp = 2* np.pi /3
n_per = 256 # 2^8

t0, tmax, h = 0, 3*(n_per-1)*T_imp, 1e-2

# Array que contiene los rozamientos con el de referencia primero
rozamientos_todos = np.concatenate([[1/2], np.linspace(0.48, 0.52, 3)])
estados_iniciales = np.array([[0.20, 0.0], [0.3, 0.0], [0.3, 0.0], [0.3, 0.0]])

# Calcula todas las soluciones en un solo paso
thetasol = []
t1 = None
for roz, estado0 in zip(rozamientos_todos, estados_iniciales):
    parametros = [Af, omega_forz, roz]
    t_temp, sol = euler_cromerr(sistema=sistema, t0=t0, tf=tmax, 
                                 estado0=estado0, params=parametros, h=h)
    if t1 is None:
        t1 = t_temp
    thetasol.append((sol[:, 0] + np.pi) % (2*np.pi) - np.pi)

N = int(0.2 * len(t1))  # descarta transitorio inicial

# Leyendas para incluir en la representacion
leyendas = ['Péndulo ref.'] + [f'q = {r:.3f}' for r in rozamientos_todos[1:]]

# Precomputa todas las transformadas de Fourier
freq = rfftfreq(len(t1[N:]), d=h) * 2 * np.pi  # frecuencias angulares (rad/s)
espectros = [np.abs(rfft(theta[N:]))**2 for theta in thetasol]

# Figura 1
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Potencia de Fourier')
fig.subplots_adjust(wspace=0.4, hspace=0.4)

colores = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']

# Usa FFTs precomputadas
for idx, (espectro, ley, color) in enumerate(zip(espectros, leyendas, colores)):
    j, k = idx % 2, idx // 2
    
    ax[j, k].plot(freq/omega_forz, espectro, lw=0.8, color=color)
    ax[j, k].set_xlim(0, 4)
    ax[j, k].set_xlabel(r'$\omega /\omega_{forz}$')
    ax[j, k].set_ylabel(r'$|FFT(\theta)|^2$')
    ax[j, k].set_title(ley)
    ax[j, k].grid(alpha=0.3)
    ax[j, k].axvline(1, color='red', lw=1, alpha=0.5, linestyle='--', label=r'$\omega_{imp}$')
    ax[j, k].legend(fontsize=8)

# Figura 2 superposicion de todos
figura, eje = plt.subplots(figsize=(9, 5))
figura.suptitle('Comparación de Potencia Espectral')

for espectro, ley, color in zip(espectros, leyendas, colores):
    eje.plot(freq/omega_forz, espectro, lw=0.7, alpha=0.8, color=color, label=ley)

eje.axvline(1, color='red', lw=0.7, alpha=0.5, linestyle='--', label=r'$\omega_{imp}$')
eje.set_xlim(0, 4)
eje.set_xlabel(r'$\omega /\omega_{forz}$')
eje.set_ylabel(r'$|FFT(\theta)|^2$')
eje.set_title('Todos los péndulos')
eje.legend()
eje.grid(alpha=0.3)

plt.show()