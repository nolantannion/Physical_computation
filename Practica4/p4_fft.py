import numpy as np
import matplotlib.pyplot as plt
from funciones_t4 import euler_cromerr
from numpy.fft import rfft, rfftfreq

# Variables y constantes
g          = 9.81
l          = 9.81
roz        = 1/2
omega_forz = 2/3
Af         = 1.2

parametros = [Af, omega_forz, roz]

def sistema(t, vars, params):
    dvars    = np.zeros_like(vars)
    dvars[0] = vars[1]
    dvars[1] = -g/l * np.sin(vars[0]) - params[2]*vars[1] + params[0]*np.sin(params[1]*t)
    return dvars

# Parámetros de integración adecuados para hacer la transformada
T_imp = 2* np.pi /omega_forz
n_per = 256 # 2^m

t0, tmax, h = 0, n_per*T_imp, 1e-2
estado01 = [0.20,0.0]

# Calculamos 
t1, sol1 = euler_cromerr(sistema=sistema, t0=t0, tf=tmax, estado0=estado01, params=parametros, h=h)
theta1 = (sol1[:, 0] + np.pi) % (2*np.pi) - np.pi

N = int(0.2 * len(t1))   # descarta transitorio inicial

# Array para almacenar las soluciones angulares
thetasol = [theta1]

# Array que contiene los rozamientos
rozamientos = np.linspace(0.48, 0.52, 3)

# Hacemos los calculos con todos los rozamientos
for rozm in rozamientos:
    parametros[2] = rozm
    _, sol2 = euler_cromerr(sistema=sistema, t0=t0, tf=tmax,
                             estado0=[0.3, 0.0], params=parametros, h=h)
    thetasol.append((sol2[:, 0] + np.pi) % (2*np.pi) - np.pi)

# Leyendas para incluir en la representacion
leyendas = ['Péndulo ref.'] + [f'q = {r:.3f}' for r in rozamientos]

# Transformada de Fourier 
freq = rfftfreq(len(t1[N:]), d=h) * 2 * np.pi   # frecuencias angulares (rad/s)

# Figura 1
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Espectro de Fourier — régimen estacionario', fontsize=13)
fig.subplots_adjust(wspace=0.4, hspace=0.4)

colores = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']

# Tomamos el indice con enumerate y las propiedades de la leyenda y el angulo con zip
for idx, (theta, ley) in enumerate(zip(thetasol, leyendas)):
    j, k  = idx % 2, idx // 2
    # Calculamos la transformada real
    espectro = np.abs(rfft(theta[N:]))

    # Representamos en un grafico con sus parametros
    ax[j, k].plot(freq/omega_forz, espectro, lw=0.8, color=colores[idx])
    ax[j, k].set_xlim(0, 4)          # rango relevante en rad/s
    ax[j, k].set_xlabel(r'$\omega /\omega_{forz}$')
    ax[j, k].set_ylabel(r'$|FFT(\theta)|$')
    ax[j, k].set_title(ley)
    ax[j, k].grid(alpha=0.3)

    # Marca la frecuencia de la fuerza impulsora
    ax[j, k].axvline(omega_forz/omega_forz, color='red', lw=1, alpha = 0.5, linestyle='--', label=r'$\omega_{imp}$')
    ax[j, k].legend(fontsize=8)

# Figura 2 superposicion de todos
figura, eje = plt.subplots(figsize=(9, 5))
figura.suptitle('Espectro de Fourier — comparación', fontsize=13)

# Usamos zip para tomar los elemtos de las listas en orden
# ya que son propiedades de un mismo calculo 
for theta, ley, color in zip(thetasol, leyendas, colores):
    espectro = np.abs(rfft(theta[N:]))
    eje.plot(freq/omega_forz, espectro, lw=0.7, alpha=0.8, color=color, label=ley)

# Linea vertical que marca la frecuencia impulsora
eje.axvline(omega_forz/omega_forz, color='red', lw=1, alpha = 0.5, linestyle='--', label=r'$\omega_{imp}$')
eje.set_xlim(0, 4)
eje.set_xlabel(r'$\omega /\omega_{forz}$')
eje.set_ylabel(r'$|FFT(\theta)|$')
eje.set_title('Todos los péndulos')
eje.legend()
eje.grid(alpha=0.3)

plt.show()