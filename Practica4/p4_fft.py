import numpy as np
import matplotlib.pyplot as plt
from funciones_t4 import euler_cromerr
from numpy.fft import rfft, rfftfreq

''' 
Ampliacion que incluye el calculo de la transformada de fourier 
 en el regimen estacionario y una representacion de la potencia espectral 
'''


# Parámetros y constantes
g = 9.81
l = 9.81
omega_forz = 2/3
Af = 1.2
roz = 0.5

# Dinamica del sistema
def sistema(t, vars, params):
    dvars = np.zeros_like(vars)
    dvars[0] = vars[1]
    dvars[1] = -g/l*np.sin(vars[0]) \
               - params[2]*vars[1] \
               + params[0]*np.sin(params[1]*t)
    return dvars

# Parametros de integracion
T_forz = 2*np.pi/omega_forz # Periodo del sistema forzado
h = T_forz/100          # Definimos un diferencial adecuado al periodo
T_total = 2000*T_forz    # Estudiamos a lo largo de 2000 periodos

t0, tmax = 0, T_total

# Cálculo soluciones
rr = np.linspace(0.3, 0.9, 3)
rozamientos = [roz] + list(rr)  # Juntamos todos los rozamientos en un array
# Array para almacenar las soluciones angulares
thetasol = []

# Calculamos la evolucion temporal para todos los rozamientos
for q in rozamientos:
    params = [Af, omega_forz, q]
    estado0 = [0.20, 0.0] if q == roz else [0.3, 0.0]

    _, sol = euler_cromerr(
        sistema=sistema,
        t0=t0,
        tf=tmax,
        estado0=estado0,
        params=params,
        h=h
    )

    theta = (sol[:,0] + np.pi) % (2*np.pi) - np.pi
    thetasol.append(theta)

# Procesado para optimizar FFT
frac_trans = 0.4  # Eliminamos un 30% de los datos para considerar el regimen estacionario
i0 = int(frac_trans * len(thetasol[0])) # Calculamos el indice correspondiente

theta_est = thetasol[0][i0:]    # array con el periodo estacionario 
N_disp = len(theta_est)         # longitud del array
m = int(np.floor(np.log2(N_disp)))  # maximo de potencias de 2 que podemos incluir en el array
N_fft = 2**m                       # Numero de puntos a incluir, en base 2 

# Calculamos las frecuencias
freq = rfftfreq(N_fft, d=h) * 2*np.pi

# FFT para todas las señales
espectros = []

for theta in thetasol:
    # Tomamos los ultimos puntos en potencia de 2 y calculamos FFT 
    theta_fft = theta[-N_fft:] 
    espectros.append(np.abs(rfft(theta_fft))**2)

# Definimos arrays con las leyendas y colores
leyendas = ['Péndulo ref.'] + [f'q = {r:.3f}' for r in rr]
colores = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']

# Figura 1
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Potencia de Espectral')
fig.subplots_adjust(wspace=0.4, hspace=0.4)

# Tomamos el indice con enumerate y los valores de espectro y leyenda en orden con zip
for idx, (esp, ley) in enumerate(zip(espectros, leyendas)):
    # Usamos los operadores resto y division entera para 
    # organizar los indices en un panel
    j, k = idx % 2, idx // 2

    ax[j, k].plot(freq/omega_forz, esp, lw=0.8, color=colores[idx])
    #ax[j, k].axvline(1, color='red', ls='--', alpha=0.5)
    ax[j, k].set_xlim(0, 4)
    ax[j, k].set_xlabel(r'$\omega/\omega_{forz}$')
    ax[j, k].set_ylabel(r'$|FFT(\theta)|^2$')
    ax[j, k].set_title(ley)
    ax[j, k].grid(alpha=0.3)

# Figura 2
fig2, eje = plt.subplots(figsize=(9,5))
fig2.suptitle('Comparación de Potencia Espectral')

# Usamos zip para tomar los elemtos de las listas en orden
for esp, ley, color in zip(espectros, leyendas, colores):
    eje.plot(freq/omega_forz, esp, lw=0.7, alpha=0.8,
             color=color, label=ley)

# Marcamos la frecuencia impulsora y los parametros de la grafica
# eje.axvline(1, color='red', ls='--', alpha=0.5)
eje.set_xlim(0, 4)
eje.set_xlabel(r'$\omega/\omega_{forz}$')
eje.set_ylabel(r'$|FFT(\theta)|^2$')
eje.legend()
eje.grid(alpha=0.3)

# Opcionalmente, almacenamos las figuras
# fig.savefig('potencias.png', dpi = 500)
# fig2.savefig('P_conj.png', dpi = 500)

plt.show()