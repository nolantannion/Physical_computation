import numpy as np

'''
Script para calcular la velocidad ideal para obtener una orbita de 76 años.
Mediante ensayo y error acotamos el tramo de velocidades. 
Podemos modificar la precision para hallar el intervalo con este codigo
'''

# Constantes
GMSol  = 4 * np.pi**2
dt     = 0.0005
r_peri = 0.59


def detectar_periodo(vy0, T_max=200.0):
    x = r_peri; y = 0.0; vx_ = 0.0; vy_ = vy0; r_ = r_peri; t = 0.0
    y_prev = 0.0; x_prev = x; t_prev = 0.0

    for _ in range(int(T_max / dt)):
        # Calculamos nuevas aceleraciones, velocidades y posiciones
        ax = -GMSol * x / r_**3
        ay = -GMSol * y / r_**3

        vx_ += ax * dt
        vy_ += ay * dt

        xn = x + vx_ * dt
        yn = y + vy_ * dt

        rn = np.hypot(xn, yn)
        tn = t + dt 
        if t > 1.0 and y_prev * yn < 0 and xn > 0:  # Condicion de cambio de signo cuando x>0
            frac    = -y_prev / (yn - y_prev)   # Fraccion de interpolacion en el ultimo paso
            t_cruce = t_prev + frac * dt        # Tiempo de cruce 
            x_cruce = x_prev + frac * (xn - x_prev)
            r_cruce = np.hypot(x_cruce, y_prev + frac * (yn - y_prev))
            return t_cruce, r_cruce
        # Actualizamos al siguiente pasow
        x, y, r_, t, y_prev, x_prev, t_prev = xn, yn, rn, tn, yn, xn, tn
    return None, None   # Si no hay cruce no devuelve nada 



# Barremos alrededor de 11.4-11.5 y elegimos el mejor valor
print(f'{'vy0'}  {'T (a)'}  {'|T-76|'}  {'dr (UA)'}')
best_v, best_T, best_err = None, None, 999  # Iniciamos sin valores y elevado error

for vy0 in np.arange(11.47, 11.475, 0.0001):
    T_p, r_r = detectar_periodo(vy0)
    if T_p:
        err = abs(T_p - 76)
        if err < best_err:
            best_err = err; best_v = vy0; best_T = T_p
        print(f'{vy0:.4f}  {T_p:.3f}  {err:.4f}  {r_r-r_peri:.2e}')
print(f'\nMejor: vy0={best_v:.5f} UA/a  T={best_T:.3f} a')



