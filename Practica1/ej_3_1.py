import numpy as np
import matplotlib.pyplot as plt

# Todos los datos
s = np.loadtxt('sunspots.txt', dtype = float)

# Figura con dos gráficas
fig, ax = plt.subplots(nrows=1, ncols= 2, figsize = (10 ,6))

# Plot de los datos en función de t
ax[0].plot(s[:,0],s[:,1], label = 'Todos los datos')
ax[0].set_xlabel('tiempo')
ax[0].set_ylabel('estrellas')
ax[0].grid()
ax[0].legend()


# Mil primeros y su correspondiente gráfica
ss = s[0:999,:]

ax[1].plot(ss[:,0],ss[:,1], label = 'Primeros 1000 ')
ax[1].set_xlabel('tiempo')
ax[1].set_ylabel('estrellas')
ax[1].grid()
ax[1].legend()

fig.suptitle('Estrellas en función de t')


plt.show()



# Función para calcular la media móvil 
r = 5
def run_average(x, r):
    n = len(x)
    y = []
    
    for i in range (r,n-r):
        temp = x[i-r:i+r]
        valor = 1/(2*r)*np.sum(temp)
        y.append(valor)
    
    return np.array(y,float)

# Cálculo de la media para ambos casos
nt = len(s[:,1])        # Longitud del array con todos los datos 
n1000 = len(ss[:,1])    # Longitud del array de 1000 datos

ymt = run_average(s[:,1], r)        # Medida mov de todos los datos
ym1000 = run_average(ss[:,1], r)    # Media mov de 1000 datos

xmt = s[r:nt-r, 0]          # Tiempo para media movil total
xm1000 = ss[r:n1000-r, 0]   # Tiempo para media movil de 1000 datos

# c) Datos con la media móvil
fig2, ax2 = plt.subplots(nrows=1, ncols= 2, figsize = (10 ,6))

# Plot de los datos en función de t
ax2[0].plot(s[:,0],s[:,1], label = 'Todos los datos')
ax2[0].set_xlabel('tiempo')
ax2[0].set_ylabel('estrellas')
ax2[0].grid()

# Mil primeros datos
ax2[1].plot(ss[:,0],ss[:,1], label = 'Primeros 1000 ')
ax2[1].set_xlabel('tiempo')
ax2[1].set_ylabel('estrellas')
ax2[1].grid()



# Representar las medias móviles en las gráficas correspondientes
ax2[0].plot(xmt, ymt, label = 'Media móvil ')
ax2[1].plot(xm1000, ym1000, label = 'Media móvil')

ax2[0].legend()
ax2[1].legend()
fig2.suptitle('Datos y media móvil')
plt.show()

