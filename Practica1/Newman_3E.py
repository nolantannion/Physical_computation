from pylab import imshow, show, jet, colorbar
from numpy import loadtxt
datos = loadtxt('circular.txt', float)
imshow(datos), jet(), colorbar()
show()