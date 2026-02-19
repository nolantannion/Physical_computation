from vpython import canvas, color, rate, sphere, vector
from math import cos, sin, pi
from numpy import arange

canvas(x = 500, y = 200, width = 500, height = 500, center = vector(0,0,1),
        forward = vector(0, 0, -1), background = color.blue, foreground = vector(1,1,0))

s = [ ]
s.append(sphere(pos=vector(1, 0, 0), radius = 0.25, color = color.yellow))
         
for theta in arange(0, 10*pi,0.1):
    rate(30)
    x = cos(theta)
    y = sin(theta)
    s[0].pos = vector(x, y, 0)
