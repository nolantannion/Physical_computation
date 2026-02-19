# Numeros de Catalan
c0 = 1
cn = c0
n = 0

# Array que almacena los números calculados
n_catalan = [c0]

while cn <= 1e9:
    print(cn)
    
    n_catalan.append(cn)  # Añade el nuevo n a la lista
    cn = int((4*n+2)/(n+2)*cn)
    n = n+1
    





