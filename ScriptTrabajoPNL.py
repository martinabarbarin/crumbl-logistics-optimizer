import numpy as np
from scipy.optimize import minimize_scalar

def dfp(f, grad, x1, n, eps=1e-5, max_iter=100):
    # Paso de inicializacion
    # Se elige un punto inicial x1 y una matriz D1 simetrica y definida positiva
    x = np.array(x1, dtype=float)
    k = 1  # Contador de iteraciones externas
    
    print(f"{'Iter(k)':<8} | {'Paso(j)':<8} | {'f(x)':<15} | {'||grad||':<12}")
    print("-" * 55)

    for _ in range(max_iter):
        # Al empezar un ciclo nuevo (k), se suele reiniciar j y a veces la matriz D
        # Segun Bazaraa, el procedimiento se reinicia cada n pasos
        D = np.eye(n) 
        y_j = x.copy() # En el texto y1 = x1
        
        for j in range(1, n + 1):
            # Paso 1: Calculo del gradiente en el punto actual
            g_j = grad(y_j)
            norm_g = np.linalg.norm(g_j)
            
            # Condicion de parada: Si la norma del gradiente es menor a epsilon
            if norm_g < eps:
                print(f"Convergencia lograda en k={k}, j={j}")
                return y_j, f(y_j)
            
            # Direccion de busqueda dj = -Dj * gj
            d_j = -np.dot(D, g_j)
            
            # Busqueda lineal para encontrar lambda_j optimo (minimizar f(y_j + lambda*d_j))
            def line_obj(lmbda):
                return f(y_j + lmbda * d_j)
            
            res = minimize_scalar(line_obj, bounds=(0, 20), method='bounded')
            lmbda_j = res.x
            
            # Definimos el nuevo punto y_j+1
            s_j = lmbda_j * d_j
            y_next = y_j + s_j
            
            # Calculamos el cambio en el gradiente q_j
            g_next = grad(y_next)
            q_j = g_next - g_j
            
            # Paso 2: Actualizacion de la matriz Dj+1 (Ecuacion 8.30)
            # Termino 1: (sj * sj.T) / (sj.T * qj)
            term1 = np.outer(s_j, s_j) / np.dot(s_j, q_j)
            
            # Termino 2: (Dj * qj * qj.T * Dj) / (qj.T * Dj * qj)
            Dq = np.dot(D, q_j)
            term2 = np.outer(Dq, Dq) / np.dot(q_j, Dq)
            
            D = D + term1 - term2
            
            # Imprimimos progreso del bucle interno
            print(f"{k:<8} | {j:<8} | {f(y_next):<15.8f} | {norm_g:<12.6f}")
            
            # Actualizamos y_j para la siguiente sub-iteracion
            y_j = y_next
            
            # Si j = n, terminamos el ciclo interno y actualizamos el punto base x
            if j == n:
                x = y_next
                k += 1
                # Volvemos al Paso 1 reiniciando el bucle de j
                break 

    return x, f(x)

# --- EJEMPLO 1: EL CRATER ASIMETRICO (3 variables) ---
# Esta funcion mezcla potencias con un logaritmo para que el minimo sea mas dificil de hallar
def f1(x):
    # Parte de polinomios (grados 4 y 2)
    term_poly = (x[0] - 5)**4 + 2*(x[1] + 2)**2 + (x[2] - 1)**2
    # Parte logaritmica que mueve un poco el centro del crater
    term_log = np.log(1 + (x[0]**2 + x[1]**2 + x[2]**2))
    return term_poly + term_log

def grad_f1(x):
    # Calculamos las derivadas para saber hacia donde bajar
    g = np.zeros(3)
    d_log = 1 / (1 + x[0]**2 + x[1]**2 + x[2]**2)
    
    # Derivada respecto a x0, x1 y x2
    g[0] = 4 * (x[0] - 5)**3 + (2 * x[0] * d_log)
    g[1] = 4 * (x[1] + 2) + (2 * x[1] * d_log)
    g[2] = 2 * (x[2] - 1) + (2 * x[2] * d_log)
    return g

# --- EJEMPLO 2: LA RED ACOPLADA (5 variables) ---
# Aqui las variables estan mezcladas (multiplicadas entre si) dentro de una exponencial
def f2(x):
    # Suma de x_i al cuadrado multiplicada por su posicion (1, 2, 3...)
    pol = sum((i + 1) * (x[i])**2 for i in range(5))
    # Exponencial donde se multiplican variables (acoplamiento)
    exp_mix = np.exp(0.2 * (x[0]*x[1] + x[2]*x[3] + x[4]))
    return pol + exp_mix

def grad_f2(x):
    # Vector de 5 ceros para guardar el gradiente
    g = np.zeros(5)
    e_val = np.exp(0.2 * (x[0]*x[1] + x[2]*x[3] + x[4]))
    
    # Derivadas parciales aplicando la regla de la cadena para la exponencial
    g[0] = 2 * 1 * x[0] + (0.2 * x[1] * e_val)
    g[1] = 2 * 2 * x[1] + (0.2 * x[0] * e_val)
    g[2] = 2 * 3 * x[2] + (0.2 * x[3] * e_val)
    g[3] = 2 * 4 * x[3] + (0.2 * x[2] * e_val)
    g[4] = 2 * 5 * x[4] + (0.2 * 1 * e_val)
    return g

# --- PRUEBAS DE EJECUCION ---

# Prueba 1: 3 variables, empezamos en el punto [1, 1, 1]
print("=== EJECUCION EJEMPLO 1: CRATER ASIMETRICO (3 VAR) ===")
# Llamamos a la funcion corregida segun Bazaraa
x_opt1, f_opt1 = dfp(f1, grad_f1, x1=[1, 1, 1], n=3)

print(f"\nResultado Ejemplo 1:")
print(f"Punto optimo: {x_opt1}")
print(f"Valor minimo: {f_opt1:.6f}")

print("\n" + "="*60 + "\n")

# Prueba 2: 5 variables, empezamos en un punto con valores positivos y negativos
print("=== EJECUCION EJEMPLO 2: RED ACOPLADA (5 VAR) ===")
# El valor n=5 es clave para que el algoritmo reinicie el ciclo correctamente
x_opt2, f_opt2 = dfp(f2, grad_f2, x1=[0.5, -0.5, 0.5, -0.5, 0.5], n=5)

print(f"\nResultado Ejemplo 2:")
print(f"Punto optimo: {x_opt2}")
print(f"Valor minimo: {f_opt2:.6f}")