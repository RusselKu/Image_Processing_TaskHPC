import sys
import os
import cv2
import time
import numpy as np

# Configurar rutas para las importaciones
sys.path.append(os.path.abspath("../PurePython(Damian)"))
sys.path.append(os.path.abspath("../NumPy (Rivaldo)"))
sys.path.append(os.path.abspath("../NumPy + Cython(Russel)"))

# Importar las 3 implementaciones
import filtros as pure_filtros
import wasa as numpy_filtros
try:
    import filtros_cython as cython_filtros
except ImportError:
    print("ERROR: Asegúrate de haber compilado el módulo de Cython.")
    sys.exit(1)

# Cargar imagen
img = cv2.imread("../input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("ERROR: Coloca una imagen 'input.jpg' en la raíz.")
    sys.exit(1)

def run_performance_test():
    print(f"--- Iniciando Profiling de Procesamiento de Imagen ---")
    print(f"Resolución: {img.shape[1]}x{img.shape[0]}\n")

    results = {}
    
    # Pruebas para cada filtro
    for filter_name in ['Gaussian', 'Sobel', 'Median']:
        print(f"Midiendo rendimiento: {filter_name} Filter...")
        
        if filter_name == 'Gaussian':
            f_p = pure_filtros.gaussian_pure
            f_n = numpy_filtros.gaussian_numpy
            f_c = cython_filtros.gaussian_filter_cython
        elif filter_name == 'Sobel':
            f_p = pure_filtros.sobel_pure
            f_n = numpy_filtros.sobel_numpy
            f_c = cython_filtros.sobel_filter_cython
        else: # Median
            f_p = pure_filtros.median_pure
            f_n = numpy_filtros.median_numpy
            f_c = cython_filtros.median_filter_cython

        t0 = time.perf_counter()
        f_p(img)
        t_pure = time.perf_counter() - t0

        t0 = time.perf_counter()
        f_n(img)
        t_numpy = time.perf_counter() - t0

        t0 = time.perf_counter()
        f_c(img)
        t_cython = time.perf_counter() - t0

        results[filter_name] = (t_pure, t_numpy, t_cython)

    # Imprimir Tabla Final
    print("\n" + "="*85)
    print(f"{'FILTRO':<12} | {'PURE PYTHON (s)':<18} | {'NUMPY (s)':<12} | {'CYTHON+NUMPY (s)':<18} | {'SPEEDUP'}")
    print("-"*85)
    for name, (tp, tn, tc) in results.items():
        speedup = tp / tc if tc > 0 else 0
        print(f"{name:<12} | {tp:<18.6f} | {tn:<12.6f} | {tc:<18.6f} | {speedup:.1f}x vs Pure")
    print("="*85)

if __name__ == "__main__":
    run_performance_test()
