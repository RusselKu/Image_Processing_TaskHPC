import sys
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# Agregar rutas de los modulos
sys.path.append(os.path.abspath("../PurePython(Damian)"))
sys.path.append(os.path.abspath("../NumPy (Rivaldo)"))
sys.path.append(os.path.abspath("../NumPy + Cython(Russel)"))

import filtros as pure_filtros
import wasa as numpy_filtros
try:
    import filtros_cython as cython_filtros
except ImportError:
    print("Error: No se pudo importar filtros_cython.")
    sys.exit(1)

# Cargar imagen
img = cv2.imread("../input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: No se encontró 'input.jpg'")
    sys.exit(1)

def measure(func, data):
    start = time.perf_counter()
    func(data)
    return time.perf_counter() - start

# Ejecutar mediciones
print("Calculando tiempos para la gráfica...")
filters_names = ['Gaussian', 'Sobel', 'Median']
data = {
    'Gaussian': [
        measure(pure_filtros.gaussian_pure, img),
        measure(numpy_filtros.gaussian_numpy, img),
        measure(cython_filtros.gaussian_filter_cython, img)
    ],
    'Sobel': [
        measure(pure_filtros.sobel_pure, img),
        measure(numpy_filtros.sobel_numpy, img),
        measure(cython_filtros.sobel_filter_cython, img)
    ],
    'Median': [
        measure(pure_filtros.median_pure, img),
        measure(numpy_filtros.median_numpy, img),
        measure(cython_filtros.median_filter_cython, img)
    ]
}

# Generar Gráfica
x = np.arange(len(filters_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(x - width, [data[f][0] for f in filters_names], width, label='Pure Python', color='indianred')
ax.bar(x, [data[f][1] for f in filters_names], width, label='NumPy', color='skyblue')
ax.bar(x + width, [data[f][2] for f in filters_names], width, label='Cython', color='seagreen')

ax.set_ylabel('Tiempo (segundos) - Escala Logarítmica')
ax.set_title('Comparativa de Rendimiento: Pure Python vs NumPy vs Cython')
ax.set_xticks(x)
ax.set_xticklabels(filters_names)
ax.legend()
ax.set_yscale('log')
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()
plt.savefig('comparativa_tiempos.png')
print("Gráfica guardada exitosamente como 'comparativa_tiempos.png'")
