import cv2
import time
import matplotlib.pyplot as plt
import filtros

# 1. Cargar imagen
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: No se encontró 'input.jpg'.")
    exit()

def run_benchmark(name, f_pure, f_np, data):
    start = time.time()
    res_p = f_pure(data)
    tp = time.time() - start
    
    start = time.time()
    res_n = f_np(data)
    tn = time.time() - start
    return res_n, tp, tn

# 2. Ejecutar Benchmark
print("Procesando filtros y midiendo rendimiento...")
g_img, g_tp, g_tn = run_benchmark("Gaussian", filtros.gaussian_pure, filtros.gaussian_numpy, img)
s_img, s_tp, s_tn = run_benchmark("Sobel", filtros.sobel_pure, filtros.sobel_numpy, img)
m_img, m_tp, m_tn = run_benchmark("Median", filtros.median_pure, filtros.median_numpy, img)

# 3. Tabla de Tiempos
print(f"\n{'FILTRO':<12} | {'PURE PYTHON (s)':<18} | {'NUMPY (s)':<12}")
print("-" * 50)
print(f"Gaussian     | {g_tp:<18.4f} | {g_tn:<12.4f}")
print(f"Sobel        | {s_tp:<18.4f} | {s_tn:<12.4f}")
print(f"Median       | {m_tp:<18.4f} | {m_tn:<12.4f}")

# 4. Mejorar contraste para el resultado visual (Opcional, pero recomendado)
g_vis = filtros.mejorar_contraste(g_img)
s_vis = filtros.mejorar_contraste(s_img) # Sobel suele verse mejor sin ecualizar, pero probemos
m_vis = filtros.mejorar_contraste(m_img)

# 5. Visualización y Guardado
titles = ['Original', 'Gaussian (Enhanced)', 'Sobel (Edges)', 'Median (Enhanced)']
imgs = [img, g_vis, s_vis, m_vis]

plt.figure(figsize=(14, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(imgs[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig('comparativa_filtros_HD.png')
print("\n[INFO] Resultados guardados en 'comparativa_filtros_HD.png'")
plt.show()