import cv2
import time
import matplotlib.pyplot as plt
import filtros

# 1. Cargar imagen en escala de grises [cite: 36, 38]
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE) # Carga la imagen de la raíz [cite: 36, 38]
if img is None:
    print("Error: Coloca una imagen 'input.jpg' en la carpeta.")
    exit()

def run_test(name, f_pure, f_np, data):
    # Medir Pure Python [cite: 42]
    t0 = time.time()
    res_p = f_pure(data)
    tp = time.time() - t0
    
    # Medir NumPy [cite: 43]
    t0 = time.time()
    res_n = f_np(data)
    tn = time.time() - t0
    return res_n, tp, tn

# 2. Ejecutar pruebas de rendimiento [cite: 45]
g_res, gt_p, gt_n = run_test("Gaussian", filtros.gaussian_pure, filtros.gaussian_numpy, img)
s_res, st_p, st_n = run_test("Sobel", filtros.sobel_pure, filtros.sobel_numpy, img)
m_res, mt_p, mt_n = run_test("Median", filtros.median_pure, filtros.median_numpy, img)

# 3. Mostrar Tabla de Ejecución [cite: 48, 61]
print(f"\n{'Filtro':<12} | {'Pure Python (s)':<15} | {'NumPy (s)':<10}")
print("-" * 45)
print(f"Gaussian     | {gt_p:<15.4f} | {gt_n:<10.4f}")
print(f"Sobel        | {st_p:<15.4f} | {st_n:<10.4f}")
print(f"Median       | {mt_p:<15.4f} | {mt_n:<10.4f}")

# 4. Visualización de Resultados [cite: 63, 65]
titles = ['Original', 'Gaussian Filter', 'Sobel (Edges)', 'Median Filter']
imgs = [img, g_res, s_res, m_res]
plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(imgs[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()