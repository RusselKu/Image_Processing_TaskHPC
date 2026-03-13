import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# FILTRO GAUSSIANO — Convolución vectorizada con NumPy
# ------------------------------------------------------------------
def gaussian_numpy(img):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16.0

    img_f = img.astype(np.float32)
    # Ventanas deslizantes 3x3 sobre toda la imagen (sin bucles)
    windows = np.lib.stride_tricks.sliding_window_view(img_f, (3, 3))
    result = np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))

    h, w = img.shape
    output = np.zeros((h, w), dtype=np.float32)
    output[1:h-1, 1:w-1] = result
    return np.clip(output, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------
# FILTRO SOBEL — Detección de bordes vectorizada
# ------------------------------------------------------------------
def sobel_numpy(img):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    img_f = img.astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(img_f, (3, 3))

    Gx = np.tensordot(windows, Kx, axes=([2, 3], [0, 1]))
    Gy = np.tensordot(windows, Ky, axes=([2, 3], [0, 1]))
    magnitude = np.sqrt(Gx**2 + Gy**2)

    h, w = img.shape
    output = np.zeros((h, w), dtype=np.float32)
    output[1:h-1, 1:w-1] = magnitude
    return np.clip(output, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------
# FILTRO DE MEDIANA — Selección vectorizada con stride_tricks
# ------------------------------------------------------------------
def median_numpy(img):
    windows = np.lib.stride_tricks.sliding_window_view(img, (3, 3))
    # windows: (h-2, w-2, 3, 3) → aplanar vecindad a 9 valores y sacar mediana
    medians = np.median(windows.reshape(windows.shape[0], windows.shape[1], -1), axis=2)

    h, w = img.shape
    output = np.zeros((h, w), dtype=np.uint8)
    output[1:h-1, 1:w-1] = medians.astype(np.uint8)
    return output

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    img = cv2.imread('../input.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: coloca 'input.jpg' en la raiz del proyecto.")
        exit()

    print(f"Imagen cargada: {img.shape[1]}x{img.shape[0]} pixeles\n")

    # --- Medir tiempos ---
    t0 = time.perf_counter()
    gauss_result = gaussian_numpy(img)
    t_gauss = time.perf_counter() - t0

    t0 = time.perf_counter()
    sobel_result = sobel_numpy(img)
    t_sobel = time.perf_counter() - t0

    t0 = time.perf_counter()
    median_result = median_numpy(img)
    t_median = time.perf_counter() - t0

    # --- Tabla de tiempos ---
    print(f"{'Filtro':<12} | {'Tiempo (s)':<12}")
    print("-" * 28)
    print(f"{'Gaussian':<12} | {t_gauss:<12.6f}")
    print(f"{'Sobel':<12} | {t_sobel:<12.6f}")
    print(f"{'Median':<12} | {t_median:<12.6f}")

    # --- Visualizacion ---
    titles = ['Original', 'Gaussian (NumPy)', 'Sobel (NumPy)', 'Median (NumPy)']
    images = [img, gauss_result, sobel_result, median_result]

    plt.figure(figsize=(12, 8))
    for i, (title, image) in enumerate(zip(titles, images)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.suptitle('Rivaldo - Implementacion NumPy (Vectorizada)', fontsize=14)
    plt.tight_layout()
    plt.show()
