import numpy as np
import cv2

# --- 1. GAUSSIAN FILTER (Blurring) ---
def gaussian_pure(img):
    img_f = img.astype(np.float32)
    k = [[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]]
    h, w = img.shape
    res = np.zeros((h, w), dtype=np.float32)
    for i in range(1, h-1):
        for j in range(1, w-1):
            val = (img_f[i-1,j-1]*k[0][0] + img_f[i-1,j]*k[0][1] + img_f[i-1,j+1]*k[0][2] +
                   img_f[i,j-1]*k[1][0]   + img_f[i,j]*k[1][1]   + img_f[i,j+1]*k[1][2] +
                   img_f[i+1,j-1]*k[2][0] + img_f[i+1,j]*k[2][1] + img_f[i+1,j+1]*k[2][2])
            res[i, j] = val
    return np.clip(res, 0, 255).astype(np.uint8)

def gaussian_numpy(img):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
    return cv2.filter2D(img, -1, kernel)

# --- 2. SOBEL FILTER (Edges) ---
def sobel_pure(img):
    img_f = img.astype(np.float32)
    sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    h, w = img.shape
    res = np.zeros((h, w), dtype=np.float32)
    for i in range(1, h-1):
        for j in range(1, w-1):
            gx = 0; gy = 0
            for ki in range(3):
                for kj in range(3):
                    pixel = img_f[i-1+ki, j-1+kj]
                    gx += pixel * sx[ki][kj]
                    gy += pixel * sy[ki][kj]
            res[i, j] = np.sqrt(gx**2 + gy**2)
    return np.clip(res, 0, 255).astype(np.uint8)

def sobel_numpy(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)

# --- 3. MEDIAN FILTER (Noise Reduction) ---
def median_pure(img):
    h, w = img.shape
    res = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            vecinos = [img[i+ki-1, j+kj-1] for ki in range(3) for kj in range(3)]
            vecinos.sort()
            res[i, j] = vecinos[4]
    return res

def median_numpy(img):
    return cv2.medianBlur(img, 3)

# --- EXCLUSIVO: MEJORA VISUAL ---
def mejorar_contraste(img):
    # Aplica ecualización de histograma para que no se vea "opaca" la imagen
    return cv2.equalizeHist(img)