import numpy as np
cimport numpy as cnp
import cython
from libc.math cimport sqrt

cnp.import_array()

# Macro de intercambio rápido con noexcept para evitar el GIL
cdef inline void swap(unsigned char *a, unsigned char *b) noexcept nogil:
    cdef unsigned char t
    if a[0] > b[0]:
        t = a[0]
        a[0] = b[0]
        b[0] = t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gaussian_filter_cython(unsigned char[:, :] image):
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] output = np.empty((h, w), dtype=np.uint8)
    cdef unsigned char[:, :] dst = output
    cdef int i, j
    cdef float val

    with nogil:
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                val = (image[i-1, j-1] + image[i-1, j] * 2 + image[i-1, j+1] +
                       image[i, j-1]   * 2 + image[i, j]   * 4 + image[i, j+1]   * 2 +
                       image[i+1, j-1] + image[i+1, j] * 2 + image[i+1, j+1]) / 16.0
                dst[i, j] = <unsigned char>(val + 0.5)
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sobel_filter_cython(unsigned char[:, :] image):
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] output = np.empty((h, w), dtype=np.uint8)
    cdef unsigned char[:, :] dst = output
    cdef int i, j
    cdef float gx, gy, mag

    with nogil:
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                gx = (-1.0 * image[i-1, j-1] + 1.0 * image[i-1, j+1] +
                      -2.0 * image[i, j-1]   + 2.0 * image[i, j+1] +
                      -1.0 * image[i+1, j-1] + 1.0 * image[i+1, j+1])
                
                gy = (-1.0 * image[i-1, j-1] - 2.0 * image[i-1, j] - 1.0 * image[i-1, j+1] +
                       1.0 * image[i+1, j-1] + 2.0 * image[i+1, j] + 1.0 * image[i+1, j+1])
                
                mag = sqrt(gx*gx + gy*gy)
                if mag > 255: mag = 255
                dst[i, j] = <unsigned char>mag
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
def median_filter_cython(unsigned char[:, :] image):
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] output = np.empty((h, w), dtype=np.uint8)
    cdef unsigned char[:, :] dst = output
    cdef int i, j
    cdef unsigned char p[9]

    with nogil:
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Cargar ventana 3x3
                p[0]=image[i-1,j-1]; p[1]=image[i-1,j]; p[2]=image[i-1,j+1]
                p[3]=image[i,j-1];   p[4]=image[i,j];   p[5]=image[i,j+1]
                p[6]=image[i+1,j-1]; p[7]=image[i+1,j]; p[8]=image[i+1,j+1]
                
                # Sorting Network de 9 elementos (noexcept + nogil)
                swap(&p[1], &p[2]); swap(&p[4], &p[5]); swap(&p[7], &p[8])
                swap(&p[0], &p[1]); swap(&p[3], &p[4]); swap(&p[6], &p[7])
                swap(&p[1], &p[2]); swap(&p[4], &p[5]); swap(&p[7], &p[8])
                swap(&p[0], &p[3]); swap(&p[3], &p[6]); swap(&p[0], &p[3])
                swap(&p[1], &p[4]); swap(&p[4], &p[7]); swap(&p[1], &p[4])
                swap(&p[2], &p[5]); swap(&p[5], &p[8]); swap(&p[2], &p[5])
                swap(&p[1], &p[3]); swap(&p[5], &p[7]); swap(&p[2], &p[4])
                swap(&p[4], &p[6]); swap(&p[2], &p[3]); swap(&p[5], &p[6])
                swap(&p[4], &p[5]); swap(&p[3], &p[4])
                
                dst[i, j] = p[4]
    return output
