# Proyecto de Procesamiento de Imágenes - Unidad 2

Este documento detalla la distribución de tareas y los requisitos técnicos para la implementación de filtros de procesamiento de imágenes utilizando tres enfoques computacionales distintos. El objetivo es comparar el rendimiento y la eficiencia de cada método.

## 👥 Integrantes y Responsabilidades

Cada integrante trabajará en su carpeta asignada, implementando los mismos tres filtros (Gaussian, Sobel y Median) bajo su metodología específica.

### 1. Damian - Python Puro
*   **Carpeta:** `PurePython(Damian)/`
*   **Metodología:** Implementación en Python estándar sin el uso de librerías externas de optimización (prohibido usar NumPy para los cálculos).
*   **Enfoque:** Uso de listas anidadas para representar imágenes y bucles manuales para las operaciones de convolución y filtrado.

### 2. Rivaldo - NumPy
*   **Carpeta:** `NumPy (Rivaldo)/`
*   **Metodología:** Uso de **NumPy** para optimizar las operaciones mediante vectorización.
*   **Enfoque:** Evitar bucles explícitos en la medida de lo posible, utilizando operaciones de matrices y funciones optimizadas de la librería NumPy para mejorar el tiempo de ejecución.

### 3. Russel - NumPy + Cython
*   **Carpeta:** `NumPy + Cython(Russel)/`
*   **Metodología:** Combinación de **NumPy** con **Cython** para obtener el máximo rendimiento.
*   **Enfoque:** Escribir el código crítico en Cython (archivo `.pyx`), compilarlo a C y utilizarlo desde Python. Se busca reducir el overhead de Python en las tareas de procesamiento intensivo.

---

## 🛠️ Filtros a Implementar

Todos los integrantes deben implementar los siguientes filtros aplicados a una imagen en escala de grises:

### 1. Filtro Gaussiano (Desenfoque/Reducción de Ruido)
*   **Objetivo:** Suavizar la imagen mediante un promedio ponderado.
*   **Kernel:** 3x3 estándar.
    ```
    G = 1/16 * [[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]]
    ```
*   **Operación:** Convolución del kernel sobre la imagen original.

### 2. Filtro Sobel (Detección de Bordes)
*   **Objetivo:** Resaltar los bordes calculando el gradiente de intensidad.
*   **Kernels:**
    *   **X-Direction (Sx):** `[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]`
    *   **Y-Direction (Sy):** `[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]`
*   **Resultado:** Calcular la magnitud del gradiente: `G = sqrt(Sx² + Sy²)`.

### 3. Filtro de Mediana (Reducción de Ruido Salt-and-Pepper)
*   **Objetivo:** Filtro no lineal para eliminar ruido manteniendo los bordes.
*   **Operación:** Reemplazar cada píxel por el valor mediano de su vecindad de 3x3.

---

## 📋 Tareas Generales del Proyecto

1.  **Carga de Imágenes:** Convertir la imagen a escala de grises y mostrar la original.
2.  **Implementación:** Cada integrante desarrolla los 3 filtros en su carpeta.
3.  **Comparación de Rendimiento:**
    *   Medir el tiempo de ejecución de cada implementación.
    *   Generar una tabla comparativa de tiempos: Python Puro vs. NumPy vs. Cython.
4.  **Visualización:** Mostrar los resultados de las imágenes filtradas para cada método.
5.  **Análisis:** Discutir las ventajas y desventajas (trade-offs) entre facilidad de implementación y velocidad de ejecución.

---

## 📁 Estructura del Repositorio

*   `/PurePython(Damian)/`: Código de Damian.
*   `/NumPy (Rivaldo)/`: Código de Rivaldo.
*   `/NumPy + Cython(Russel)/`: Código de Russel.
*   `Image Processing - Unit 2.pdf`: Especificaciones originales.
*   `PLAN_TRABAJO.md`: Este documento.
