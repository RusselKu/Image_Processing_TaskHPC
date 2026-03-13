# Image Processing - Unit 2

Este proyecto implementa y compara el rendimiento de tres filtros de procesamiento de imágenes (Gaussian, Sobel y Median) utilizando tres enfoques computacionales diferentes:

1. **Pure Python** (Sin librerías de optimización)
2. **NumPy** (Operaciones vectorizadas)
3. **NumPy + Cython** (Operaciones optimizadas compiladas en C)

## 🛠️ Setup Instructions

### 1. Clonar el repositorio e ingresar al directorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd Image_Processing_TaskHPC
```

### 2. Crear y activar un entorno virtual (Recomendado)
* **Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **Linux/macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instalar dependencias
Instala todas las librerías necesarias con el archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Compilar el código Cython (Carpeta de Russel)
Para que las funciones de Cython estén disponibles para importar en Python, primero debes compilar el archivo `.pyx`.

Dirígete a la carpeta correspondiente y ejecuta el archivo de configuración (asegúrate de renombrar tu archivo `cython.pyx` a `cython_filters.pyx` para evitar conflictos de nombres, y utiliza el `setup.py` provisto):

```bash
cd " NumPy + Cython(Russel)/"
python setup.py build_ext --inplace
```
Esto generará un archivo `.so` (Linux/Mac) o `.pyd` (Windows) que podrás importar en tu script principal de Python de la siguiente manera:
```python
import cython_filters
```

## 📊 Medición de Rendimiento
Para utilizar los profilers incluidos en el proyecto:
* **Line Profiler:** Usa el decorador `@profile` en tus funciones y ejecuta tu script con:
  ```bash
  kernprof -l -v tu_script.py
  ```
* **Memory Profiler:** Usa el decorador `@profile` e inicia tu script con:
  ```bash
  python -m memory_profiler tu_script.py
  ```
