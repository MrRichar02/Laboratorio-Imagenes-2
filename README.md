# Clasificador de Imágenes - Animals10

**Integrantes:**  
- Ricardo Medina Herrera  
- Santiago Villegas Naranjo  

---

## Descripción

Este proyecto consiste en un clasificador de imágenes de animales utilizando técnicas de visión por computadora y redes neuronales.  

El sistema permite cargar una imagen desde una interfaz gráfica hecha con Tkinter y clasificarla entre las siguientes categorías:

- butterfly
- cat
- chicken
- cow
- dog
- elephant
- horse
- sheep
- spyder
- squirrel

Para el procesamiento de características se trabajó con dos descriptores:

- **HOG (Histogram of Oriented Gradients)**
- **LBP (Local Binary Patterns)**

Y para la clasificación se utilizó un modelo **MLP (Multi-Layer Perceptron)** entrenado en PyTorch.

---

# Estructura del proyecto

```bash
.
├── butterfly.jpg
├── cat.jpg
├── chicken.png
├── cow.jpg
├── dog.png
├── elephant.jpg
├── filtros.ipynb
├── horse.jpg
├── lab2_imagenes.ipynb
├── main.py
├── mejor_modelo_hog.pth
├── mejor_modelo_lbp.pth
├── pyproject.toml
├── README.md
├── scaler_hog.pkl
├── scaler_lbp.pkl
├── sheep.png
├── spider.jpg
├── squirrel.jpg
└── uv.lock
```

---

# Tecnologías utilizadas

- Python
- Tkinter
- OpenCV
- PyTorch
- NumPy
- PIL
- scikit-image
- joblib

---

# Archivos principales

## `lab2_imagenes.ipynb`

Notebook principal donde se realiza:

- Carga y organización del dataset
- Preprocesamiento de imágenes
- Extracción de características HOG y LBP
- Entrenamiento de modelos MLP
- Evaluación de resultados
- Guardado de modelos `.pth`
- Guardado de scalers `.pkl`

---

## `filtros.ipynb`

Notebook utilizado para experimentar y probar distintos filtros y transformaciones de imágenes antes del entrenamiento.

Aquí se trabajó con procesos como:

- Conversión a escala de grises
- Equalización de histograma
- Resize de imágenes
- Visualización de resultados

---

## `main.py`

Aplicación principal con interfaz gráfica en Tkinter.

Permite:

- Cargar imágenes
- Seleccionar descriptor (HOG o LBP)
- Seleccionar modelo
- Clasificar imágenes
- Mostrar probabilidades y confianza
- Visualizar el top 3 de predicciones

La interfaz tiene un estilo inspirado en terminal/científico con tema oscuro.

---

# Descriptores utilizados

## HOG

El descriptor HOG se utilizó para capturar bordes y formas presentes en las imágenes.

Configuración utilizada:

- Imagen de entrada: `128x128`
- `blockSize = 16x16`
- `blockStride = 8x8`
- `cellSize = 8x8`
- `nbins = 9`

Genera un vector de aproximadamente:

```text
8100 features
```

---

## LBP

El descriptor LBP se utilizó para capturar patrones de textura en las imágenes.

Configuración utilizada:

- Radius = 3
- Points = 24
- Método uniforme (`uniform`)
- Equalización de histograma previa

Genera un histograma de:

```text
26 features
```

---

# Modelos entrenados

Se entrenaron dos modelos MLP:

| Modelo | Descriptor |
|---|---|
| `mejor_modelo_hog.pth` | HOG |
| `mejor_modelo_lbp.pth` | LBP |

Ambos modelos usan:

- Capas fully connected
- Función de activación ReLU
- Dropout
- Softmax para clasificación final

---

# Cómo ejecutar el proyecto

## 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

O usando `uv`:

```bash
uv sync
```

---

## 2. Ejecutar la aplicación

```bash
python main.py
```

---

# Uso de la aplicación

1. Abrir la aplicación
2. Presionar **“CARGAR IMAGEN”**
3. Seleccionar una imagen
4. Elegir el descriptor:
   - HOG
   - LBP
5. Presionar **“CLASIFICAR”**
6. Ver la predicción y el porcentaje de confianza

---

# Resultados

El sistema logra clasificar imágenes de animales utilizando dos enfoques distintos de extracción de características.

Se comparó el desempeño entre:

- HOG + MLP
- LBP + MLP

Mostrando cómo diferentes descriptores pueden afectar la precisión del modelo dependiendo de las características visuales de cada imagen.

---

# Observaciones

- Los modelos utilizan `StandardScaler`, por eso se incluyen:
  - `scaler_hog.pkl`
  - `scaler_lbp.pkl`

- Las imágenes son redimensionadas automáticamente a `128x128`.

---

# Posibles mejoras

- Agregar más clases de animales
- Implementar CNNs
- Mejorar precisión del modelo
- Agregar métricas visuales
- Exportar resultados

---

# Conclusión

Este proyecto permitió aplicar conceptos de:

- procesamiento de imágenes,
- extracción de características,
- redes neuronales,
- e interfaces gráficas en Python.

Además, sirvió para comparar el comportamiento de distintos descriptores clásicos como HOG y LBP en tareas de clasificación de imágenes.