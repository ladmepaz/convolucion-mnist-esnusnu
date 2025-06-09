# Visualización y Análisis de Convoluciones en MNIST

Este proyecto muestra cómo visualizar y analizar el funcionamiento interno de una red neuronal convolucional (CNN) entrenada sobre el dataset MNIST, utilizando TensorFlow y Keras. Además, permite probar el modelo con imágenes propias y observar cómo se procesan internamente.

---

## Requisitos previos

- Python 3.8 o superior
- Paquetes: `tensorflow`, `numpy`, `matplotlib`, `Pillow`, `requests`
- Conexión a internet para descargar el modelo preentrenado

Instala los paquetes necesarios con:

```
pip install tensorflow numpy matplotlib pillow requests
```

---

## Estructura del código

El script principal es `solucion.py` y está dividido en varias secciones:

### 1. **Configuración inicial y descarga del modelo**

- Descarga automáticamente el modelo preentrenado `mnist_model.h5` si no está presente.
- Carga el modelo y los datos de prueba de MNIST.
- Evalúa la precisión del modelo sobre el conjunto de prueba estándar.

**Resultado esperado:**  
Se imprime la precisión del modelo sobre MNIST (debería ser mayor al 98%).

---

### 2. **Visualización de los kernels (filtros) de la primera capa**

- Extrae y muestra los primeros 10 filtros de la primera capa convolucional del modelo.

**Resultado esperado:**  
Se muestra una figura con 10 imágenes, cada una representando un filtro aprendido por la red.

---

### 3. **Visualización de mapas de características (feature maps) con una imagen de MNIST**

- Selecciona una imagen de prueba de MNIST.
- Muestra la imagen original y los primeros 10 mapas de características generados por la primera capa convolucional.

**Resultado esperado:**  
Se visualiza cómo la red "ve" la imagen tras pasar por la primera capa.

---

### 4. **Predicción y análisis de imágenes propias**

- Carga imágenes propias (0.jpg a 9.jpg), las preprocesa y predice su etiqueta con el modelo.
- Muestra las imágenes junto con la etiqueta real y la predicha.

**Resultado esperado:**  
Se visualizan tus imágenes y se comparan las etiquetas reales con las predichas por el modelo.

---

### 5. **Visualización de mapas de características con una imagen propia**

- Selecciona una de tus imágenes y muestra sus primeros 10 mapas de características tras la primera convolución.

**Resultado esperado:**  
Permite comparar cómo procesa la red una imagen propia frente a una de MNIST.

---

## Ejecución

Coloca tus imágenes propias (0.jpg, 1.jpg, ..., 9.jpg) en la misma carpeta que el script. Luego ejecuta:

```
python solucion.py
```

Sigue las instrucciones y observa las visualizaciones generadas.

---

## Notas

- Si tus imágenes propias no se ven bien clasificadas, prueba a invertir los colores (modificando la línea `img_array = 1.0 - img_array`).
- Puedes cambiar el índice de la imagen propia a analizar en la sección 4 modificando la variable `IMG_INDEX_TO_ANALYZE`.

---

## Resultados

- **Precisión del modelo en MNIST:**  
  Esperada >98%.

- **Visualización de filtros:**  
  Permite entender qué patrones básicos aprende la red.

- **Mapas de características:**  
  Muestran cómo la red extrae información relevante de las imágenes.

- **Predicción en imágenes propias:**  
  Permite evaluar la generalización del modelo fuera del dataset original.

---
