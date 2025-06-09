# ===================================================================
# SECCIÓN 0: SETUP INICIAL (Importaciones, Descarga y Carga del Modelo)
# ===================================================================
print("--- [INICIO] SECCIÓN 0: Configuración inicial ---")

# Importaciones necesarias para todo el script
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# --- Descarga del modelo si no existe ---
MODEL_FILE = "mnist_model.h5"
if not os.path.exists(MODEL_FILE):
    print(f"El modelo '{MODEL_FILE}' no se encontró. Descargando...")
    url = "https://huggingface.co/spaces/ayaanzaveri/mnist/resolve/main/mnist-model.h5"
    try:
        r = requests.get(url)
        r.raise_for_status()  # Lanza un error si la descarga falla
        with open(MODEL_FILE, "wb") as f:
            f.write(r.content)
        print("Modelo descargado y guardado correctamente.")
    except Exception as e:
        print(f"Error al descargar el modelo: {e}")
        # Si la descarga falla, no podemos continuar.
        exit()
else:
    print(f"El modelo '{MODEL_FILE}' ya existe localmente.")

# --- Carga del modelo y datos de prueba de MNIST ---
try:
    # Cargar el modelo preentrenado
    model = load_model(MODEL_FILE)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("\nModelo cargado y compilado.")
    # model.summary() # Muestra la arquitectura del modelo

    # Cargar datos de prueba de MNIST (se usarán para comparaciones)
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    print("Datos de prueba de MNIST cargados y preprocesados.")

except Exception as e:
    print(f"Error al cargar el modelo o los datos: {e}")
    exit()

# --- Evaluación del modelo en el set de prueba estándar ---
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nPrecisión del modelo en el conjunto de prueba MNIST: {accuracy:.4f}")

print("--- [FIN] SECCIÓN 0: Configuración completada ---\n")


# ======================================================================
# REQUISITO 1: Visualizar los Kernels (Filtros) del Modelo
# ======================================================================
print("--- [INICIO] REQUISITO 1: Visualizando Kernels ---")

# Extraer los filtros (kernels) de la primera capa convolucional
first_conv_layer = model.layers[0]
weights, biases = first_conv_layer.get_weights()

print(f"La primera capa es de tipo: {type(first_conv_layer).__name__}")
print(f"Forma de los kernels (pesos): {weights.shape}")

# Visualizar los primeros 10 filtros
fig, axes = plt.subplots(1, 10, figsize=(20, 3))
fig.suptitle("Requisito 1: Primeros 10 Kernels (Filtros) de la 1ra Capa Convolucional", fontsize=16)
for i in range(10):
    if i < weights.shape[3]: # Asegurarse de no exceder el número de filtros
        kernel = weights[:, :, 0, i]
        axes[i].imshow(kernel, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Filtro {i}")
plt.show()

print("--- [FIN] REQUISITO 1 ---\n")


# ======================================================================
# REQUISITO 2: Mapas de Características (con imagen de MNIST)
# ======================================================================
print("--- [INICIO] REQUISITO 2: Mapas de Características con Imagen de MNIST ---")

# Crear un nuevo modelo que termine en la primera capa convolucional
model_truncated = Model(inputs=model.inputs, outputs=model.layers[0].output)

# Seleccionar una imagen del conjunto de prueba MNIST (ej: el primer '7')
img_mnist = x_test[0].reshape(1, 28, 28, 1)

# Obtener las salidas (feature maps)
feature_maps_mnist = model_truncated.predict(img_mnist)

# Mostrar la imagen original y las primeras 10 salidas
fig, axes = plt.subplots(1, 11, figsize=(22, 3))
fig.suptitle("Requisito 2: Imagen Original de MNIST y sus Primeros 10 Mapas de Características", fontsize=16)
axes[0].imshow(img_mnist.squeeze(), cmap="gray")
axes[0].set_title("Imagen Original")
axes[0].axis("off")

for i in range(10):
    if i < feature_maps_mnist.shape[3]:
        axes[i+1].imshow(feature_maps_mnist[0, :, :, i], cmap="gray")
        axes[i+1].axis("off")
        axes[i+1].set_title(f"Mapa {i}")
plt.show()

print("--- [FIN] REQUISITO 2 ---\n")


# ======================================================================
# REQUISITO 3: Carga, Preprocesamiento y Predicción de Imágenes Propias
# ======================================================================
print("--- [INICIO] REQUISITO 3: Procesando y Prediciendo con Imágenes Propias ---")

# --- 3.1: Carga y Preprocesamiento de Imágenes Propias ---

# SOLUCIÓN: Definimos aquí las etiquetas REALES de nuestras imágenes, en orden.
# El primer número corresponde a 0.jpg, el segundo a 1.jpg, y así sucesivamente.
my_actual_labels = [3, 5, 6, 2, 9, 4, 2, 8, 7, 1]

my_images_list = []
my_labels_list = []
print("Iniciando la carga de imágenes propias (0.jpg a 9.jpg)...")

for i in range(10):
    filename = f"{i}.jpg"
    if not os.path.exists(filename):
        print(f"¡Advertencia! No se encontró el archivo {filename}. Saltando este número.")
        # Si un archivo falta, también saltamos su etiqueta para mantener la sincronización
        if i < len(my_actual_labels):
            my_actual_labels.pop(i)
        continue

    img = Image.open(filename).convert('L') # Convertir a escala de grises
    img = img.resize((28, 28))             # Cambiar tamaño a 28x28
    img_array = np.array(img)
    img_array = img_array.astype("float32") / 255.0 # Normalizar

    # ¡PASO CRÍTICO! Invertir colores: MNIST usa blanco sobre negro.
    # Si dibujaste negro sobre blanco, esta línea es necesaria.
    # Basado en tu imagen, tus números ya son blancos sobre fondo negro, así que esta línea podría no ser necesaria.
    # Si las predicciones son muy malas, prueba a comentar la siguiente línea.
    img_array = 1.0 - img_array # Comenta o descomenta según cómo hayas creado tus imágenes.

    img_array = np.expand_dims(img_array, axis=-1) # Añadir dimensión de canal
    my_images_list.append(img_array)

    # Usamos la lista de etiquetas reales que definimos arriba
    if i < len(my_actual_labels):
        my_labels_list.append(my_actual_labels[i])

# Es importante que el número de imágenes cargadas coincida con el de etiquetas
my_labels_list = my_labels_list[:len(my_images_list)]

my_images_processed = np.array(my_images_list)
my_labels = np.array(my_labels_list)

print(f"\nProceso finalizado. Se cargaron {len(my_images_processed)} imágenes.")
print(f"Forma final del array de imágenes: {my_images_processed.shape}")
print(f"Etiquetas reales asignadas: {my_labels}")

# --- 3.2: Calcular y Mostrar las Predicciones ---
# (Esta parte del código no necesita cambios, ya que usa la variable `my_labels` que ahora es correcta)
print("\nRealizando predicciones en las imágenes propias...")
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("Requisito 3: Predicciones en Imágenes Propias (Corregido)", fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < len(my_images_processed):
        img_to_predict = my_images_processed[i]
        # El modelo espera un 'batch' de imágenes, por eso añadimos una dimensión al principio
        prediction_input = np.expand_dims(img_to_predict, axis=0)

        # Hacemos la predicción
        predictions = model.predict(prediction_input)
        pred = np.argmax(predictions, axis=1)[0]

        ax.imshow(img_to_predict.squeeze(), cmap="gray")
        ax.set_title(f"Real: {my_labels[i]} | Pred: {pred}")
        ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("--- [FIN] REQUISITO 3 ---\n")

# ======================================================================
# REQUISITO 4: Mapas de Características (con imagen propia)
# ======================================================================
print("--- [INICIO] REQUISITO 4: Mapas de Características con Imagen Propia ---")

# Seleccionar una de tus imágenes para analizar (ej. la que corresponde al número 7)
IMG_INDEX_TO_ANALYZE = 7
if IMG_INDEX_TO_ANALYZE < len(my_images_processed):
    my_single_image = my_images_processed[IMG_INDEX_TO_ANALYZE]
    # Añadir la dimensión de 'batch'
    img_for_feature_maps = np.expand_dims(my_single_image, axis=0)

    # Obtener las salidas (feature maps) usando el mismo modelo truncado de antes
    feature_maps_own = model_truncated.predict(img_for_feature_maps)

    # Mostrar la imagen propia y las primeras 10 salidas
    fig, axes = plt.subplots(1, 11, figsize=(22, 3))
    fig.suptitle(f"Requisito 4: Imagen Propia '{my_labels[IMG_INDEX_TO_ANALYZE]}.jpg' y sus Mapas de Características", fontsize=16)
    axes[0].imshow(img_for_feature_maps.squeeze(), cmap="gray")
    axes[0].set_title("Imagen Propia")
    axes[0].axis("off")

    for i in range(10):
        if i < feature_maps_own.shape[3]:
            axes[i+1].imshow(feature_maps_own[0, :, :, i], cmap="gray")
            axes[i+1].axis("off")
            axes[i+1].set_title(f"Mapa {i}")
    plt.show()
else:
    print(f"No se pudo realizar el requisito 4 porque la imagen con índice {IMG_INDEX_TO_ANALYZE} no se cargó.")

print("--- [FIN] REQUISITO 4 ---")
print("\n¡Todos los pasos han sido ejecutados!")