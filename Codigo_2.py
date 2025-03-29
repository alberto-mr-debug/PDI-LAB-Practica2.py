import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
imagen = cv2.imread("/content/resonancia.jpg", cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se cargó correctamente
if imagen is None:
    raise ValueError("No se pudo cargar la imagen. Verifica el nombre y la ruta del archivo.")

# Obtener dimensiones de la imagen
alto, ancho = imagen.shape

# --- Ejercicio 1: Ecualización de histograma ---
# Obtener histograma de la imagen original
histograma_original = cv2.calcHist([imagen], [0], None, [256], [0, 256])

# Aplicar ecualización de histograma
imagen_ecualizada = cv2.equalizeHist(imagen)

# Obtener histograma de la imagen ecualizada
histograma_ecualizado = cv2.calcHist([imagen_ecualizada], [0], None, [256], [0, 256])

# --- Mostrar resultados ---
plt.figure(figsize=(12, 8))

# Mostrar imágenes
plt.subplot(2, 2, 1), plt.imshow(imagen, cmap='gray'), plt.title("Original")
plt.axis("off")
plt.subplot(2, 2, 2), plt.imshow(imagen_ecualizada, cmap='gray'), plt.title("Ecualizada")
plt.axis("off")

# Mostrar histogramas
plt.subplot(2, 2, 3), plt.plot(histograma_original, color='black'), plt.title("Histograma Original")
plt.subplot(2, 2, 4), plt.plot(histograma_ecualizado, color='black'), plt.title("Histograma Ecualizado")

plt.tight_layout()
plt.show()
