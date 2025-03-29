import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
imagen = cv2.imread("imagen_medica.jpg", cv2.IMREAD_GRAYSCALE)

if imagen is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta del archivo.")

# Obtener dimensiones de la imagen
alto, ancho = imagen.shape

# --- Ejercicio 1: Traslación ---
# Matriz de traslación entera (50,30)
M1 = np.float32([[1, 0, 50], [0, 1, 30]])
imagen_trasladada1 = cv2.warpAffine(imagen, M1, (ancho, alto))

# Matriz de traslación con decimales (20.5, 15.5)
M2 = np.float32([[1, 0, 20.5], [0, 1, 15.5]])
imagen_trasladada2 = cv2.warpAffine(imagen, M2, (ancho, alto))

# --- Ejercicio 2: Rotación ---
# Definir el centro de la imagen
centro = (ancho // 2, alto // 2)

# Matriz de rotación de 45°
M_rotacion = cv2.getRotationMatrix2D(centro, 45, 1)
imagen_rotada = cv2.warpAffine(imagen, M_rotacion, (ancho, alto))

# --- Ejercicio 3: Escalado ---
# Escalado al 150% y 50%
imagen_escalada_150 = cv2.resize(imagen, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
imagen_escalada_50 = cv2.resize(imagen, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# --- Mostrar resultados ---
plt.figure(figsize=(12, 8))

# Mostrar traslaciones
plt.subplot(2,3,1), plt.imshow(imagen, cmap='gray'), plt.title("Original")
plt.subplot(2,3,2), plt.imshow(imagen_trasladada1, cmap='gray'), plt.title("Traslación (50,30)")
plt.subplot(2,3,3), plt.imshow(imagen_trasladada2, cmap='gray'), plt.title("Traslación (20.5,15.5)")

# Mostrar rotación
plt.subplot(2,3,4), plt.imshow(imagen_rotada, cmap='gray'), plt.title("Rotación 45°")

# Mostrar escalado
plt.subplot(2,3,5), plt.imshow(imagen_escalada_150, cmap='gray'), plt.title("Escala 150%")
plt.subplot(2,3,6), plt.imshow(imagen_escalada_50, cmap='gray'), plt.title("Escala 50%")

plt.tight_layout()
plt.show()
