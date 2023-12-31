import cv2
import numpy as np
import matplotlib.pyplot as plt


def local_histogram_equalization(image, window_size):
    # Obtener dimensiones de la imagen
    height, width = image.shape

    # Calcula el desplazamiento de la mitad del tamaño de la ventana
    half_window_size = window_size // 2

    # Agregar bordes a la imagen
    border_type = cv2.BORDER_REPLICATE
    image_with_border = cv2.copyMakeBorder(
        image,
        half_window_size,
        half_window_size,
        half_window_size,
        half_window_size,
        border_type,
    )

    # Crear una copia de la imagen para almacenar el resultado
    result_image = np.copy(image)

    for y in range(half_window_size, height + half_window_size):
        for x in range(half_window_size, width + half_window_size):
            # Definir las coordenadas de la ventana
            y_start = y - half_window_size
            y_end = y + half_window_size + 1
            x_start = x - half_window_size
            x_end = x + half_window_size + 1

            # Extraer la ventana de la imagen con bordes
            window = image_with_border[y_start:y_end, x_start:x_end]

            # Aplicar ecualización de histograma local usando cv2.equalizeHist
            window_equalized = cv2.equalizeHist(window)

            # Asignar la ventana ecualizada al resultado
            result_image[y - half_window_size, x - half_window_size] = window_equalized[
                half_window_size, half_window_size
            ]

    return result_image


# Cargar la imagen
image = cv2.imread("img/Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)

# Tamaño de la ventana de procesamiento (ajustar según sea necesario)
window_size = 21

# Aplicar la ecualización local del histograma con umbral
equalized_image = local_histogram_equalization(image, window_size)

# Eliminar ruido
result_image = cv2.medianBlur(equalized_image, 3)

# Mostrar la imagen original y la imagen procesada
fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)
fig.suptitle("Ecualización local de histograma")
axs[0].imshow(image, cmap="gray")
axs[0].set_title("Imagen Original")
axs[1].imshow(equalized_image, cmap="gray")
axs[1].set_title("Imagen Ecualizada")
axs[2].imshow(result_image, cmap="gray")
axs[2].set_title("Imagen Suavizada")
fig.tight_layout()
fig.show()
