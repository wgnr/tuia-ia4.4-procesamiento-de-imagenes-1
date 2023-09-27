import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import sys

assert len(sys.argv)>1, "INGRESE EL ARCHIVO!"


def g(x):
    plt.figure()
    plt.imshow(x, cmap="gray")
    plt.colorbar()
    plt.show(block=False)

# Invetamos la imagen original
img = cv2.imread(sys.argv[-1], cv2.IMREAD_GRAYSCALE)

# Buscamos las ROI y las guardamos
roi = {}

# vemos que tiene una escala de grises las lineas asi que la convertimos en una image binaria
img_binary = (255 - img) > 90

# atravezamos las lineas
def traverse(x0, y, x_dir=True) -> tuple[int, int]:
    entered = False
    if x_dir:
        # buscamos hacia valores crecientes en la direccion x |
        iterator = zip(range(x0, img_binary.shape[0]), [y] * (img_binary.shape[0] - x0))
    else:
        # buscamos hacia volres decrecientes en la direccion de y <--
        iterator = zip([x0] * y, range(y, 0, -1))

    for x, y in iterator:
        if img_binary[x, y] == 1:
            entered = True
        if entered and img_binary[x, y] == 0:
            # estamos en la region interna
            break
    return x, y


x, _ = traverse(0, img_binary.shape[1] // 2)
for y in range(img_binary.shape[1] // 2, img_binary.shape[1]):
    if img_binary[x, y] == 1:
        break
upper_right = (x, y - 1)



def get_roi(x, y) -> tuple[int, int, int, int]:
    # tenemos que atravesar 10 lineas
    upper_left_x, _ = traverse(x, y)
    # vamos hasta el lado izquierdo
    for upper_left_y in range(y, 0, -1):
        if img_binary[upper_left_x, upper_left_y] == 1:
            upper_left_y += 1
            break
    # hasta abajo papi
    for lower_right_x in range(upper_left_x, img_binary.shape[0]):
        if img_binary[lower_right_x, upper_left_y] == 1:
            lower_right_x -= 1
            break
    return upper_left_x, lower_right_x, upper_left_y, y


# Nombre Y Apellido
n_y_a_x0, n_y_a_x1, n_y_a_y0, n_y_a_y1 = get_roi(*upper_right)
roi["nombre_y_apellido"] = n_y_a_x0, n_y_a_x1, n_y_a_y0, n_y_a_y1

# Edad
e_x0, e_x1, e_y0, e_y1 = get_roi(n_y_a_x1, n_y_a_y1)
roi["edad"] = e_x0, e_x1, e_y0, e_y1

# Mail
m_x0, m_x1, m_y0, m_y1 = get_roi(e_x1, e_y1)
roi["mail"] = m_x0, m_x1, m_y0, m_y1

# Legajo
l_x0, l_x1, l_y0, l_y1 = get_roi(m_x1, m_y1)
roi["legajo"] = l_x0, l_x1, l_y0, l_y1

# salteamos una linea, yendo para las preguntas
p_1_x, _ = traverse(l_x1, l_y1)
# veamos las coordenas de l si
_, p_yes_y = traverse(p_1_x, l_y1, x_dir=False)

# Preg 1
p_1_no_x0, p_1_no_x1, p_1_no_y0, p_1_no_y1 = get_roi(p_1_x, l_y1)
roi["p1_no"] = p_1_no_x0, p_1_no_x1, p_1_no_y0, p_1_no_y1

p_1_yes_x0, p_1_yes_x1, p_1_yes_y0, p_1_yes_y1 = get_roi(p_1_x, p_yes_y)
roi["p1_yes"] = p_1_yes_x0, p_1_yes_x1, p_1_yes_y0, p_1_yes_y1

# Preg 2
p_2_no_x0, p_2_no_x1, p_2_no_y0, p_2_no_y1 = get_roi(p_1_no_x1, p_1_no_y1)
roi["p2_no"] = p_2_no_x0, p_2_no_x1, p_2_no_y0, p_2_no_y1

p_2_yes_x0, p_2_yes_x1, p_2_yes_y0, p_2_yes_y1 = get_roi(p_1_no_x1, p_yes_y)
roi["p2_yes"] = p_2_yes_x0, p_2_yes_x1, p_2_yes_y0, p_2_yes_y1

# Preg 3
p_3_no_x0, p_3_no_x1, p_3_no_y0, p_3_no_y1 = get_roi(p_2_no_x1, p_2_no_y1)
roi["p3_no"] = p_3_no_x0, p_3_no_x1, p_3_no_y0, p_3_no_y1

p_2_yes_x0, p_2_yes_x1, p_2_yes_y0, p_2_yes_y1 = get_roi(p_2_no_x1, p_yes_y)
roi["p3_yes"] = p_2_yes_x0, p_2_yes_x1, p_2_yes_y0, p_2_yes_y1

# Coment arios
c_x0, c_x1, c_y0, c_y1 = get_roi(p_3_no_x1, p_3_no_y1)
roi["com"] = c_x0, c_x1, c_y0, c_y1


# Ahora tenemos que cumplir unos requisitos
EMPTY_THRESHOLD = 255-90
WORD_DIST = 7

def get_frontiers_index_ranges(of: np.ndarray) -> list[tuple[int, int]]:
    """Identificamos los indices donde un array cambia de valores"""
    # identificamos donde hubo cambios
    indexes = np.nonzero(np.diff(of))[0]
    # agrupamos de a pares
    ranges = [(indexes[i], indexes[i + 1]) for i in range(len(indexes)) if i % 2 == 0]
    return ranges


def group_by_threshold(of: list[tuple[int, int]], threshold: int) -> list[list[tuple[int, int]]]:
    """Agrupa rangos que esten a menos distancia que el umbral."""
    arr: list[list[tuple[int, int]]] = []
    aux = []
    for i in range(len(of)):
        aux.append(of[i])
        if i == len(of) - 1:
            arr.append(aux)
            continue

        if abs(of[i][1] - of[i + 1][0]) > threshold:
            # salto de linea
            arr.append(aux)
            aux = []
            continue
    return arr


def get_content(x0, x1, y0, y1, word_dist=WORD_DIST):
    """
    Devuelve cantidad de caracteres y palabaras
    """
    caracteres_ranges = get_frontiers_index_ranges(
        (img[x0:x1, y0:y1] < EMPTY_THRESHOLD).any(axis=0)
    )
    words = group_by_threshold(caracteres_ranges, word_dist)
    return len(caracteres_ranges), len(words)


ok,mal="OK","MAL"

caracteres, palabras = get_content(*roi["nombre_y_apellido"])
print("Nombre y apellido:",ok if palabras>=2 and palabras-1+caracteres<=25 else mal)

caracteres, palabras = get_content(*roi["edad"])
print("Edad:",ok if 1<caracteres<4 else mal)

caracteres, palabras = get_content(*roi["mail"])
print("Mail:",ok if palabras==1 and caracteres<=25 else mal)

caracteres, palabras = get_content(*roi["legajo"])
print("Legajo:",ok if palabras==1 and caracteres==8 else mal)


for p in range(1,4):
    cp_y=get_content(*roi[f"p{p}_yes"])[0]
    cp_n= get_content(*roi[f"p{p}_no"])[0]
    print(f"Pregunta {p}:",ok if cp_y<=1 and cp_n<=1 and cp_y!=cp_n else mal)

caracteres, palabras = get_content(*roi["com"])
print("Comentarios:",ok if caracteres<=25 else mal)
