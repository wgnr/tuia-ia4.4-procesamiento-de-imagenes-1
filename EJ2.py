import cv2
import numpy as np
import sys

THRESHOLD = 90
EMPTY_THRESHOLD = 255 - THRESHOLD
WORD_DIST = 7


def validate_form(img_path: str):
    # Cargamos la imagen
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Invertimos la imagen y luego la convertimos a una imagen binaria
    img_binary = (255 - img) > THRESHOLD

    def traverse(x0: int, y: int, x_dir=True) -> tuple[int, int]:
        """
        Funcion para atravesar continuos. Devuelve el pixel inmediato posterior luego de
        atravesar un continuo.
        """
        entered = False
        if x_dir:
            # buscamos en direccion x creciente
            iterator = zip(
                range(x0, img_binary.shape[0]), [y] * (img_binary.shape[0] - x0)
            )
        else:
            # buscamos en direccion y decreciente
            iterator = zip([x0] * y, range(y, 0, -1))

        for x, y in iterator:
            if img_binary[x, y] == 1:
                entered = True
            if entered and img_binary[x, y] == 0:
                # estamos en la region interna
                break
        return x, y

    # buscamos el primer punto dentro del formulario
    x, _ = traverse(0, img_binary.shape[1] // 2)

    # vamos hasta el extremo derecho interno de la imagen
    for y in range(img_binary.shape[1] // 2, img_binary.shape[1]):
        if img_binary[x, y] == 1:
            break
    upper_right = (x, y - 1)

    def get_roi(x: int, upper_right_y: int) -> tuple[int, int, int, int]:
        """
        Devuelve los valores de x e y en las cuales se encuentra contenido el ROI a paritr
        de LA PROXIMA REGION! NO DE LA CUAL ESTA CONTENIDA EN LOS PARAMETROS x e upper_right_y
        """
        # Busca proxima region
        upper_left_x, _ = traverse(x, upper_right_y)
        # busca extremo izquierdo
        for upper_left_y in range(upper_right_y, 0, -1):
            if img_binary[upper_left_x, upper_left_y] == 1:
                upper_left_y += 1
                break
        # busca extremo derecho
        for lower_right_x in range(upper_left_x, img_binary.shape[0]):
            if img_binary[lower_right_x, upper_left_y] == 1:
                lower_right_x -= 1
                break
        return upper_left_x, lower_right_x, upper_left_y, upper_right_y

    roi = {}

    roi["nombre_y_apellido"] = get_roi(*upper_right)
    roi["edad"] = get_roi(roi["nombre_y_apellido"][1], roi["nombre_y_apellido"][-1])
    roi["mail"] = get_roi(roi["edad"][1], roi["edad"][-1])
    roi["legajo"] = get_roi(roi["mail"][1], roi["mail"][-1])

    # Saltamos la fila de las Si y No
    p_1_x, _ = traverse(roi["legajo"][1], roi["legajo"][-1])
    # Buscamos conocer las coordenadas de las respuestas Si.
    _, p_yes_y = traverse(p_1_x, roi["legajo"][-1], x_dir=False)

    # Preg 1
    roi["p1_no"] = get_roi(p_1_x, roi["legajo"][-1])
    roi["p1_yes"] = get_roi(p_1_x, p_yes_y)
    # Preg 2
    roi["p2_no"] = get_roi(roi["p1_no"][1], roi["p1_no"][-1])
    roi["p2_yes"] = get_roi(roi["p1_no"][1], p_yes_y)
    # Preg 3
    roi["p3_no"] = get_roi(roi["p2_no"][1], roi["p2_no"][-1])
    roi["p3_yes"] = get_roi(roi["p2_no"][1], p_yes_y)

    roi["com"] = get_roi(roi["p3_no"][1], roi["p3_no"][-1])

    def get_frontiers_index_ranges(of: np.ndarray) -> list[tuple[int, int]]:
        """Identificamos los indices donde un array cambia de valores"""
        # identificamos donde hubo cambios
        indexes = np.nonzero(np.diff(of))[0]
        # agrupamos de a pares
        ranges = [
            (indexes[i], indexes[i + 1]) for i in range(len(indexes)) if i % 2 == 0
        ]
        return ranges

    def group_by_threshold(
        of: list[tuple[int, int]], threshold: int
    ) -> list[list[tuple[int, int]]]:
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

    ok, mal = "OK", "MAL"

    caracteres, palabras = get_content(*roi["nombre_y_apellido"])
    print(
        "{:<20}{}".format(
            "Nombre y apellido:",
            ok if palabras >= 2 and palabras - 1 + caracteres <= 25 else mal,
        )
    )

    caracteres, palabras = get_content(*roi["edad"])
    print("{:<20}{}".format("Edad:", ok if 1 < caracteres < 4 else mal))

    caracteres, palabras = get_content(*roi["mail"])
    print("{:<20}{}".format("Mail:", ok if palabras == 1 and caracteres <= 25 else mal))

    caracteres, palabras = get_content(*roi["legajo"])
    print(
        "{:<20}{}".format("Legajo:", ok if palabras == 1 and caracteres == 8 else mal)
    )

    for p in range(1, 4):
        cp_y = get_content(*roi[f"p{p}_yes"])[0]
        cp_n = get_content(*roi[f"p{p}_no"])[0]
        print(
            "{:<20}{}".format(
                f"Pregunta {p}:",
                ok if cp_y <= 1 and cp_n <= 1 and cp_y != cp_n else mal,
            )
        )

    caracteres, palabras = get_content(*roi["com"])
    print("{:<20}{}".format("Comentarios:", ok if caracteres <= 25 else mal))


if __name__ == "__main__":
    assert len(sys.argv) > 1, "INGRESE EL ARCHIVO!"
    validate_form(sys.argv[-1])
