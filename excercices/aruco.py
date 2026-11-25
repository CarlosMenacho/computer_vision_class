"""
Deteccion de marcadores ArUco (diccionario DICT_4X4_50) con la webcam.

Que se ve en Rerun:
  camara            -> imagen de la webcam
  camara/marcadores -> contorno de cada marcador con su ID
  camara/esquina0   -> primera esquina de cada marcador (indica la orientacion)
"""

import cv2
import numpy as np

from utils import cvlog

diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(diccionario, cv2.aruco.DetectorParameters())

cvlog.start("aruco")

for frame in cvlog.frames(0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # esquinas: lista de arrays (1, 4, 2); ids: array (N, 1) o None
    esquinas, ids, _ = detector.detectMarkers(gray)

    cvlog.image("camara", frame)

    if ids is None:
        cvlog.lines2d("camara/marcadores", [])  # borra los dibujos viejos
        cvlog.points2d("camara/esquina0", np.zeros((0, 2)))
        continue

    # Cerrar cada contorno repitiendo la primera esquina
    contornos = [np.vstack([c[0], c[0][:1]]) for c in esquinas]
    primeras = np.array([c[0][0] for c in esquinas])

    cvlog.lines2d("camara/marcadores", contornos, colors=(0, 255, 0), width=3)
    cvlog.points2d("camara/esquina0", primeras, colors=(255, 0, 0), radius=6,
                   labels=[f"ID {i}" for i in ids.ravel()])
    print("IDs detectados:", ids.ravel().tolist())


# Para imprimir un marcador de prueba (ID 7, 200x200 px):
#   img = cv2.aruco.generateImageMarker(diccionario, 7, 200)
#   cv2.imwrite("aruco_7.png", img)
