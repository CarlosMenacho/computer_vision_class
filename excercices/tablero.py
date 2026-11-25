import cv2
import numpy as np

from utils import cvlog

cvlog.start("tablero")

columnas = 8   # esquinas interiores
filas = 5      # esquinas interiores

flag = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for frame in cvlog.frames(0):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    encontrado, esquinas = cv2.findChessboardCorners(gray, (columnas, filas), flags=flag)

    cvlog.image("camara", frame)
    cvlog.scalar("detectado", int(encontrado))

    if not encontrado:
        cvlog.points2d("camara/esquinas", np.zeros((0, 2)))  # borra los puntos viejos
        cvlog.lines2d("camara/orden", [])
        continue

    esquinas = cv2.cornerSubPix(gray, esquinas, (11, 11), (-1, -1), criteria=criterio)
    esquinas = esquinas.reshape(-1, 2)

    cvlog.points2d("camara/esquinas", esquinas, colors=(0, 255, 0), radius=4)
    cvlog.lines2d("camara/orden", [esquinas], colors=(255, 255, 0))
