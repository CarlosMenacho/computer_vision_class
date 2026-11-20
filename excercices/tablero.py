import cv2
import numpy as np

from utils import cvlog

cvlog.start("tablero")

for frame in cvlog.frames(0):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flag = cv2.CALIB_CB_NORMALIZE_IMAGE

    columnas = 9
    filas = 6

    encontrado, esquinas = cv2.findChessboardCorners(
        gray,
        (filas, columnas), 
        flags=flag)

    # cvlog.scalar("detectado", [])
    # cvlog.points2d("camara/esquinas", [])

    if encontrado:
        print(f"Se encuentra chessboards")
        criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        esquinas = cv2.cornerSubPix(gray, esquinas, (11,11), (-1,-1), criteria=criterio)
        esquinas = esquinas.reshape(-1, 2)

        cvlog.scalar("detectado", esquinas is not None)
        cvlog.points2d("camara/esquinas", esquinas, colors=(0,255,0), radius=4)

        cvlog.lines2d("camara/orden", esquinas, colors=(255,255,0))  

    cvlog.image("image/frame",frame)