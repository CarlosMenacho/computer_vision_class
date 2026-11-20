import cv2
import numpy as np
import glob
import os
from pathlib import Path

carpeta = "/home/carlos/classes/computer_vision_class/excercices/camera_calib"
list_images = os.listdir(carpeta)

print(list_images)
for file in list_images:
    img = cv2.imread(str(Path(carpeta, file)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flag = cv2.CALIB_CB_NORMALIZE_IMAGE
    columnas = 9
    filas = 6

    encontrado, esquinas = cv2.findChessboardCorners(
        gray,
        (filas, columnas), 
        flags=flag)
    
    print(Path(carpeta + file), f"  tablero {encontrado}")

