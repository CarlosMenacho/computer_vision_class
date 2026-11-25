import cv2
import numpy as np
from pathlib import Path
from utils import cvlog

cvlog.start("calibracion")

carpeta = Path("/home/carlos/classes/computer_vision_class/excercices/camera_calib")
salida = "calibracion.npz"

columnas = 8   # esquinas interiores
filas = 5      # esquinas interiores

lado_cuadrado = 0.025  # en metros

objp = np.zeros((columnas * filas, 3), np.float32)
objp[:, :2] = np.mgrid[0:columnas, 0:filas].T.reshape(-1, 2) * lado_cuadrado

CRITERIO = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flag = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

puntos_3d = []   # uno por foto valida
puntos_2d = []
nombres = []
tamano = None

extensiones = {".jpg", ".jpeg", ".png", ".bmp"}
fotos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in extensiones)

for ruta in fotos:
    img = cv2.imread(str(ruta))
    if img is None:
        print(f"{ruta.name}: no se pudo leer")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tamano = gray.shape[::-1]  # (ancho, alto)

    encontrado, esquinas = cv2.findChessboardCorners(gray, (columnas, filas), flags=flag)
    if encontrado:
        esquinas = cv2.cornerSubPix(gray, esquinas, (11, 11), (-1, -1), CRITERIO)
    else:
        # Respaldo: detector por sectores, más robusto a oclusiones y desenfoque
        encontrado, esquinas = cv2.findChessboardCornersSB(gray, (columnas, filas))
    print(f"{ruta.name}: tablero {encontrado}")

    if not encontrado:
        continue

    puntos_3d.append(objp)
    puntos_2d.append(esquinas)
    nombres.append(ruta.stem)

    cvlog.image(f"fotos/{ruta.stem}", img)
    cvlog.points2d(f"fotos/{ruta.stem}/esquinas", esquinas.reshape(-1, 2),
                   colors=(0, 255, 0), radius=6)

if len(puntos_2d) < 3:
    raise SystemExit(f"Solo {len(puntos_2d)} fotos con tablero detectado; se necesitan al menos 3.")

# Calibrar
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(puntos_3d, puntos_2d, tamano, None, None)

np.set_printoptions(precision=3, suppress=True)
print(f"\nFotos usadas: {len(puntos_2d)}")
print(f"Error RMS de reproyeccion: {rms:.3f} px")
print("Matriz K:\n", K)
print("Distorsion (k1, k2, p1, p2, k3):", dist.ravel())

# Verificar la calibración
for nombre, p3, p2, rvec, tvec in zip(nombres, puntos_3d, puntos_2d, rvecs, tvecs):
    proy, _ = cv2.projectPoints(p3, rvec, tvec, K, None)          # sin distorsión
    p2_und = cv2.undistortPoints(p2.reshape(-1, 1, 2).astype(np.float64),
                                 K, dist, P=K)                     # a píxeles con K
    error = np.linalg.norm(proy.reshape(-1, 2) - p2_und.reshape(-1, 2), axis=1).mean()
    print(f"  {nombre}: {error:.3f} px   distancia {np.linalg.norm(tvec):.2f} m")
    cvlog.points2d(f"fotos/{nombre}/reproy", proy.reshape(-1, 2),
                   colors=(255, 0, 0), radius=3)

# Comparación original vs. sin distorsión
ruta0 = next(p for p in fotos if p.stem == nombres[0])
img = cv2.imread(str(ruta0))
K_nueva, _ = cv2.getOptimalNewCameraMatrix(K, dist, tamano, alpha=0)
cvlog.image("corregida/original", img)
cvlog.image("corregida/sin_distorsion", cv2.undistort(img, K, dist, None, K_nueva))

np.savez(salida, K=K, dist=dist, rms=rms, tamano=np.array(tamano))
print(f"\nGuardado en {salida}")
