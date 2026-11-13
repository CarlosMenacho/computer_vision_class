"""
Tutorial de cvlog: como visualizar imagenes y resultados con Rerun
==================================================================

cvlog es una capa simple sobre Rerun. En vez de llenar el notebook de
cv2.imshow / plt.imshow, se "registra" (log) cada resultado con un NOMBRE
y todo aparece ordenado en el visor de Rerun.

Ejecutar desde la carpeta excercices/:

    python test_cvlog.py          # abre el visor de Rerun
    python test_cvlog.py --save   # guarda tutorial_cvlog.rrd (abrir con: rerun tutorial_cvlog.rrd)

Lea el codigo paso a paso y compare con lo que aparece en el visor.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from utils import cvlog  # cvlog.py esta en excercices/utils/


# ---------------------------------------------------------------------------
# PASO 0. Iniciar Rerun (una sola vez, al comienzo)
# ---------------------------------------------------------------------------
# mode puede ser:
#   "viewer"   -> app de escritorio (por defecto)
#   "notebook" -> visor dentro de Jupyter
#   "web"      -> visor en el navegador
#   "save"     -> guarda un archivo .rrd para verlo despues
mode = "save" if "--save" in sys.argv else "viewer"
cvlog.start("tutorial_cvlog", mode=mode)

# Importante: Rerun solo muestra en la linea de tiempo "step" lo que se
# registro CON un step. Fijamos step 0 desde el inicio para que todos los
# ejemplos sean visibles al recorrer la barra de tiempo.
cvlog.step(0)


# ---------------------------------------------------------------------------
# PASO 1. Registrar imagenes
# ---------------------------------------------------------------------------
# cvlog.image(nombre, imagen)
#   * El nombre es la "ruta" en el visor. Use "/" para agrupar:
#     "1_imagenes/original" y "1_imagenes/gris" quedan en la misma carpeta.
#   * OpenCV carga en BGR; cvlog lo convierte a RGB automaticamente.
#     Si su imagen ya es RGB (matplotlib, skimage) use bgr=False.
# La ruta se arma desde la carpeta de ESTE archivo, asi funciona sin importar
# desde donde se ejecute el script (terminal, boton Run de VS Code, ...).
path = Path(__file__).resolve().parent / "gatito.jpeg"
img = cv2.imread(str(path))
if img is None:  # si no esta la imagen, se dibuja una de prueba
    print(f"[aviso] no se encontro {path}; uso una imagen sintetica")
    img = np.full((240, 320, 3), 40, np.uint8)
    cv2.circle(img, (200, 120), 60, (0, 0, 255), -1)
    cv2.rectangle(img, (30, 40), (120, 140), (0, 255, 0), -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

cvlog.image("1_imagenes/original", img)          # color BGR
cvlog.image("1_imagenes/gris", gray)             # un canal (H, W)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cvlog.image("1_imagenes/rgb", rgb, bgr=False)    # ya es RGB -> bgr=False

# Tipos de datos: cvlog los ajusta para que se vean bien
sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)        # float con negativos -> se normaliza a [0, 255]
cvlog.image("1_imagenes/sobel_x", sobel)
umbral = gray > 128                              # bool -> se muestra como 0 / 255
cvlog.image("1_imagenes/umbral", umbral)


# ---------------------------------------------------------------------------
# PASO 2. Histograma
# ---------------------------------------------------------------------------
cvlog.histogram("2_histograma/gris", gray)


# ---------------------------------------------------------------------------
# PASO 3. Dibujar SOBRE una imagen (mismo prefijo)
# ---------------------------------------------------------------------------
# Si el nombre empieza con el de una imagen ("3_anotaciones/img/..."),
# el dibujo se superpone a esa imagen. Coordenadas en pixeles (x, y).
# Colores en RGB: (255, 0, 0) = rojo.
cvlog.image("3_anotaciones/img", img)

# Cajas: formato "xywh" (como cv2.boundingRect) o "xyxy"
cvlog.boxes2d("3_anotaciones/img/caja", [[w // 4, h // 4, w // 2, h // 2]],
              colors=(255, 0, 0), labels=["region"])

# Puntos: arreglo (N, 2)
corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
if corners is not None:
    cvlog.points2d("3_anotaciones/img/esquinas", corners.reshape(-1, 2),
                   colors=(0, 255, 0), radius=3)

# Contornos: directo desde cv2.findContours
cnts, _ = cv2.findContours(cv2.Canny(gray, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cvlog.contours("3_anotaciones/img/contornos", cnts, colors=(255, 255, 0))

# Lineas: segmentos de cv2.HoughLinesP (N, 1, 4)
lines = cv2.HoughLinesP(cv2.Canny(gray, 100, 200), 1, np.pi / 180, 50,
                        minLineLength=30, maxLineGap=5)
if lines is not None:
    cvlog.lines2d("3_anotaciones/img/lineas", lines, colors=(255, 0, 255), width=2)


# ---------------------------------------------------------------------------
# PASO 4. Mascaras de segmentacion
# ---------------------------------------------------------------------------
# Enteros: 0 = fondo, 1, 2, ... = clases. Se superpone con un nombre hijo.
_, otsu = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cvlog.image("4_mascara/img", img)
cvlog.mask("4_mascara/img/otsu", otsu, opacity=0.5)


# ---------------------------------------------------------------------------
# PASO 5. Linea de tiempo: ver como cambia un resultado
# ---------------------------------------------------------------------------
# cvlog.step(i) fija el "paso" actual. Todo lo registrado despues queda en ese
# paso; en el visor se recorre con la barra de tiempo (abajo).
# cvlog.scalar() con step() dibuja una curva.
for i, k in enumerate(range(1, 22, 2)):
    cvlog.step(i)
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    cvlog.image("5_tiempo/blur", blur)
    cvlog.scalar("5_tiempo/nitidez", cv2.Laplacian(blur, cv2.CV_64F).var())
    cvlog.text("5_tiempo/log", f"kernel = {k}x{k}")


# ---------------------------------------------------------------------------
# PASO 6. Keypoints y correspondencias (features)
# ---------------------------------------------------------------------------
# Volvemos a step 0: lo que sigue no depende del tiempo y asi se ve desde el
# inicio de la barra (si no, quedaria en el ultimo step del bucle anterior).
cvlog.step(0)

orb = cv2.ORB_create(500)
M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 0.9)    # segunda vista: rotada y escalada
gray2 = cv2.warpAffine(gray, M, (w, h))
kp1, des1 = orb.detectAndCompute(gray, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

cvlog.image("6_features/img", gray)
cvlog.keypoints("6_features/img/orb", kp1, colors=(0, 255, 0), show_size=True)

if des1 is not None and des2 is not None:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cvlog.matches("6_features/matches", gray, kp1, gray2, kp2, bf.match(des1, des2),
                  max_matches=50)


# ---------------------------------------------------------------------------
# PASO 7. 3D: nube de puntos y camara
# ---------------------------------------------------------------------------
# La imagen en gris como "superficie": x, y = pixel, z = intensidad
small = cv2.resize(gray, (80, 60))
ys, xs = np.mgrid[0:small.shape[0], 0:small.shape[1]]
pts = np.stack([xs.ravel(), ys.ravel(), small.ravel() / 10.0], axis=1)
colores = cv2.cvtColor(cv2.resize(img, (80, 60)), cv2.COLOR_BGR2RGB).reshape(-1, 3)
cvlog.points3d("7_3d/superficie", pts, colors=colores, radius=0.4)

# Camara pinhole con su imagen proyectada (K = matriz intrinseca)
K = np.array([[300, 0, w / 2], [0, 300, h / 2], [0, 0, 1]])
cvlog.camera("7_3d/camara", K, width=w, height=h, t=[40, 30, -60], img=img)


# ---------------------------------------------------------------------------
# PASO 8. Video (ejemplo, no se ejecuta)
# ---------------------------------------------------------------------------
# for frame in cvlog.frames("video.mp4", every=2, max_frames=200):   # o frames(0) = webcam
#     bordes = cv2.Canny(frame, 100, 200)
#     cvlog.image("video/original", frame)
#     cvlog.image("video/bordes", bordes)
#
# frames() avanza la linea de tiempo solo: no hace falta llamar step().


# ---------------------------------------------------------------------------
# PASO 9. Organizar el visor (blueprint)
# ---------------------------------------------------------------------------
# Con muchas entidades, Rerun arma solo una cuadricula de vistas y algunas
# quedan muy pequenas o encimadas. Un "blueprint" dice que vista mostrar y
# donde. Aqui: una pestana por paso (arriba del visor).
#   origin = la entidad raiz de la vista; incluye a sus hijos
#            (p. ej. "6_features/img" muestra la imagen + "img/orb").
# Sin blueprint tambien puede: panel izquierdo -> clic en la entidad, o
# boton "+" -> agregar vista 2D con origin = la entidad.
rr = cvlog.rr  # Rerun directo, para cosas avanzadas
import rerun.blueprint as rrb  # noqa: E402

blueprint = rrb.Blueprint(
    rrb.Tabs(
        rrb.Grid(
            *[rrb.Spatial2DView(origin=f"1_imagenes/{n}", name=n)
              for n in ["original", "gris", "rgb", "sobel_x", "umbral"]],
            name="1 Imagenes",
        ),
        rrb.BarChartView(origin="2_histograma/gris", name="2 Histograma"),
        rrb.Spatial2DView(origin="3_anotaciones", name="3 Anotaciones"),
        rrb.Spatial2DView(origin="4_mascara", name="4 Mascara"),
        rrb.Horizontal(
            rrb.Spatial2DView(origin="5_tiempo/blur", name="blur"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="5_tiempo/nitidez", name="nitidez"),
                rrb.TextLogView(origin="5_tiempo/log", name="log"),
            ),
            name="5 Tiempo",
        ),
        rrb.Vertical(
            rrb.Spatial2DView(origin="6_features/img", name="Keypoints ORB"),
            rrb.Spatial2DView(origin="6_features/matches", name="Correspondencias"),
            name="6 Features",
        ),
        rrb.Spatial3DView(origin="7_3d", name="7 3D"),
    ),
    collapse_panels=False,
)
rr.send_blueprint(blueprint)


print(f"Listo: tutorial de cvlog ejecutado en modo '{mode}'")
