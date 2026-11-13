"""
cvlog - registro visual simple con Rerun para la clase de Vision por Computador.

Uso minimo:

    import cv2
    import cvlog

    cvlog.start("lab_1")                      # abre el visor de Rerun
    img = cv2.imread("gatito.jpeg")           # OpenCV carga en BGR
    cvlog.image("original", img)              # se convierte a RGB automaticamente
    cvlog.image("gris", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

Video (procesar cuadro por cuadro):

    for frame in cvlog.frames("video.mp4"):   # avanza la linea de tiempo solo
        bordes = cv2.Canny(frame, 100, 200)
        cvlog.image("video/original", frame)
        cvlog.image("video/bordes", bordes)

Reglas:
  * El primer argumento siempre es el NOMBRE de la entidad. Use "/" para agrupar:
    "filtros/gauss", "filtros/sobel" quedan juntos en el visor.
  * Para dibujar sobre una imagen, use el mismo prefijo:
    cvlog.image("cam", img); cvlog.boxes2d("cam/caras", cajas)
  * Los colores son RGB, p. ej. (255, 0, 0) = rojo.
  * Si necesita algo avanzado use Rerun directo: `from cvlog import rr`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr

__all__ = [
    "rr", "start", "save", "step", "time",
    "image", "image_file", "depth", "mask", "histogram",
    "video", "frames",
    "points2d", "boxes2d", "lines2d", "contours", "keypoints", "matches",
    "points3d", "lines3d", "boxes3d", "arrows3d", "mesh3d", "camera",
    "scalar", "text",
]

_started = False


# ---------------------------------------------------------------------------
# Inicio / tiempo
# ---------------------------------------------------------------------------

def start(name: str = "cv_class", mode: str = "viewer", path: str | None = None) -> None:
    """Inicia Rerun. Llamar UNA vez al comienzo del script o notebook.

    mode:
      "viewer"   -> abre la app de escritorio de Rerun (por defecto).
      "notebook" -> muestra el visor dentro de Jupyter (requiere `rerun-notebook`).
      "web"      -> abre el visor en el navegador.
      "save"     -> guarda todo en un archivo .rrd (abrir luego con `rerun archivo.rrd`).
    """
    global _started
    rr.init(name)
    if mode == "viewer":
        rr.spawn()
    elif mode == "notebook":
        try:
            import rerun_notebook  # noqa: F401
        except ImportError:
            print("[cvlog] rerun-notebook no esta instalado; uso el visor de escritorio.\n"
                  "        Para verlo en Jupyter: uv pip install 'rerun-sdk[notebook]==0.36.0'")
            rr.spawn()
        else:
            rr.notebook_show(height=600)
    elif mode == "web":
        rr.serve_web_viewer(connect_to=rr.serve_grpc())
    elif mode == "save":
        rr.save(path or f"{name}.rrd")
        print(f"[cvlog] guardando en {path or name + '.rrd'}")
    else:
        raise ValueError('mode debe ser "viewer", "notebook", "web" o "save"')
    _started = True


def save(path: str = "recording.rrd") -> None:
    """Ademas de mostrar, guarda lo que se registre desde ahora en un archivo .rrd."""
    _ensure_started()
    rr.save(path)


def step(i: int, timeline: str = "step") -> None:
    """Fija el paso actual (iteracion, numero de cuadro, ...). Todo lo que se
    registre despues queda en ese paso y se puede recorrer con la barra de tiempo."""
    _ensure_started()
    rr.set_time(timeline, sequence=int(i))


def time(seconds: float, timeline: str = "time") -> None:
    """Fija el tiempo actual en segundos (util para video)."""
    _ensure_started()
    rr.set_time(timeline, duration=float(seconds))


# ---------------------------------------------------------------------------
# Imagenes
# ---------------------------------------------------------------------------

def image(name: str, img: np.ndarray, bgr: bool = True) -> None:
    """Registra una imagen de NumPy/OpenCV.

    * Escala de grises (H, W), color (H, W, 3) o con alfa (H, W, 4).
    * bgr=True (por defecto) porque cv2.imread / VideoCapture entregan BGR.
      Use bgr=False si la imagen ya es RGB (p. ej. de matplotlib o skimage).
    * Imagenes float se muestran tal cual si estan en [0, 1]; si no, se
      normalizan a [0, 255] (util para Sobel, Laplaciano, etc.).
    * bool se muestra como 0/255.
    """
    _ensure_started()
    img = _to_displayable(np.asarray(img))
    color_model = None
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim == 3:
        ch = img.shape[2]
        if ch == 3:
            color_model = "BGR" if bgr else "RGB"
        elif ch == 4:
            color_model = "BGRA" if bgr else "RGBA"
        else:
            raise ValueError(f"image(): {ch} canales no soportados (use 1, 3 o 4)")
    elif img.ndim != 2:
        raise ValueError(f"image(): forma {img.shape} no es una imagen")
    rr.log(name, rr.Image(img, color_model=color_model))


def image_file(name: str, path: str | Path) -> None:
    """Registra un archivo de imagen (jpg/png) sin decodificarlo."""
    _ensure_started()
    rr.log(name, rr.EncodedImage(path=path))


def depth(name: str, depth_map: np.ndarray, meter: float = 1.0) -> None:
    """Registra un mapa de profundidad. `meter` = cuantas unidades equivalen a 1 m
    (p. ej. 1000 si esta en milimetros)."""
    _ensure_started()
    rr.log(name, rr.DepthImage(np.asarray(depth_map), meter=meter))


def mask(name: str, labels: np.ndarray, opacity: float = 0.5) -> None:
    """Registra una mascara / segmentacion (enteros: 0 = fondo, 1, 2, ... = clases).
    Para superponerla a una imagen use un nombre hijo: "img/mascara"."""
    _ensure_started()
    labels = np.asarray(labels)
    if labels.dtype == bool:
        labels = labels.astype(np.uint8)
    rr.log(name, rr.SegmentationImage(labels, opacity=opacity))


def histogram(name: str, img: np.ndarray, bins: int = 256) -> None:
    """Registra el histograma de intensidades como grafico de barras."""
    _ensure_started()
    hist, _ = np.histogram(np.asarray(img).ravel(), bins=bins, range=(0, 256))
    rr.log(name, rr.BarChart(hist))


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def video(name: str, path: str | Path) -> None:
    """Registra un archivo de video completo (MP4 con H.264/H.265/AV1) para
    reproducirlo en el visor. Para PROCESAR cuadros use `frames()`."""
    _ensure_started()
    asset = rr.AssetVideo(path=path)
    try:
        ts = asset.read_frame_timestamps_nanos()
    except RuntimeError as e:
        raise RuntimeError(
            f"video(): {e}\n  Convierta el video a H.264:  "
            f"ffmpeg -i {path} -c:v libx264 -pix_fmt yuv420p salida.mp4\n"
            "  o use `for frame in cvlog.frames(...)` que acepta cualquier formato."
        ) from None
    rr.log(name, asset, static=True)
    rr.send_columns(
        name,
        indexes=[rr.TimeColumn("time", duration=ts * 1e-9)],
        columns=rr.VideoFrameReference.columns_nanos(ts),
    )


def frames(source=0, every: int = 1, max_frames: int | None = None, resize: float = 1.0):
    """Recorre un video (ruta) o camara (0, 1, ...) cuadro a cuadro.

    Entrega cada cuadro en BGR y fija automaticamente "step" y "time", asi
    todo lo que registre dentro del for queda sincronizado.

        for frame in cvlog.frames("calle.mp4", every=2, max_frames=300):
            cvlog.image("calle", frame)
    """
    import cv2

    _ensure_started()
    cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"frames(): no se pudo abrir {source!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                if resize != 1.0:
                    frame = cv2.resize(frame, None, fx=resize, fy=resize)
                rr.set_time("step", sequence=i)
                rr.set_time("time", duration=i / fps)
                yield frame
                n += 1
                if max_frames is not None and n >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# 2D (coordenadas en pixeles: x hacia la derecha, y hacia abajo)
# ---------------------------------------------------------------------------

def points2d(name: str, pts, colors=None, radius: float = 2.0, labels=None) -> None:
    """Puntos 2D. pts: (N, 2) con columnas (x, y)."""
    _ensure_started()
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    rr.log(name, rr.Points2D(pts, colors=colors, radii=radius, labels=labels))


def boxes2d(name: str, boxes, fmt: str = "xywh", colors=None, labels=None) -> None:
    """Cajas 2D. boxes: (N, 4).
    fmt="xywh" (como cv2.boundingRect / detectMultiScale) o "xyxy" (x1, y1, x2, y2)."""
    _ensure_started()
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if fmt == "xyxy":
        mins, sizes = b[:, :2], b[:, 2:] - b[:, :2]
    elif fmt == "xywh":
        mins, sizes = b[:, :2], b[:, 2:]
    else:
        raise ValueError('fmt debe ser "xywh" o "xyxy"')
    rr.log(name, rr.Boxes2D(mins=mins, sizes=sizes, colors=colors, labels=labels))


def lines2d(name: str, strips, colors=None, width: float = 1.0) -> None:
    """Polilineas 2D. strips: lista de arreglos (M, 2), o un solo arreglo (M, 2).
    Para segmentos sueltos (p. ej. HoughLinesP) pase (N, 2, 2) o (N, 4)."""
    _ensure_started()
    rr.log(name, rr.LineStrips2D(_as_strips(strips, 2), colors=colors, radii=width / 2))


def contours(name: str, cnts, colors=None, width: float = 1.0) -> None:
    """Contornos de cv2.findContours (lista de arreglos (M, 1, 2)), cerrados."""
    strips = []
    for c in cnts:
        c = np.asarray(c, dtype=np.float32).reshape(-1, 2)
        if len(c):
            strips.append(np.vstack([c, c[:1]]))
    lines2d(name, strips, colors=colors, width=width)


def keypoints(name: str, kps, colors=None, show_size: bool = False) -> None:
    """Keypoints de OpenCV (SIFT, ORB, FAST, ...). Lista de cv2.KeyPoint."""
    _ensure_started()
    if len(kps) == 0:
        rr.log(name, rr.Points2D(np.zeros((0, 2), np.float32)))
        return
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    radii = np.array([k.size / 2 for k in kps], dtype=np.float32) if show_size else 2.0
    rr.log(name, rr.Points2D(pts, radii=radii, colors=colors))


def matches(name: str, img1: np.ndarray, kps1, img2: np.ndarray, kps2, dmatches,
            bgr: bool = True, max_matches: int | None = 100) -> None:
    """Muestra dos imagenes lado a lado con lineas entre correspondencias
    (resultado de cv2.BFMatcher().match / FlannBasedMatcher)."""
    _ensure_started()
    a, b = _to_displayable(np.asarray(img1)), _to_displayable(np.asarray(img2))
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if b.ndim == 2:
        b = np.stack([b] * 3, -1)
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), dtype=np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a[:, :, :3]
    canvas[: b.shape[0], a.shape[1]:] = b[:, :, :3]
    image(name, canvas, bgr=bgr)
    ms = sorted(dmatches, key=lambda m: m.distance)[:max_matches]
    off = np.array([a.shape[1], 0], np.float32)
    segs = [np.array([kps1[m.queryIdx].pt, np.array(kps2[m.trainIdx].pt) + off], np.float32) for m in ms]
    rr.log(f"{name}/matches", rr.LineStrips2D(segs, colors=(0, 255, 0), radii=0.5))


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

def points3d(name: str, pts, colors=None, radius: float = 0.01, labels=None) -> None:
    """Nube de puntos 3D. pts: (N, 3). colors: (N, 3) uint8 RGB o un solo color."""
    _ensure_started()
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    rr.log(name, rr.Points3D(pts, colors=colors, radii=radius, labels=labels))


def lines3d(name: str, strips, colors=None, width: float = 0.005) -> None:
    """Polilineas 3D. strips: lista de (M, 3) o un arreglo (M, 3)."""
    _ensure_started()
    rr.log(name, rr.LineStrips3D(_as_strips(strips, 3), colors=colors, radii=width / 2))


def boxes3d(name: str, centers, sizes, colors=None, labels=None) -> None:
    """Cajas 3D alineadas a los ejes. centers y sizes: (N, 3)."""
    _ensure_started()
    rr.log(name, rr.Boxes3D(centers=centers, sizes=sizes, colors=colors, labels=labels))


def arrows3d(name: str, vectors, origins=None, colors=None) -> None:
    """Flechas 3D (normales, ejes, flujo, ...). vectors y origins: (N, 3)."""
    _ensure_started()
    rr.log(name, rr.Arrows3D(vectors=vectors, origins=origins, colors=colors))


def mesh3d(name: str, vertices, triangles=None, colors=None) -> None:
    """Malla de triangulos. vertices: (N, 3); triangles: (M, 3) indices;
    colors: (N, 3) por vertice. Para archivos .obj/.glb/.stl use
    rr.log(name, rr.Asset3D(path=...))."""
    _ensure_started()
    rr.log(name, rr.Mesh3D(vertex_positions=vertices, triangle_indices=triangles,
                           vertex_colors=colors))


def camera(name: str, K, width: int, height: int, R=None, t=None,
           img: np.ndarray | None = None, bgr: bool = True) -> None:
    """Dibuja una camara en 3D (modelo pinhole).

    K: matriz intrinseca 3x3. R, t: pose de la camara en el mundo
    (rotacion 3x3 y traslacion 3, camara -> mundo). Si se pasa `img`
    se proyecta en el plano de imagen de la camara.
    Ojo: cv2.solvePnP / calibrateCamera dan mundo -> camara; invierta con
    R_cw = R.T, t_cw = -R.T @ t.
    """
    _ensure_started()
    if R is not None or t is not None:
        rr.log(name, rr.Transform3D(
            translation=np.zeros(3) if t is None else np.asarray(t, np.float32).ravel(),
            mat3x3=np.eye(3) if R is None else np.asarray(R, np.float32),
        ))
    rr.log(name, rr.Pinhole(image_from_camera=np.asarray(K, np.float32),
                            width=width, height=height,
                            camera_xyz=rr.ViewCoordinates.RDF))
    if img is not None:
        image(f"{name}/image", img, bgr=bgr)


# ---------------------------------------------------------------------------
# Graficos y texto
# ---------------------------------------------------------------------------

def scalar(name: str, value: float) -> None:
    """Un valor en el tiempo (perdida, PSNR, numero de keypoints, ...).
    Combinar con step() para obtener una curva."""
    _ensure_started()
    rr.log(name, rr.Scalars(float(value)))


def text(name: str, msg: str, level: str = "INFO") -> None:
    """Mensaje de texto en el registro (log) del visor."""
    _ensure_started()
    rr.log(name, rr.TextLog(str(msg), level=level))


# ---------------------------------------------------------------------------
# Internos
# ---------------------------------------------------------------------------

def _ensure_started() -> None:
    if not _started:
        start()


def _to_displayable(img: np.ndarray) -> np.ndarray:
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    if np.issubdtype(img.dtype, np.floating):
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
        if lo >= 0.0 and hi <= 1.0:
            return img.astype(np.float32)
        if hi - lo < 1e-12:
            return np.zeros(img.shape, np.uint8)
        return ((img - lo) / (hi - lo) * 255).astype(np.uint8)
    if img.dtype in (np.int16, np.int32, np.int64):  # p. ej. cv2.Sobel con CV_16S
        img = np.abs(img)
        return (img * (255.0 / max(int(img.max()), 1))).astype(np.uint8)
    return img


def _as_strips(strips, dim: int):
    if isinstance(strips, np.ndarray):
        a = strips.astype(np.float32)
        if a.ndim == 2 and a.shape[1] == dim:
            return [a]
        if a.ndim == 2 and a.shape[1] == 2 * dim:          # segmentos (N, 4)/(N, 6)
            return list(a.reshape(-1, 2, dim))
        if a.ndim == 3 and a.shape[1:] == (1, 2 * dim):    # HoughLinesP (N, 1, 4)
            return list(a.reshape(-1, 2, dim))
        if a.ndim == 3 and a.shape[2] == dim:              # (N, M, dim)
            return list(a)
        raise ValueError(f"forma {a.shape} no valida para lineas {dim}D")
    return [np.asarray(s, dtype=np.float32).reshape(-1, dim) for s in strips]
