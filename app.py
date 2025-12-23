import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------------------------
# Configuración Streamlit
# ---------------------------
st.set_page_config(page_title="EPP Detector Demo", layout="wide")
DEFAULT_MODEL_PATH = "weights/best.pt"

# ---------------------------
# Modelo
# ---------------------------
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

# ---------------------------
# Colores por clase
# ---------------------------
def color_for_class(cls_id: int):
    # Paleta BGR (OpenCV), estable
    palette = [
        (255, 56, 56),
        (255, 157, 151),
        (255, 112, 31),
        (255, 178, 29),
        (207, 210, 49),
        (72, 249, 10),
        (146, 204, 23),
        (61, 219, 134),
        (26, 147, 52),
        (0, 212, 187),
        (44, 153, 168),
        (0, 194, 255),
        (52, 69, 147),
        (100, 115, 255),
        (0, 24, 236),
        (132, 56, 255),
        (82, 0, 133),
        (203, 56, 255),
        (255, 149, 200),
        (255, 55, 199),
    ]
    return palette[cls_id % len(palette)]

# ---------------------------
# Dibujo: etiqueta con fondo transparente + clamp al frame
# ---------------------------
def draw_label_alpha(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color_bgr: tuple,
    font_scale: float,
    thickness: int,
    alpha: float = 0.35,
    pad: int = 4,
):
    """
    Dibuja un label con fondo semitransparente, asegurando que NO se salga del frame.
    x,y son la esquina superior-izquierda "deseada" del área del texto (baseline se ajusta internamente).
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Tamaño del texto
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Caja del fondo (top-left y bottom-right)
    # Queremos que el texto quede dentro de: [x+pad, y+pad+th] aprox
    bg_w = tw + 2 * pad
    bg_h = th + baseline + 2 * pad

    # Clamp X para que el fondo no se salga
    x = max(0, min(x, w - bg_w))

    # Si no entra arriba, lo ponemos abajo; si no entra abajo, lo pegamos dentro
    # "y" lo interpretamos como top del fondo
    if y < 0:
        y = 0
    if y + bg_h > h:
        y = max(0, h - bg_h)

    # Fondo semitransparente (overlay)
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + bg_w, y + bg_h),
        color_bgr,
        thickness=-1,
    )
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Texto (blanco para contraste)
    tx = x + pad
    ty = y + pad + th  # y del texto es baseline-ish; esto lo coloca dentro del fondo
    cv2.putText(
        img,
        text,
        (tx, ty),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )

def infer_image(model, image_bgr: np.ndarray, conf: float):
    """
    Inferencia + dibujo manual de bounding boxes
    - Texto auto-escalado según tamaño del bbox
    - Label con fondo transparente
    - Label nunca se sale del frame
    """
    results = model.predict(image_bgr, conf=conf, verbose=False)
    res = results[0]
    img = image_bgr.copy()

    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names

        H, W = img.shape[:2]

        for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, classes):
            # Clamp bbox al frame
            x1 = float(max(0, min(x1, W - 1)))
            y1 = float(max(0, min(y1, H - 1)))
            x2 = float(max(0, min(x2, W - 1)))
            y2 = float(max(0, min(y2, H - 1)))

            color = color_for_class(cls_id)
            label = f"{names[cls_id]} {score:.2f}"

            # BBox
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness=2
            )

            # --- Texto auto-escalado por tamaño medio del bbox ---
            box_w = max(1.0, x2 - x1)
            box_h = max(1.0, y2 - y1)
            box_mean = (box_w + box_h) / 2.0

            # Factor clave (ajústalo si quieres): más alto = texto más grande
            font_scale = box_mean * 0.003
            font_scale = max(0.35, min(font_scale, 0.95))  # límites seguros

            thickness = max(1, int(round(font_scale * 2)))

            # Posición deseada: arriba del bbox
            # Calculamos el alto del label para decidir si cabe arriba; si no, lo ponemos abajo
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            pad = 4
            bg_h = th + baseline + 2 * pad

            desired_x = int(x1)
            desired_y_top = int(y1) - bg_h - 2  # arriba del bbox
            if desired_y_top < 0:
                # Si no cabe arriba, lo ponemos abajo
                desired_y_top = int(y1) + 2

            # Dibujo label con fondo transparente y clamp automático
            draw_label_alpha(
                img,
                label,
                desired_x,
                desired_y_top,
                color_bgr=color,
                font_scale=font_scale,
                thickness=thickness,
                alpha=0.35,   # transparencia (0=transparente, 1=opaco)
                pad=pad
            )

    return img, res

def process_video(model, input_path: str, conf: float, sample_every_n_frames: int = 1):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmp_out.name
    tmp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    last_annot = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every_n_frames == 0:
            annot, _ = infer_image(model, frame, conf)
            last_annot = annot
        else:
            annot = last_annot if last_annot is not None else frame

        writer.write(annot)
        frame_idx += 1

    cap.release()
    writer.release()
    return out_path

# ---------------------------
# UI
# ---------------------------
st.title("🦺 EPP Detector (Imagen / Video / Webcam)")
st.write("Demo de detección de EPP con YOLO + Streamlit (labels auto-escalados, fondo transparente, sin salirse del frame)")

with st.sidebar:
    st.header("Configuración")
    model_path = st.text_input("Ruta del modelo", value=DEFAULT_MODEL_PATH)
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    sample_n = st.selectbox("Video: inferencia cada N frames", [1, 2, 3, 5, 10], index=2)

model_file = Path(model_path)
if not model_file.exists():
    st.error(f"No se encontró el modelo en: {model_path}")
    st.stop()

model = load_model(str(model_file))

tab1, tab2, tab3 = st.tabs(["📷 Imagen", "🎞️ Video", "📹 Webcam"])

# ---------------------------
# Imagen
# ---------------------------
with tab1:
    st.subheader("Detección en imagen")
    img_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if img_file:
        pil = Image.open(img_file).convert("RGB")
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Original", use_container_width=True)

        with col2:
            annotated_bgr, res = infer_image(model, img_bgr, conf)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Predicción", use_container_width=True)

        if res.boxes is not None and len(res.boxes) > 0:
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            names = res.names
            counts = {}
            for c in cls_ids:
                counts[names[c]] = counts.get(names[c], 0) + 1
            st.markdown("**Conteo por clase:**")
            st.json(counts)
        else:
            st.info("No se detectaron objetos con ese umbral.")

# ---------------------------
# Video
# ---------------------------
with tab2:
    st.subheader("Detección en video")
    vid_file = st.file_uploader("Sube un video", type=["mp4", "mov", "avi", "mkv"])

    if vid_file:
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=f".{vid_file.name.split('.')[-1]}")
        tmp_in.write(vid_file.read())
        tmp_in.close()

        st.video(tmp_in.name)

        if st.button("Procesar video"):
            with st.spinner("Procesando..."):
                out_path = process_video(model, tmp_in.name, conf, sample_every_n_frames=sample_n)
            st.success("Listo. Video procesado:")
            st.video(out_path)

# ---------------------------
# Webcam (simple)
# ---------------------------
with tab3:
    st.subheader("Webcam (captura simple)")
    st.write("Este modo captura una imagen. Para tiempo real, lo ideal es WebRTC.")

    cam_img = st.camera_input("Captura una imagen desde tu webcam")
    if cam_img:
        pil = Image.open(cam_img).convert("RGB")
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        annotated_bgr, _ = infer_image(model, img_bgr, conf)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, caption="Predicción", use_container_width=True)
