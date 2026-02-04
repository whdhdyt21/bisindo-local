import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO
from PIL import Image

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="BISINDO Detection",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.image("assets/logo.png", width=160)
st.sidebar.markdown("## **BISINDO Detection**")

mode = st.sidebar.radio(
    "🔎 Mode Deteksi",
    ["Webcam (Realtime)", "Unggah Gambar"]
)

confidence = st.sidebar.slider(
    "🎯 Ambang Confidence",
    0.05, 1.0, 0.30, 0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Model** : YOLO v11  
    **Tugas** : Deteksi Isyarat Tangan  
    **Bahasa** : BISINDO  
    """
)

hasil_deteksi_box = st.sidebar.empty()

# =====================================================
# HEADER UTAMA
# =====================================================
st.markdown(
    """
    <h1 style="margin-bottom:0;">🖐️ Deteksi Bahasa Isyarat Indonesia (BISINDO)</h1>
    <p style="color:gray;">
    Sistem pendeteksi isyarat tangan menggunakan model YOLO untuk pengenalan
    Bahasa Isyarat Indonesia secara realtime dan citra statis.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =====================================================
# MODE 1: WEBCAM
# =====================================================
if mode == "Webcam (Realtime)":

    if "kamera_aktif" not in st.session_state:
        st.session_state.kamera_aktif = False

    col_kiri, col_kanan = st.columns([4, 1])

    with col_kanan:
        st.markdown("### 🎥 Kontrol Kamera")
        if st.button("▶ Mulai Kamera"):
            st.session_state.kamera_aktif = True
        if st.button("⏹ Hentikan Kamera"):
            st.session_state.kamera_aktif = False

    frame_area = col_kiri.empty()
    fps_area = col_kanan.empty()

    if st.session_state.kamera_aktif:
        cap = cv2.VideoCapture(0)
        waktu_mulai = time.time()

        ret, frame = cap.read()
        if ret:
            hasil = model(frame, conf=confidence, imgsz=640)
            frame_annotasi = hasil[0].plot()
            frame_annotasi = cv2.cvtColor(frame_annotasi, cv2.COLOR_BGR2RGB)

            frame_area.image(
                frame_annotasi,
                caption="Hasil Deteksi Realtime",
                use_container_width=True
            )

            fps = 1 / (time.time() - waktu_mulai)
            fps_area.metric("FPS", f"{fps:.2f}")

            daftar_label = []
            if hasil[0].boxes is not None:
                for box in hasil[0].boxes:
                    label = model.names[int(box.cls[0])]
                    skor = float(box.conf[0])
                    daftar_label.append(f"{label} ({skor:.2f})")

            if daftar_label:
                hasil_deteksi_box.success(
                    "### ✅ Isyarat Terdeteksi\n" +
                    "\n".join([f"- {l}" for l in daftar_label])
                )
            else:
                hasil_deteksi_box.info("Tidak ada isyarat tangan terdeteksi")

        cap.release()
    else:
        frame_area.info("Klik **Mulai Kamera** untuk memulai deteksi.")

# =====================================================
# MODE 2: UNGGAH GAMBAR
# =====================================================
else:
    file_gambar = st.file_uploader(
        "📤 Unggah Gambar Isyarat Tangan",
        type=["jpg", "jpeg", "png"]
    )

    if file_gambar is not None:
        gambar = Image.open(file_gambar).convert("RGB")
        img_np = np.array(gambar)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🖼️ Gambar Asli")
            st.image(gambar, use_container_width=True)

        hasil = model(img_np, conf=confidence, imgsz=640)
        img_hasil = hasil[0].plot()

        with col2:
            st.markdown("### 📌 Hasil Deteksi")
            st.image(img_hasil, use_container_width=True)

        daftar_label = []
        if hasil[0].boxes is not None:
            for box in hasil[0].boxes:
                label = model.names[int(box.cls[0])]
                skor = float(box.conf[0])
                daftar_label.append(f"{label} ({skor:.2f})")

        if daftar_label:
            hasil_deteksi_box.success(
                "### ✅ Isyarat Terdeteksi\n" +
                "\n".join([f"- {l}" for l in daftar_label])
            )
        else:
            hasil_deteksi_box.warning(
                "Tidak ada isyarat terdeteksi.\n\n"
                "_Kemungkinan perbedaan distribusi data (domain shift)._"
            )
