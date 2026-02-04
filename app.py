import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import csv
import os
from datetime import datetime

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
# KONFIGURASI LOG CSV
# =====================================================
LOG_FILE = "detection_log.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "nama_file",
            "label",
            "confidence",
            "mode"
        ])

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.image("assets/logo.png", width=160)
st.sidebar.markdown("## **BISINDO Detection**")

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

    ℹ️ *Versi online hanya mendukung deteksi melalui unggahan gambar.*
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
    Sistem pendeteksi isyarat tangan berbasis YOLO v11 untuk pengenalan
    Bahasa Isyarat Indonesia menggunakan citra statis (unggahan gambar).
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
# MODE: UNGGAH GAMBAR
# =====================================================
st.subheader("📤 Unggah Gambar Isyarat Tangan")

file_gambar = st.file_uploader(
    "Pilih gambar (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if file_gambar is not None:
    # Baca & konversi gambar
    gambar = Image.open(file_gambar).convert("RGB")
    img_np = np.array(gambar)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖼️ Gambar Asli")
        st.image(gambar, use_container_width=True)

    # Inference
    hasil = model(img_np, conf=confidence, imgsz=640)
    img_hasil = hasil[0].plot()

    with col2:
        st.markdown("### 📌 Hasil Deteksi")
        st.image(img_hasil, use_container_width=True)

    # Ambil label & confidence
    daftar_label = []
    log_rows = []

    if hasil[0].boxes is not None and len(hasil[0].boxes) > 0:
        for box in hasil[0].boxes:
            label = model.names[int(box.cls[0])]
            skor = float(box.conf[0])
            daftar_label.append(f"{label} ({skor:.2f})")

            log_rows.append([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                file_gambar.name,
                label,
                f"{skor:.4f}",
                "image_upload_online"
            ])

    # Tampilkan hasil di sidebar
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

    # =============================
    # SIMPAN KE CSV
    # =============================
    if log_rows:
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(log_rows)

# =====================================================
# DOWNLOAD CSV
# =====================================================
st.markdown("---")
st.subheader("📥 Unduh Log Hasil Deteksi")

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "rb") as f:
        st.download_button(
            label="⬇️ Download detection_log.csv",
            data=f,
            file_name="detection_log.csv",
            mime="text/csv"
        )
else:
    st.info("Belum ada data log yang tersimpan.")
