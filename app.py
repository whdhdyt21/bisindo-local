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
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS - DARK THEME
# =====================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Root Variables */
    :root {
        --bg-primary: #0f1419;
        --bg-secondary: #1a1f2e;
        --bg-card: #1e2433;
        --bg-card-hover: #252d3d;
        --accent-gold: #f5a623;
        --accent-cyan: #00d4ff;
        --accent-purple: #a855f7;
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-color: #2d3748;
        --gradient-gold: linear-gradient(135deg, #f5a623 0%, #f7c35c 100%);
        --gradient-cyan: linear-gradient(135deg, #00d4ff 0%, #00f7ff 100%);
    }
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Hide Streamlit Default */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding: 1.5rem 2rem 2rem 2rem;
        max-width: 1400px;
    }
    
    /* Header */
    .header-container {
        background: var(--bg-secondary);
        padding: 1.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid var(--border-color);
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .header-left .logo {
        font-size: 2.2rem;
    }
    
    .header-left h1 {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-left h1 span {
        background: var(--gradient-gold);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-left p {
        color: var(--text-secondary);
        font-size: 0.8rem;
        margin: 0;
    }
    
    .header-stats {
        display: flex;
        gap: 2rem;
    }
    
    .header-stat {
        text-align: center;
    }
    
    .header-stat .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--accent-gold);
    }
    
    .header-stat .label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Card Style */
    .card {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 1.2rem;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    
    .card:hover {
        background: var(--bg-card-hover);
        border-color: var(--accent-gold);
    }
    
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .card-title .icon {
        color: var(--accent-gold);
    }
    
    /* Section Title */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .section-title .highlight {
        color: var(--accent-gold);
    }
    
    /* Image Box */
    .image-box {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px solid var(--border-color);
    }
    
    .image-box-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Gesture Tags */
    .gesture-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }
    
    .gesture-tag {
        background: var(--bg-secondary);
        color: var(--text-secondary);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 500;
        border: 1px solid var(--border-color);
        transition: all 0.2s;
    }
    
    .gesture-tag:hover {
        background: var(--accent-gold);
        color: var(--bg-primary);
        border-color: var(--accent-gold);
    }
    
    /* Detection Badge */
    .detection-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 0.8rem;
    }
    
    .detection-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: var(--gradient-gold);
        color: var(--bg-primary);
        padding: 0.7rem 1.2rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(245, 166, 35, 0.3);
    }
    
    .detection-badge .conf {
        background: rgba(0,0,0,0.2);
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* No Detection */
    .no-detection {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 0.9rem;
    }
    
    .no-detection .icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        background: var(--bg-secondary);
        border-radius: 16px;
        border: 1px dashed var(--border-color);
    }
    
    .empty-state .icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .empty-state h3 {
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 0.3rem 0;
    }
    
    .empty-state p {
        color: var(--text-muted);
        font-size: 0.85rem;
        margin: 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.2rem;
        color: var(--text-muted);
        font-size: 0.75rem;
        margin-top: 1.5rem;
        border-top: 1px solid var(--border-color);
    }
    
    .footer strong {
        color: var(--accent-gold);
    }
    
    /* Streamlit Overrides */
    .stSlider > div > div > div {
        background: var(--accent-gold) !important;
    }
    
    .stSlider > div > div > div > div {
        background: var(--accent-gold) !important;
    }
    
    div[data-baseweb="slider"] > div {
        background: var(--border-color) !important;
    }
    
    div[data-baseweb="slider"] > div > div {
        background: var(--accent-gold) !important;
    }
    
    .stButton > button {
        background: var(--gradient-gold) !important;
        color: var(--bg-primary) !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        width: 100% !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 15px rgba(245, 166, 35, 0.25) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(245, 166, 35, 0.35) !important;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #00a8cc 100%) !important;
        color: var(--bg-primary) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* File Uploader */
    .stFileUploader > div > div {
        background: var(--bg-secondary) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: var(--accent-gold) !important;
        background: var(--bg-card) !important;
    }
    
    div[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        color: var(--text-muted) !important;
        font-size: 0.8rem !important;
    }
    
    div[data-testid="stFileUploaderDropzoneInstructions"] > div > span > small {
        color: var(--text-muted) !important;
    }
    
    button[data-testid="stBaseButton-secondary"] {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Slider Label */
    .stSlider label {
        color: var(--text-secondary) !important;
    }
    
    /* Caption */
    .stCaption {
        color: var(--text-muted) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    
    /* Divider */
    hr {
        border-color: var(--border-color) !important;
    }
    
    /* Label Text */
    .label-text {
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOG CSV
# =====================================================
LOG_FILE = "detection_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "nama_file", "label", "confidence"])

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="header-container">
    <div class="header-left">
        <span class="logo">🤟</span>
        <div>
            <h1>BISINDO <span>Detection</span></h1>
            <p>Deteksi Bahasa Isyarat Indonesia dengan YOLOv11</p>
        </div>
    </div>
    <div class="header-stats">
        <div class="header-stat">
            <div class="value">94.46%</div>
            <div class="label">Akurasi</div>
        </div>
        <div class="header-stat">
            <div class="value">39 FPS</div>
            <div class="label">Kecepatan</div>
        </div>
        <div class="header-stat">
            <div class="value">47</div>
            <div class="label">Gesture</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MAIN LAYOUT
# =====================================================
col_left, col_right = st.columns([1, 3])

# =====================================================
# LEFT PANEL
# =====================================================
with col_left:
    # Confidence
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="icon">🎯</span> Confidence</div>', unsafe_allow_html=True)
    confidence = st.slider(
        "conf_slider",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        label_visibility="collapsed"
    )
    st.caption(f"Threshold: {confidence:.0%}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Upload
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="icon">📤</span> Upload Gambar</div>', unsafe_allow_html=True)
    file_gambar = st.file_uploader(
        "upload_file",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gesture List
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="icon">🖐️</span> Gesture</div>', unsafe_allow_html=True)
    
    st.markdown('<p class="label-text">Huruf</p>', unsafe_allow_html=True)
    huruf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    st.markdown(
        '<div class="gesture-tags">' + 
        ''.join([f'<span class="gesture-tag">{h}</span>' for h in huruf]) + 
        '</div>', 
        unsafe_allow_html=True
    )
    
    st.markdown('<br><p class="label-text">Kata</p>', unsafe_allow_html=True)
    kata = ["AKU", "APA", "AYAH", "BAIK", "BANTU", "DIA", "JANGAN", "KAKAK", 
            "KAMU", "KAPAN", "KEREN", "KERJA", "MAAF", "MARAH", "MINUM", 
            "RUMAH", "SABAR", "SEDIH", "SENANG", "SUKA", "BERMAIN"]
    st.markdown(
        '<div class="gesture-tags">' + 
        ''.join([f'<span class="gesture-tag">{k}</span>' for k in kata]) + 
        '</div>', 
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Download
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button(
                label="📥 Download Log",
                data=f,
                file_name="detection_log.csv",
                mime="text/csv",
                use_container_width=True
            )

# =====================================================
# RIGHT PANEL
# =====================================================
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 <span class="highlight">Hasil</span> Deteksi</div>', unsafe_allow_html=True)
    
    if file_gambar is not None:
        # Process
        gambar = Image.open(file_gambar).convert("RGB")
        img_np = np.array(gambar)
        img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Inference
        hasil = model(img_np_bgr, conf=confidence, imgsz=640)
        img_hasil = hasil[0].plot()
        img_hasil_rgb = cv2.cvtColor(img_hasil, cv2.COLOR_BGR2RGB)
        
        # Images
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.markdown('<div class="image-box"><div class="image-box-title">🖼️ Input</div></div>', unsafe_allow_html=True)
            st.image(gambar, use_container_width=True)
        
        with img_col2:
            st.markdown('<div class="image-box"><div class="image-box-title">✅ Output</div></div>', unsafe_allow_html=True)
            st.image(img_hasil_rgb, use_container_width=True)
        
        # Results
        daftar_label = []
        if hasil[0].boxes is not None and len(hasil[0].boxes) > 0:
            for box in hasil[0].boxes:
                label = model.names[int(box.cls[0])]
                skor = float(box.conf[0])
                daftar_label.append((label, skor))
                
                with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        file_gambar.name,
                        label,
                        f"{skor:.4f}"
                    ])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if daftar_label:
            badges_html = '<div class="detection-badges">'
            for label, skor in daftar_label:
                badges_html += f'<div class="detection-badge">{label}<span class="conf">{skor:.0%}</span></div>'
            badges_html += '</div>'
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="no-detection">
                <div class="icon">⚠️</div>
                <div>Tidak ada gesture terdeteksi — coba turunkan threshold</div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📷</div>
            <h3>Upload Gambar untuk Memulai</h3>
            <p>Pilih gambar gesture BISINDO dari panel kiri</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    <strong>BISINDO Detection</strong> • YOLOv11 • Universitas Negeri Semarang
</div>
""", unsafe_allow_html=True)