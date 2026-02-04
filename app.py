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
# CUSTOM CSS - MODERN & PROPORSIONAL
# =====================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Reset & Global */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: #f8fafc;
    }
    
    /* Hide Streamlit Default */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1400px;
    }
    
    /* Compact Header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.25);
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .header-left .logo {
        font-size: 2.5rem;
    }
    
    .header-left h1 {
        color: white;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-left p {
        color: rgba(255,255,255,0.85);
        font-size: 0.85rem;
        margin: 0;
    }
    
    .header-stats {
        display: flex;
        gap: 1.5rem;
    }
    
    .header-stat {
        text-align: center;
        color: white;
    }
    
    .header-stat .value {
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .header-stat .label {
        font-size: 0.7rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Main Content Grid */
    .content-grid {
        display: grid;
        grid-template-columns: 280px 1fr;
        gap: 1.2rem;
        align-items: start;
    }
    
    /* Control Panel (Left) */
    .control-panel {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #e8ecf0;
        position: sticky;
        top: 1rem;
    }
    
    .control-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .control-section {
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .control-section:last-child {
        margin-bottom: 0;
        padding-bottom: 0;
        border-bottom: none;
    }
    
    /* Gesture Tags */
    .gesture-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
    }
    
    .gesture-tag {
        background: #f0f4f8;
        color: #4a5568;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    
    /* Detection Area (Right) */
    .detection-area {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #e8ecf0;
    }
    
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Image Grid */
    .image-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    
    .image-box {
        background: #f8fafc;
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px solid #e8ecf0;
    }
    
    .image-box-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Result Section */
    .result-section {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #f0f0f0;
    }
    
    /* Detection Badge */
    .detection-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.5rem;
    }
    
    .detection-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.25);
    }
    
    .detection-badge .conf {
        background: rgba(255,255,255,0.2);
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* No Detection */
    .no-detection {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 0.9rem;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 12px;
    }
    
    .empty-state .icon {
        font-size: 3.5rem;
        margin-bottom: 0.8rem;
    }
    
    .empty-state h3 {
        color: #334155;
        font-size: 1.1rem;
        margin: 0 0 0.3rem 0;
    }
    
    .empty-state p {
        color: #64748b;
        font-size: 0.85rem;
        margin: 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 1.5rem;
    }
    
    /* Streamlit Overrides */
    .stSlider > div > div {
        background: #667eea;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* File Uploader */
    .stFileUploader > div {
        padding: 0;
    }
    
    .stFileUploader > div > div {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 1.5rem;
        transition: all 0.2s;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #667eea;
        background: #f5f3ff;
    }
    
    /* Hide extra elements */
    .stFileUploader > label {display: none;}
    div[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        font-size: 0.85rem;
        color: #64748b;
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
            <h1>BISINDO Detection</h1>
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
# MAIN LAYOUT: 2 COLUMNS
# =====================================================
col_left, col_right = st.columns([1, 3])

# =====================================================
# LEFT: CONTROL PANEL
# =====================================================
with col_left:
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
    # Confidence Slider
    st.markdown('<div class="control-title">🎯 Confidence</div>', unsafe_allow_html=True)
    confidence = st.slider(
        "conf",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        label_visibility="collapsed"
    )
    st.caption(f"Threshold: {confidence:.0%}")
    
    st.markdown('<div class="control-section"></div>', unsafe_allow_html=True)
    
    # Upload
    st.markdown('<div class="control-title">📤 Upload Gambar</div>', unsafe_allow_html=True)
    file_gambar = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="control-section"></div>', unsafe_allow_html=True)
    
    # Supported Gestures
    st.markdown('<div class="control-title">🖐️ Gesture</div>', unsafe_allow_html=True)
    
    st.markdown("**Huruf**", unsafe_allow_html=True)
    huruf = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split()
    st.markdown(
        '<div class="gesture-tags">' + 
        ''.join([f'<span class="gesture-tag">{h}</span>' for h in huruf]) + 
        '</div>', 
        unsafe_allow_html=True
    )
    
    st.markdown("<br>**Kata**", unsafe_allow_html=True)
    kata = ["AKU", "APA", "AYAH", "BAIK", "BANTU", "DIA", "JANGAN", "KAKAK", 
            "KAMU", "KAPAN", "KEREN", "KERJA", "MAAF", "MARAH", "MINUM", 
            "RUMAH", "SABAR", "SEDIH", "SENANG", "SUKA", "BERMAIN"]
    st.markdown(
        '<div class="gesture-tags">' + 
        ''.join([f'<span class="gesture-tag">{k}</span>' for k in kata]) + 
        '</div>', 
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="control-section"></div>', unsafe_allow_html=True)
    
    # Download Log
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button(
                label="📥 Download Log",
                data=f,
                file_name="detection_log.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# RIGHT: DETECTION AREA
# =====================================================
with col_right:
    st.markdown('<div class="detection-area">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Hasil Deteksi</div>', unsafe_allow_html=True)
    
    if file_gambar is not None:
        # Process Image
        gambar = Image.open(file_gambar).convert("RGB")
        img_np = np.array(gambar)
        img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Inference
        hasil = model(img_np_bgr, conf=confidence, imgsz=640)
        img_hasil = hasil[0].plot()
        img_hasil_rgb = cv2.cvtColor(img_hasil, cv2.COLOR_BGR2RGB)
        
        # Display Images Side by Side
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.markdown("""
            <div class="image-box">
                <div class="image-box-title">🖼️ Input</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(gambar, use_container_width=True)
        
        with img_col2:
            st.markdown("""
            <div class="image-box">
                <div class="image-box-title">✅ Output</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(img_hasil_rgb, use_container_width=True)
        
        # Extract Results
        daftar_label = []
        if hasil[0].boxes is not None and len(hasil[0].boxes) > 0:
            for box in hasil[0].boxes:
                label = model.names[int(box.cls[0])]
                skor = float(box.conf[0])
                daftar_label.append((label, skor))
                
                # Log to CSV
                with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        file_gambar.name,
                        label,
                        f"{skor:.4f}"
                    ])
        
        # Display Results
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        
        if daftar_label:
            badges_html = '<div class="detection-badges">'
            for label, skor in daftar_label:
                badges_html += f'<div class="detection-badge">{label}<span class="conf">{skor:.0%}</span></div>'
            badges_html += '</div>'
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="no-detection">
                ⚠️ Tidak ada gesture terdeteksi — coba turunkan confidence threshold
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Empty State
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📷</div>
            <h3>Upload Gambar untuk Memulai</h3>
            <p>Pilih gambar gesture BISINDO dari panel sebelah kiri</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    <strong>BISINDO Gesture Detection</strong> • YOLOv11l • Universitas Negeri Semarang
</div>
""", unsafe_allow_html=True)