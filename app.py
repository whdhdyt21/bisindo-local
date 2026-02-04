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
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS - MODERN UI
# =====================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.8rem;
        margin-bottom: 0;
    }
    
    /* Stats Cards */
    .stat-card {
        background: white;
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        transition: transform 0.2s, box-shadow 0.2s;
        text-align: center;
    }
    
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .stat-card .icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .stat-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .stat-card .label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Result Cards */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    .result-card h3 {
        color: #1a1a2e;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Detection Badge */
    .detection-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0.3rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .detection-badge .confidence {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    
    /* No Detection */
    .no-detection {
        background: #fff3cd;
        color: #856404;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #ffeeba;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #1a1a2e;
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
    }
    
    .info-box p {
        color: #555;
        margin: 0;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    
    /* Gesture Grid */
    .gesture-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 0.5rem;
    }
    
    .gesture-item {
        background: #f0f2f5;
        padding: 0.4rem 0.7rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #333;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecef 100%);
        border-radius: 20px;
        margin: 2rem 0;
    }
    
    .empty-state .icon {
        font-size: 5rem;
        margin-bottom: 1rem;
    }
    
    .empty-state h3 {
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .empty-state p {
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

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
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 4rem;">🤟</div>
        <h2 style="margin: 0.5rem 0; color: #1a1a2e;">BISINDO Detection</h2>
        <p style="color: #666; font-size: 0.9rem;">Bahasa Isyarat Indonesia</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Confidence Slider
    st.markdown("#### 🎯 Pengaturan Deteksi")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="Turunkan nilai jika gesture tidak terdeteksi"
    )
    
    st.markdown("---")
    
    # Model Info
    st.markdown("""
    <div class="info-box">
        <h4>📊 Informasi Model</h4>
        <p>
        <strong>Arsitektur:</strong> YOLOv11l<br>
        <strong>Akurasi:</strong> 94.46% mAP@0.5<br>
        <strong>Kecepatan:</strong> 39.04 FPS<br>
        <strong>Total Gesture:</strong> 47 kelas
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Supported Gestures
    st.markdown("#### 🖐️ Gesture yang Didukung")
    
    with st.expander("📝 Huruf (A-Z)", expanded=False):
        st.markdown("""
        <div class="gesture-grid">
            <span class="gesture-item">A</span>
            <span class="gesture-item">B</span>
            <span class="gesture-item">C</span>
            <span class="gesture-item">D</span>
            <span class="gesture-item">E</span>
            <span class="gesture-item">F</span>
            <span class="gesture-item">G</span>
            <span class="gesture-item">H</span>
            <span class="gesture-item">I</span>
            <span class="gesture-item">J</span>
            <span class="gesture-item">K</span>
            <span class="gesture-item">L</span>
            <span class="gesture-item">M</span>
            <span class="gesture-item">N</span>
            <span class="gesture-item">O</span>
            <span class="gesture-item">P</span>
            <span class="gesture-item">Q</span>
            <span class="gesture-item">R</span>
            <span class="gesture-item">S</span>
            <span class="gesture-item">T</span>
            <span class="gesture-item">U</span>
            <span class="gesture-item">V</span>
            <span class="gesture-item">W</span>
            <span class="gesture-item">X</span>
            <span class="gesture-item">Y</span>
            <span class="gesture-item">Z</span>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("💬 Kata", expanded=False):
        st.markdown("""
        <div class="gesture-grid">
            <span class="gesture-item">AKU</span>
            <span class="gesture-item">APA</span>
            <span class="gesture-item">AYAH</span>
            <span class="gesture-item">BAIK</span>
            <span class="gesture-item">BANTU</span>
            <span class="gesture-item">BERMAIN</span>
            <span class="gesture-item">DIA</span>
            <span class="gesture-item">JANGAN</span>
            <span class="gesture-item">KAKAK</span>
            <span class="gesture-item">KAMU</span>
            <span class="gesture-item">KAPAN</span>
            <span class="gesture-item">KEREN</span>
            <span class="gesture-item">KERJA</span>
            <span class="gesture-item">MAAF</span>
            <span class="gesture-item">MARAH</span>
            <span class="gesture-item">MINUM</span>
            <span class="gesture-item">RUMAH</span>
            <span class="gesture-item">SABAR</span>
            <span class="gesture-item">SEDIH</span>
            <span class="gesture-item">SENANG</span>
            <span class="gesture-item">SUKA</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Detection Result Placeholder
    st.markdown("---")
    st.markdown("#### 📋 Hasil Terakhir")
    hasil_deteksi_box = st.empty()

# =====================================================
# MAIN CONTENT
# =====================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🤟 BISINDO Gesture Detection</h1>
    <p>Sistem pendeteksi Bahasa Isyarat Indonesia berbasis Deep Learning dengan model YOLOv11l. 
    Upload gambar gesture tangan untuk memulai deteksi.</p>
</div>
""", unsafe_allow_html=True)

# Stats Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="icon">🎯</div>
        <div class="value">94.46%</div>
        <div class="label">Akurasi</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="icon">⚡</div>
        <div class="value">39 FPS</div>
        <div class="label">Kecepatan</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="icon">🖐️</div>
        <div class="value">47</div>
        <div class="label">Gesture</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="icon">🧠</div>
        <div class="value">YOLOv11</div>
        <div class="label">Model</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Upload Section
st.markdown("""
<div class="result-card">
    <h3>📤 Upload Gambar Isyarat Tangan</h3>
</div>
""", unsafe_allow_html=True)

file_gambar = st.file_uploader(
    "Pilih gambar (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# Processing
if file_gambar is not None:
    # Read & convert image
    gambar = Image.open(file_gambar).convert("RGB")
    img_np = np.array(gambar)
    img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="result-card">
            <h3>🖼️ Gambar Input</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(gambar, use_container_width=True)

    # Inference
    hasil = model(img_np_bgr, conf=confidence, imgsz=640)
    img_hasil = hasil[0].plot()
    img_hasil_rgb = cv2.cvtColor(img_hasil, cv2.COLOR_BGR2RGB)

    with col2:
        st.markdown("""
        <div class="result-card">
            <h3>📌 Hasil Deteksi</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(img_hasil_rgb, use_container_width=True)

    # Extract labels & confidence
    daftar_label = []
    log_rows = []

    if hasil[0].boxes is not None and len(hasil[0].boxes) > 0:
        for box in hasil[0].boxes:
            label = model.names[int(box.cls[0])]
            skor = float(box.conf[0])
            daftar_label.append((label, skor))

            log_rows.append([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                file_gambar.name,
                label,
                f"{skor:.4f}",
                "image_upload"
            ])

    # Display Results
    st.markdown("<br>", unsafe_allow_html=True)
    
    if daftar_label:
        st.markdown("""
        <div class="result-card">
            <h3>✅ Gesture Terdeteksi</h3>
        </div>
        """, unsafe_allow_html=True)
        
        badges_html = ""
        for label, skor in daftar_label:
            badges_html += f"""
            <div class="detection-badge">
                {label}
                <span class="confidence">{skor:.1%}</span>
            </div>
            """
        
        st.markdown(badges_html, unsafe_allow_html=True)
        
        # Update sidebar
        hasil_deteksi_box.success(f"✅ {', '.join([l[0] for l in daftar_label])}")
    else:
        st.markdown("""
        <div class="no-detection">
            <strong>⚠️ Tidak ada gesture terdeteksi</strong><br>
            <small>Coba turunkan confidence threshold atau gunakan gambar dengan kualitas lebih baik</small>
        </div>
        """, unsafe_allow_html=True)
        
        hasil_deteksi_box.warning("Tidak terdeteksi")

    # Save to CSV
    if log_rows:
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(log_rows)

else:
    # Empty State
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📷</div>
        <h3>Belum ada gambar</h3>
        <p>Upload gambar gesture BISINDO untuk memulai deteksi</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# DOWNLOAD LOG SECTION
# =====================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="result-card">
    <h3>📥 Download Log Deteksi</h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("Download riwayat hasil deteksi dalam format CSV")

with col2:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button(
                label="⬇️ Download CSV",
                data=f,
                file_name="detection_log.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Belum ada log")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="custom-footer">
    <p>
        <strong>BISINDO Gesture Detection</strong><br>
        Dibuat untuk Tugas Akhir/Skripsi<br>
        <strong>Universitas Negeri Semarang</strong><br><br>
        Model: YOLOv11l | Dataset: BISINDO v16 | Accuracy: 94.46% mAP@0.5
    </p>
</div>
""", unsafe_allow_html=True)