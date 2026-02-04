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
# CUSTOM CSS - MINIMALIST DARK THEME
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0f1419;
        --bg-card: #1a1f2b;
        --accent: #f5a623;
        --accent-dim: rgba(245, 166, 35, 0.15);
        --text: #ffffff;
        --text-dim: #6b7280;
        --border: #2a3441;
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--bg-primary);
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    .block-container {
        padding: 1rem 1.5rem !important;
        max-width: 1300px !important;
    }
    
    /* Header */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.5rem;
        background: var(--bg-card);
        border-radius: 12px;
        border: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .header h1 {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text);
        margin: 0;
    }
    
    .header h1 span { color: var(--accent); }
    
    .header p {
        font-size: 0.75rem;
        color: var(--text-dim);
        margin: 0;
    }
    
    .stats {
        display: flex;
        gap: 1.5rem;
    }
    
    .stat {
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--accent);
    }
    
    .stat-label {
        font-size: 0.6rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Main Grid */
    .main-grid {
        display: grid;
        grid-template-columns: 240px 1fr;
        gap: 1rem;
        height: calc(100vh - 140px);
    }
    
    /* Panel */
    .panel {
        background: var(--bg-card);
        border-radius: 12px;
        border: 1px solid var(--border);
        padding: 1rem;
        height: fit-content;
    }
    
    .panel-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .panel-title span { color: var(--accent); }
    
    /* Tags */
    .tags {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
    }
    
    .tag {
        background: var(--bg-primary);
        color: var(--text-dim);
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 500;
        border: 1px solid var(--border);
    }
    
    /* Detection Area */
    .detection-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
    }
    
    .img-box {
        background: var(--bg-primary);
        border-radius: 10px;
        padding: 0.6rem;
        border: 1px solid var(--border);
    }
    
    .img-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-dim);
        margin-bottom: 0.4rem;
    }
    
    /* Result Badge */
    .result-area {
        margin-top: 0.8rem;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #f5a623 0%, #e09000 100%);
        color: #000;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-conf {
        background: rgba(0,0,0,0.2);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    
    /* Empty State */
    .empty {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 300px;
        color: var(--text-dim);
        font-size: 0.85rem;
    }
    
    .empty-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        opacity: 0.4;
    }
    
    /* No Detection */
    .no-detect {
        background: var(--accent-dim);
        color: var(--accent);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        text-align: center;
    }
    
    /* Footer */
    .foot {
        text-align: center;
        padding: 0.8rem;
        color: var(--text-dim);
        font-size: 0.65rem;
        border-top: 1px solid var(--border);
        margin-top: 1rem;
    }
    
    /* Streamlit Overrides */
    .stSlider label { display: none !important; }
    .stSlider > div { padding-top: 0 !important; }
    
    .stFileUploader label { display: none !important; }
    .stFileUploader > div > div {
        background: var(--bg-primary) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 8px !important;
        padding: 0.8rem !important;
    }
    
    .stDownloadButton button {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        padding: 0.5rem !important;
        width: 100% !important;
    }
    
    .stCaption { color: var(--text-dim) !important; font-size: 0.7rem !important; }
    
    div[data-baseweb="slider"] { background: var(--border) !important; }
    div[data-baseweb="slider"] > div { background: var(--accent) !important; }
    
    section[data-testid="stFileUploadDropzone"] span { font-size: 0.75rem !important; }
    section[data-testid="stFileUploadDropzone"] small { font-size: 0.65rem !important; }
    
    button[data-testid="baseButton-secondary"] {
        background: var(--bg-card) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        font-size: 0.75rem !important;
    }
    
    .label-sm {
        font-size: 0.7rem;
        color: var(--text-dim);
        margin-bottom: 0.3rem;
    }
    
    .divider {
        height: 1px;
        background: var(--border);
        margin: 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# INIT
# =====================================================
LOG_FILE = "detection_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "file", "label", "confidence"])

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="header">
    <div class="header-left">
        <span style="font-size:1.8rem">🤟</span>
        <div>
            <h1>BISINDO <span>Detection</span></h1>
            <p>Bahasa Isyarat Indonesia • YOLOv11</p>
        </div>
    </div>
    <div class="stats">
        <div class="stat"><div class="stat-value">94.5%</div><div class="stat-label">Akurasi</div></div>
        <div class="stat"><div class="stat-value">39fps</div><div class="stat-label">Speed</div></div>
        <div class="stat"><div class="stat-value">47</div><div class="stat-label">Gesture</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LAYOUT
# =====================================================
left, right = st.columns([1, 4])

with left:
    # Confidence
    st.markdown('<div class="panel-title"><span>🎯</span> Confidence</div>', unsafe_allow_html=True)
    conf = st.slider("c", 0.10, 0.90, 0.25, 0.05, label_visibility="collapsed")
    st.caption(f"{conf:.0%}")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Upload
    st.markdown('<div class="panel-title"><span>📤</span> Upload</div>', unsafe_allow_html=True)
    file = st.file_uploader("u", ["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Gestures
    st.markdown('<div class="panel-title"><span>🖐️</span> Gesture</div>', unsafe_allow_html=True)
    st.markdown('<p class="label-sm">Huruf</p>', unsafe_allow_html=True)
    st.markdown('<div class="tags">' + ''.join([f'<span class="tag">{c}</span>' for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]) + '</div>', unsafe_allow_html=True)
    
    st.markdown('<p class="label-sm" style="margin-top:0.5rem">Kata</p>', unsafe_allow_html=True)
    kata = ["AKU","APA","AYAH","BAIK","BANTU","DIA","JANGAN","KAKAK","KAMU","KAPAN","KEREN","KERJA","MAAF","MARAH","MINUM","RUMAH","SABAR","SEDIH","SENANG","SUKA","BERMAIN"]
    st.markdown('<div class="tags">' + ''.join([f'<span class="tag">{k}</span>' for k in kata]) + '</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Download
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("📥 Download Log", f, "log.csv", "text/csv", use_container_width=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span>📊</span> Hasil Deteksi</div>', unsafe_allow_html=True)
    
    if file:
        img = Image.open(file).convert("RGB")
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        result = model(img_np, conf=conf, imgsz=640)
        img_out = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="img-box"><div class="img-label">🖼️ Input</div></div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        with c2:
            st.markdown('<div class="img-box"><div class="img-label">✅ Output</div></div>', unsafe_allow_html=True)
            st.image(img_out, use_container_width=True)
        
        # Results
        labels = []
        if result[0].boxes and len(result[0].boxes) > 0:
            for box in result[0].boxes:
                lbl = model.names[int(box.cls[0])]
                scr = float(box.conf[0])
                labels.append((lbl, scr))
                with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file.name, lbl, f"{scr:.4f}"])
        
        st.markdown('<div class="result-area">', unsafe_allow_html=True)
        if labels:
            badges = ''.join([f'<span class="badge">{l}<span class="badge-conf">{s:.0%}</span></span>' for l, s in labels])
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-detect">⚠️ Tidak terdeteksi — turunkan threshold</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty"><div class="empty-icon">📷</div>Upload gambar untuk memulai</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="foot"><b>BISINDO Detection</b> • YOLOv11 • Universitas Negeri Semarang</div>', unsafe_allow_html=True)