import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import csv
import os
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="BISINDO Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# DESIGN SYSTEM - DARK THEME
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-base: #0a0e13;
        --bg-surface: #12171f;
        --bg-elevated: #1a2029;
        --bg-hover: #232b38;
        
        --accent: #f0b429;
        --accent-soft: rgba(240, 180, 41, 0.12);
        --success: #10b981;
        --success-soft: rgba(16, 185, 129, 0.12);
        
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #475569;
        
        --border: #1e293b;
        --border-light: #334155;
        
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
    }
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: var(--bg-base);
        color: var(--text-primary);
    }
    
    #MainMenu, footer, header { visibility: hidden; }
    
    .block-container {
        padding: 1.5rem 2rem !important;
        max-width: 1200px !important;
    }
    
    /* ========== HEADER ========== */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    
    .brand-icon {
        width: 48px;
        height: 48px;
        background: var(--accent-soft);
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    
    .brand-text h1 {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .brand-text p {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin: 0;
    }
    
    .metrics {
        display: flex;
        gap: 0.5rem;
    }
    
    .metric {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.6rem 1rem;
        text-align: center;
        min-width: 80px;
    }
    
    .metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--accent);
    }
    
    .metric-label {
        font-size: 0.6rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* ========== MAIN CARD ========== */
    .main-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
    }
    
    .card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .card-title-icon {
        color: var(--accent);
    }
    
    /* ========== IMAGE DISPLAY ========== */
    .image-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    
    .image-wrapper {
        background: var(--bg-base);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    
    .image-header {
        padding: 0.6rem 1rem;
        border-bottom: 1px solid var(--border);
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .image-body {
        padding: 0.5rem;
    }
    
    /* ========== RESULTS ========== */
    .results-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.2rem;
        background: var(--bg-elevated);
        border-radius: var(--radius-md);
        border: 1px solid var(--border);
    }
    
    .results-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .results-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, var(--accent) 0%, #d99a1e 100%);
        color: var(--bg-base);
        padding: 0.55rem 1rem;
        border-radius: var(--radius-sm);
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    .badge-conf {
        background: rgba(0,0,0,0.15);
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .no-result {
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    
    /* ========== EMPTY STATE ========== */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        text-align: center;
    }
    
    .empty-icon {
        width: 64px;
        height: 64px;
        background: var(--bg-elevated);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border);
    }
    
    .empty-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }
    
    .empty-desc {
        font-size: 0.85rem;
        color: var(--text-muted);
    }
    
    /* ========== BOTTOM SECTION ========== */
    .bottom-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1rem;
    }
    
    .info-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
    }
    
    .info-card-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .info-card-title span {
        color: var(--accent);
    }
    
    .tags-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
    }
    
    .tag {
        background: var(--bg-base);
        color: var(--text-secondary);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 500;
        border: 1px solid var(--border);
        transition: all 0.15s ease;
    }
    
    .tag:hover {
        background: var(--accent);
        color: var(--bg-base);
        border-color: var(--accent);
    }
    
    /* ========== CONTROLS ========== */
    .control-row {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .control-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        min-width: 70px;
    }
    
    .conf-display {
        background: var(--accent-soft);
        color: var(--accent);
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    /* ========== FOOTER ========== */
    .app-footer {
        text-align: center;
        padding: 1.2rem;
        color: var(--text-muted);
        font-size: 0.7rem;
        margin-top: 1.5rem;
    }
    
    .app-footer strong {
        color: var(--accent);
    }
    
    /* ========== STREAMLIT OVERRIDES ========== */
    .stSlider { padding-top: 0 !important; }
    .stSlider label { display: none !important; }
    .stSlider > div > div > div { background: var(--accent) !important; }
    div[data-baseweb="slider"] { background: var(--border) !important; height: 4px !important; }
    div[data-baseweb="slider"] > div { background: var(--accent) !important; }
    div[data-baseweb="slider"] > div > div { 
        background: var(--accent) !important; 
        width: 16px !important; 
        height: 16px !important;
        box-shadow: 0 0 0 4px var(--accent-soft) !important;
    }
    
    .stFileUploader { margin-top: 0.5rem; }
    .stFileUploader label { display: none !important; }
    .stFileUploader > div > div {
        background: var(--bg-base) !important;
        border: 1px dashed var(--border-light) !important;
        border-radius: var(--radius-sm) !important;
        padding: 1rem !important;
        transition: all 0.15s ease !important;
    }
    .stFileUploader > div > div:hover {
        border-color: var(--accent) !important;
        background: var(--bg-elevated) !important;
    }
    
    section[data-testid="stFileUploadDropzone"] span { 
        font-size: 0.8rem !important; 
        color: var(--text-secondary) !important;
    }
    section[data-testid="stFileUploadDropzone"] small { 
        font-size: 0.7rem !important; 
        color: var(--text-muted) !important;
    }
    
    button[data-testid="baseButton-secondary"] {
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.75rem !important;
    }
    
    .stDownloadButton button {
        background: var(--success) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 0.6rem !important;
    }
    
    .stCaption {
        color: var(--text-muted) !important;
        font-size: 0.7rem !important;
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
<div class="app-header">
    <div class="brand">
        <div class="brand-icon">🤟</div>
        <div class="brand-text">
            <h1>BISINDO Detection</h1>
            <p>Bahasa Isyarat Indonesia • YOLOv11</p>
        </div>
    </div>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">94.5%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">39</div>
            <div class="metric-label">FPS</div>
        </div>
        <div class="metric">
            <div class="metric-value">47</div>
            <div class="metric-label">Classes</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MAIN DETECTION CARD
# =====================================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# Card Header with Controls
col_title, col_conf, col_upload = st.columns([2, 3, 2])

with col_title:
    st.markdown('<div class="card-title"><span class="card-title-icon">📊</span> Detection</div>', unsafe_allow_html=True)

with col_conf:
    conf_col1, conf_col2 = st.columns([4, 1])
    with conf_col1:
        conf = st.slider("conf", 0.10, 0.90, 0.25, 0.05, label_visibility="collapsed")
    with conf_col2:
        st.markdown(f'<div class="conf-display">{conf:.0%}</div>', unsafe_allow_html=True)

with col_upload:
    file = st.file_uploader("upload", ["jpg", "jpeg", "png"], label_visibility="collapsed")

st.markdown("<div style='height: 0.8rem'></div>", unsafe_allow_html=True)

# Detection Area
if file:
    img = Image.open(file).convert("RGB")
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    result = model(img_np, conf=conf, imgsz=640)
    img_out = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
    
    # Images
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('''
        <div class="image-wrapper">
            <div class="image-header">🖼️ Input Image</div>
            <div class="image-body"></div>
        </div>
        ''', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    with c2:
        st.markdown('''
        <div class="image-wrapper">
            <div class="image-header">✅ Detection Result</div>
            <div class="image-body"></div>
        </div>
        ''', unsafe_allow_html=True)
        st.image(img_out, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results
    labels = []
    if result[0].boxes and len(result[0].boxes) > 0:
        for box in result[0].boxes:
            lbl = model.names[int(box.cls[0])]
            scr = float(box.conf[0])
            labels.append((lbl, scr))
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file.name, lbl, f"{scr:.4f}"])
    
    if labels:
        badges = ''.join([f'<span class="badge">{l}<span class="badge-conf">{s:.0%}</span></span>' for l, s in labels])
        st.markdown(f'''
        <div class="results-bar">
            <div class="results-label">🎯 Detected</div>
            <div class="results-badges">{badges}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="results-bar">
            <div class="results-label">🎯 Result</div>
            <div class="no-result">No gesture detected — try lowering the threshold</div>
        </div>
        ''', unsafe_allow_html=True)

else:
    st.markdown('''
    <div class="empty-state">
        <div class="empty-icon">📷</div>
        <div class="empty-title">Upload an Image</div>
        <div class="empty-desc">Select a BISINDO gesture image to start detection</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# BOTTOM INFO CARDS
# =====================================================
st.markdown('<div class="bottom-grid">', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('''
    <div class="info-card">
        <div class="info-card-title"><span>🔤</span> Letters (A-Z)</div>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('<div class="tags-grid">' + ''.join([f'<span class="tag">{c}</span>' for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]) + '</div>', unsafe_allow_html=True)

with c2:
    kata = ["AKU", "KAMU", "APA", "DIA", "AYAH", "KAKAK", "BAIK", "MAAF", "MARAH", "SABAR", "SEDIH", "SENANG", "SUKA", "MINUM", "RUMAH", "KERJA", "BERMAIN", "BANTU", "JANGAN", "KAPAN", "KEREN"]
    st.markdown('''
    <div class="info-card">
        <div class="info-card-title"><span>💬</span> Words</div>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('<div class="tags-grid">' + ''.join([f'<span class="tag">{k}</span>' for k in kata]) + '</div>', unsafe_allow_html=True)

with c3:
    st.markdown('''
    <div class="info-card">
        <div class="info-card-title"><span>📥</span> Export</div>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; color:#64748b; margin-bottom:0.8rem;'>Download detection history as CSV file</p>", unsafe_allow_html=True)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("Download Log", f, "bisindo_log.csv", "text/csv", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown('''
<div class="app-footer">
    <strong>BISINDO Detection</strong> • Built with YOLOv11 • Universitas Negeri Semarang
</div>
''', unsafe_allow_html=True)