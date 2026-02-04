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
# CSS
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    :root {
        --bg: #0a0e13;
        --surface: #131920;
        --border: #1f2937;
        --accent: #f0b429;
        --text: #f1f5f9;
        --muted: #6b7280;
        --success: #10b981;
    }
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background: var(--bg); }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1.2rem 2rem !important; max-width: 1100px !important; }
    
    /* ===== HEADER ===== */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.5rem;
        background: var(--surface);
        border-radius: 12px;
        border: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    .logo { display: flex; align-items: center; gap: 12px; }
    .logo-icon {
        width: 42px; height: 42px;
        background: rgba(240,180,41,0.15);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
    }
    .logo h1 { font-size: 1.2rem; font-weight: 700; color: var(--text); margin: 0; }
    .logo p { font-size: 0.7rem; color: var(--muted); margin: 0; }
    .stats { display: flex; gap: 1.5rem; }
    .stat { text-align: center; }
    .stat-val { font-size: 1.1rem; font-weight: 700; color: var(--accent); }
    .stat-lbl { font-size: 0.6rem; color: var(--muted); text-transform: uppercase; }
    
    /* ===== MAIN CARD ===== */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    
    /* ===== TOOLBAR ===== */
    .toolbar {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .tool-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .tool-title span { color: var(--accent); }
    
    /* ===== SLIDER ===== */
    .slider-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        background: var(--bg);
        padding: 8px 14px;
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    .slider-label {
        font-size: 0.7rem;
        color: var(--muted);
        white-space: nowrap;
    }
    .slider-val {
        background: var(--accent);
        color: var(--bg);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        min-width: 40px;
        text-align: center;
    }
    
    /* ===== IMAGES ===== */
    .img-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .img-box {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }
    .img-head {
        padding: 0.5rem 0.8rem;
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--muted);
        border-bottom: 1px solid var(--border);
    }
    
    /* ===== RESULT ===== */
    .result {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.8rem 1rem;
        background: var(--bg);
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    .result-lbl { font-size: 0.75rem; color: var(--muted); }
    .badge {
        background: linear-gradient(135deg, var(--accent), #d49a1a);
        color: var(--bg);
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-c {
        background: rgba(0,0,0,0.15);
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.7rem;
    }
    .no-res { color: var(--muted); font-size: 0.8rem; }
    
    /* ===== EMPTY ===== */
    .empty {
        text-align: center;
        padding: 3rem;
        color: var(--muted);
    }
    .empty-icon {
        width: 56px; height: 56px;
        background: var(--bg);
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 0.8rem;
        border: 1px solid var(--border);
    }
    .empty h3 { font-size: 0.95rem; color: var(--text); margin: 0 0 0.2rem 0; }
    .empty p { font-size: 0.8rem; margin: 0; }
    
    /* ===== BOTTOM ===== */
    .info-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    .info-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.6rem;
    }
    .info-title span { color: var(--accent); }
    .tags { display: flex; flex-wrap: wrap; gap: 4px; }
    .tag {
        background: var(--bg);
        color: var(--muted);
        padding: 3px 7px;
        border-radius: 4px;
        font-size: 0.6rem;
        border: 1px solid var(--border);
    }
    
    /* ===== FOOTER ===== */
    .foot {
        text-align: center;
        padding: 1rem;
        color: var(--muted);
        font-size: 0.65rem;
    }
    .foot b { color: var(--accent); }
    
    /* ===== STREAMLIT ===== */
    .stSlider { display: none !important; }
    .stFileUploader label { display: none !important; }
    .stFileUploader > div > div {
        background: var(--bg) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 8px !important;
        padding: 0.6rem !important;
    }
    .stFileUploader > div > div:hover { border-color: var(--accent) !important; }
    section[data-testid="stFileUploadDropzone"] span { font-size: 0.7rem !important; color: var(--muted) !important; }
    section[data-testid="stFileUploadDropzone"] small { font-size: 0.6rem !important; }
    button[data-testid="baseButton-secondary"] {
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        font-size: 0.65rem !important;
        padding: 0.3rem 0.8rem !important;
    }
    .stDownloadButton button {
        background: var(--success) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    .stSelectbox label, .stNumberInput label { display: none !important; }
    div[data-baseweb="select"] > div {
        background: var(--bg) !important;
        border-color: var(--border) !important;
        font-size: 0.75rem !important;
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
    <div class="logo">
        <div class="logo-icon">🤟</div>
        <div>
            <h1>BISINDO Detection</h1>
            <p>Bahasa Isyarat Indonesia • YOLOv11</p>
        </div>
    </div>
    <div class="stats">
        <div class="stat"><div class="stat-val">94.5%</div><div class="stat-lbl">Accuracy</div></div>
        <div class="stat"><div class="stat-val">39</div><div class="stat-lbl">FPS</div></div>
        <div class="stat"><div class="stat-val">47</div><div class="stat-lbl">Classes</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MAIN CARD
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

# Toolbar - semua dalam 1 baris
t1, t2, t3 = st.columns([1.5, 2, 3])

with t1:
    st.markdown('<div class="tool-title"><span>📊</span> Detection</div>', unsafe_allow_html=True)

with t2:
    conf = st.select_slider(
        "Confidence",
        options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        value=0.25,
        format_func=lambda x: f"{int(x*100)}%",
        label_visibility="collapsed"
    )

with t3:
    file = st.file_uploader("u", ["jpg", "jpeg", "png"], label_visibility="collapsed")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# Detection
if file:
    img = Image.open(file).convert("RGB")
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    result = model(img_np, conf=conf, imgsz=640)
    img_out = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="img-box"><div class="img-head">🖼️ Input</div></div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    with c2:
        st.markdown('<div class="img-box"><div class="img-head">✅ Output</div></div>', unsafe_allow_html=True)
        st.image(img_out, use_container_width=True)
    
    labels = []
    if result[0].boxes and len(result[0].boxes) > 0:
        for box in result[0].boxes:
            lbl = model.names[int(box.cls[0])]
            scr = float(box.conf[0])
            labels.append((lbl, scr))
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file.name, lbl, f"{scr:.4f}"])
    
    if labels:
        badges = ' '.join([f'<span class="badge">{l}<span class="badge-c">{s:.0%}</span></span>' for l, s in labels])
        st.markdown(f'<div class="result"><span class="result-lbl">🎯 Detected</span>{badges}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result"><span class="result-lbl">🎯 Result</span><span class="no-res">No detection — lower threshold</span></div>', unsafe_allow_html=True)
else:
    st.markdown('''
    <div class="empty">
        <div class="empty-icon">📷</div>
        <h3>Upload Image</h3>
        <p>Select a BISINDO gesture image</p>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# BOTTOM
# =====================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="info-card"><div class="info-title"><span>🔤</span> Letters</div><div class="tags">' + ''.join([f'<span class="tag">{c}</span>' for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]) + '</div></div>', unsafe_allow_html=True)

with c2:
    kata = ["AKU","KAMU","APA","DIA","AYAH","KAKAK","BAIK","MAAF","MARAH","SABAR","SEDIH","SENANG","SUKA","MINUM","RUMAH","KERJA","BERMAIN","BANTU","JANGAN","KAPAN","KEREN"]
    st.markdown('<div class="info-card"><div class="info-title"><span>💬</span> Words</div><div class="tags">' + ''.join([f'<span class="tag">{k}</span>' for k in kata]) + '</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="info-card"><div class="info-title"><span>📥</span> Export</div>', unsafe_allow_html=True)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("Download Log", f, "log.csv", "text/csv", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="foot"><b>BISINDO Detection</b> • YOLOv11 • Universitas Negeri Semarang</div>', unsafe_allow_html=True)