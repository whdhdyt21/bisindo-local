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
# CSS - CLEAN DARK THEME
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    :root {
        --bg: #0c1015;
        --surface: #14191f;
        --surface-2: #1a2028;
        --border: #242d38;
        --accent: #f0b429;
        --text: #f1f5f9;
        --muted: #64748b;
        --success: #10b981;
    }
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; box-sizing: border-box; }
    .stApp { background: var(--bg); }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1rem 1.5rem !important; max-width: 1000px !important; }
    
    /* ===== HEADER ===== */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 1.2rem;
        background: var(--surface);
        border-radius: 10px;
        border: 1px solid var(--border);
        margin-bottom: 0.8rem;
    }
    .logo { display: flex; align-items: center; gap: 10px; }
    .logo-icon {
        width: 38px; height: 38px;
        background: rgba(240,180,41,0.12);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem;
    }
    .logo h1 { font-size: 1.1rem; font-weight: 700; color: var(--text); margin: 0; }
    .logo p { font-size: 0.65rem; color: var(--muted); margin: 0; }
    .stats { display: flex; gap: 1.2rem; }
    .stat { text-align: center; }
    .stat-val { font-size: 1rem; font-weight: 700; color: var(--accent); }
    .stat-lbl { font-size: 0.55rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* ===== TOOLBAR ===== */
    .toolbar {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.7rem 1rem;
        background: var(--surface);
        border-radius: 10px;
        border: 1px solid var(--border);
        margin-bottom: 0.8rem;
    }
    .tool-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .tool-label span { color: var(--accent); }
    .divider {
        width: 1px;
        height: 24px;
        background: var(--border);
        margin: 0 0.3rem;
    }
    
    /* ===== MAIN CONTENT ===== */
    .main-area {
        background: var(--surface);
        border-radius: 10px;
        border: 1px solid var(--border);
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    /* ===== IMAGES ===== */
    .img-box {
        background: var(--bg);
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .img-head {
        padding: 0.4rem 0.7rem;
        font-size: 0.65rem;
        font-weight: 500;
        color: var(--muted);
        border-bottom: 1px solid var(--border);
        background: var(--surface-2);
    }
    
    /* ===== RESULT ===== */
    .result {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.6rem 0.9rem;
        background: var(--surface-2);
        border-radius: 8px;
        margin-top: 0.8rem;
    }
    .result-lbl { font-size: 0.7rem; color: var(--muted); }
    .badge {
        background: linear-gradient(135deg, var(--accent), #d49a1a);
        color: #0c1015;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .badge-c {
        background: rgba(0,0,0,0.12);
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 0.65rem;
    }
    .no-res { color: var(--muted); font-size: 0.75rem; }
    
    /* ===== EMPTY ===== */
    .empty {
        text-align: center;
        padding: 2.5rem 1.5rem;
        color: var(--muted);
    }
    .empty-icon {
        width: 48px; height: 48px;
        background: var(--surface-2);
        border-radius: 50%;
        display: inline-flex;
        align-items: center; justify-content: center;
        font-size: 1.3rem;
        margin-bottom: 0.6rem;
        border: 1px solid var(--border);
    }
    .empty h3 { font-size: 0.85rem; color: var(--text); margin: 0 0 0.15rem 0; font-weight: 600; }
    .empty p { font-size: 0.7rem; margin: 0; }
    
    /* ===== BOTTOM CARDS ===== */
    .bottom-grid {
        display: grid;
        grid-template-columns: 1fr 1.5fr 1fr;
        gap: 0.6rem;
    }
    .info-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
    }
    .info-title {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .info-title span { color: var(--accent); }
    .tags { display: flex; flex-wrap: wrap; gap: 3px; }
    .tag {
        background: var(--bg);
        color: var(--muted);
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.55rem;
        border: 1px solid var(--border);
    }
    
    /* ===== FOOTER ===== */
    .foot {
        text-align: center;
        padding: 0.8rem;
        color: var(--muted);
        font-size: 0.6rem;
    }
    .foot b { color: var(--accent); }
    
    /* ===== STREAMLIT OVERRIDES ===== */
    
    /* Hide all labels */
    .stSelectbox label, .stNumberInput label, .stFileUploader label { display: none !important; }
    
    /* Select box (for confidence) */
    div[data-baseweb="select"] {
        font-size: 0.75rem !important;
    }
    div[data-baseweb="select"] > div {
        background: var(--bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        min-height: 36px !important;
        padding: 0 0.5rem !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: var(--accent) !important;
    }
    div[data-baseweb="popover"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }
    div[role="listbox"] {
        background: var(--surface) !important;
    }
    div[role="option"] {
        background: var(--surface) !important;
        color: var(--text) !important;
        font-size: 0.75rem !important;
    }
    div[role="option"]:hover {
        background: var(--surface-2) !important;
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: var(--bg) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.8rem !important;
    }
    .stFileUploader > div > div:hover { 
        border-color: var(--accent) !important; 
    }
    section[data-testid="stFileUploadDropzone"] {
        padding: 0.3rem !important;
    }
    section[data-testid="stFileUploadDropzone"] span { 
        font-size: 0.65rem !important; 
        color: var(--muted) !important; 
    }
    section[data-testid="stFileUploadDropzone"] small { 
        font-size: 0.55rem !important; 
        color: var(--muted) !important;
    }
    button[data-testid="baseButton-secondary"] {
        background: var(--surface-2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        font-size: 0.6rem !important;
        padding: 0.25rem 0.6rem !important;
        border-radius: 4px !important;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: var(--success) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 5px !important;
        font-size: 0.65rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 0.5rem !important;
    }
    
    /* Remove extra padding */
    .stSelectbox, .stFileUploader {
        margin-bottom: 0 !important;
    }
    div[data-testid="column"] {
        padding: 0 0.3rem !important;
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
# TOOLBAR
# =====================================================
st.markdown('<div class="toolbar">', unsafe_allow_html=True)

t1, t2, t3, t4 = st.columns([1.3, 0.05, 1, 2])

with t1:
    st.markdown('<div class="tool-label"><span>📊</span> Detection</div>', unsafe_allow_html=True)

with t2:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with t3:
    conf_options = ["10%", "15%", "20%", "25%", "30%", "40%", "50%", "60%", "70%", "80%"]
    conf_select = st.selectbox("conf", conf_options, index=3, label_visibility="collapsed")
    conf = int(conf_select.replace("%", "")) / 100

with t4:
    file = st.file_uploader("u", ["jpg", "jpeg", "png"], label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# MAIN CONTENT
# =====================================================
st.markdown('<div class="main-area">', unsafe_allow_html=True)

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
        st.markdown('<div class="result"><span class="result-lbl">🎯 Result</span><span class="no-res">No detection — try lower threshold</span></div>', unsafe_allow_html=True)
else:
    st.markdown('''
    <div class="empty">
        <div class="empty-icon">📷</div>
        <h3>Upload Image</h3>
        <p>Select a BISINDO gesture image to start</p>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# BOTTOM INFO
# =====================================================
st.markdown('<div class="bottom-grid">', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1.5, 1])

with c1:
    st.markdown('<div class="info-card"><div class="info-title"><span>🔤</span> Letters</div><div class="tags">' + 
                ''.join([f'<span class="tag">{c}</span>' for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]) + 
                '</div></div>', unsafe_allow_html=True)

with c2:
    kata = ["AKU","KAMU","APA","DIA","AYAH","KAKAK","BAIK","MAAF","MARAH","SABAR","SEDIH","SENANG","SUKA","MINUM","RUMAH","KERJA","BERMAIN","BANTU","JANGAN","KAPAN","KEREN"]
    st.markdown('<div class="info-card"><div class="info-title"><span>💬</span> Words</div><div class="tags">' + 
                ''.join([f'<span class="tag">{k}</span>' for k in kata]) + 
                '</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="info-card"><div class="info-title"><span>📥</span> Export</div>', unsafe_allow_html=True)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("Download Log", f, "log.csv", "text/csv", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="foot"><b>BISINDO Detection</b> • YOLOv11 • Universitas Negeri Semarang</div>', unsafe_allow_html=True)