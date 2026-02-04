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
# DESIGN SYSTEM
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-base: #0a0e13;
        --bg-surface: #12171f;
        --bg-elevated: #1a2029;

        --accent: #f0b429;
        --accent-soft: rgba(240, 180, 41, 0.12);
        --success: #10b981;

        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #475569;

        --border: #1e293b;

        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
    }

    * { font-family: 'Plus Jakarta Sans', sans-serif; }

    .stApp { background: var(--bg-base); color: var(--text-primary); }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding: 1.5rem 2rem !important;
        max-width: 1200px !important;
    }

    /* HEADER */
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
        margin: 0;
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
    }

    /* MAIN CARD */
    .main-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        display: flex;
        gap: 8px;
    }

    /* CONTROLS */
    .control-group {
        background: var(--bg-base);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.6rem 1rem;
    }

    .control-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-bottom: 0.4rem;
    }

    .conf-badge {
        background: var(--accent);
        color: var(--bg-base);
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-align: center;
    }

    /* SLIDER FIX */
    .stSlider label { display: none !important; }

    div[data-baseweb="slider"] {
        background: var(--border) !important;
        height: 6px !important;
    }

    div[data-baseweb="slider"] > div:first-child {
        background: var(--accent) !important;
        height: 6px !important;
    }

    div[data-baseweb="slider"] div[role="slider"] {
        background: var(--accent) !important;
        border: 3px solid var(--bg-base) !important;
        width: 18px !important;
        height: 18px !important;
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
        <div class="metric"><div class="metric-value">94.5%</div><div class="metric-label">Accuracy</div></div>
        <div class="metric"><div class="metric-value">39</div><div class="metric-label">FPS</div></div>
        <div class="metric"><div class="metric-value">47</div><div class="metric-label">Classes</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MAIN CARD
# =====================================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📊 Detection</div>', unsafe_allow_html=True)

ctrl1, ctrl2 = st.columns([1.2, 1])

# ================= CONFIDENCE + PRESET =================
with ctrl1:
    if "conf_percent" not in st.session_state:
        st.session_state.conf_percent = 50

    preset_map = {"Low": 30, "Balanced": 50, "High": 70}

    st.markdown('<div class="control-group">', unsafe_allow_html=True)
    st.markdown('<div class="control-label">Confidence Threshold</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    for col, (name, val) in zip([p1, p2, p3], preset_map.items()):
        with col:
            if st.button(name, use_container_width=True):
                st.session_state.conf_percent = val

    s1, s2 = st.columns([6, 1])
    with s1:
        st.session_state.conf_percent = st.slider(
            "conf",
            10, 90,
            value=st.session_state.conf_percent,
            step=5,
            label_visibility="collapsed"
        )

    with s2:
        st.markdown(
            f"<div class='conf-badge'>{st.session_state.conf_percent}%</div>",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    conf = st.session_state.conf_percent / 100

# ================= UPLOAD =================
with ctrl2:
    st.markdown('<div class="control-label">Upload Image</div>', unsafe_allow_html=True)
    file = st.file_uploader("upload", ["jpg", "jpeg", "png"], label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# DETECTION
# =====================================================
if file:
    img = Image.open(file).convert("RGB")
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    result = model(img_np, conf=conf, imgsz=640)
    img_out = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)

    c1, c2 = st.columns(2)
    with c1: st.image(img, caption="Input", use_container_width=True)
    with c2: st.image(img_out, caption="Output", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
