# 🤟 BISINDO Detection

**Sistem Deteksi Bahasa Isyarat Indonesia Berbasis YOLO v11**

Aplikasi ini merupakan sistem **deteksi Bahasa Isyarat Indonesia (BISINDO)** berbasis *deep learning* menggunakan **YOLO v11**.
Sistem mendukung dua skenario penggunaan:

1. **Realtime webcam (local / offline)**
2. **Deteksi citra statis (online / web)**

Proyek ini dikembangkan untuk keperluan **penelitian, demonstrasi sistem, dan tugas akhir/skripsi**.

---

## 🌐 Demo Online

Aplikasi versi web dapat diakses melalui:

🔗 **[https://bisindo-detection.streamlit.app/](https://bisindo-detection.streamlit.app/)**

> ⚠️ **Catatan Penting:**
> Versi online **tidak mendukung webcam realtime** karena keterbatasan lingkungan server Streamlit Cloud.
> Untuk versi online, silakan gunakan **mode unggah gambar**.

---

## ✨ Fitur Utama

* 🖼️ **Deteksi gesture dari unggahan gambar (Online)**
* 🎥 **Deteksi realtime melalui webcam (Local)**
* 🏷️ **Menampilkan label kelas dan confidence**
* 🎯 **Pengaturan confidence threshold**
* ⚡ **FPS realtime (mode local)**
* 🌐 **Antarmuka web modern (Streamlit)**
* 💻 **Mode native OpenCV untuk realtime cepat**

---

## 🗂️ Struktur Proyek

```
bisindo-local/
├── app.py              # Aplikasi web (Streamlit)
├── run_local.py        # Realtime webcam (local, OpenCV)
├── best.pt             # Model YOLO v11
├── requirements.txt    # Dependency Python
├── packages.txt        # Dependency sistem (Streamlit Cloud)
├── assets/
│   └── logo.png
└── README.md
```

---

## 📋 Kebutuhan Sistem

### 🔹 Local

* Python 3.8 – 3.11
* Webcam
* OS: Windows / Linux / macOS

### 🔹 Online

* Browser modern
* Koneksi internet

---

## ⚙️ Instalasi Dependency (Local)

```bash
pip install -r requirements.txt
```

Isi `requirements.txt` (local):

```txt
streamlit
ultralytics
opencv-python
pillow
numpy
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1️⃣ Realtime Webcam (LOCAL – DISARANKAN)

Mode ini digunakan untuk:

* Demo langsung
* Pengujian performa realtime
* FPS tinggi

```bash
python run_local.py
```

Kontrol:

* Webcam otomatis aktif
* Tekan **Q** untuk keluar

---

### 2️⃣ Versi Web (ONLINE / DEMO SISTEM)

```bash
streamlit run app.py
```

Akses melalui browser:

```
http://localhost:8501
```

Atau gunakan versi online:
🔗 [https://bisindo-detection.streamlit.app/](https://bisindo-detection.streamlit.app/)

---

## 🎯 Kelas Gesture yang Didukung

Total **47 kelas**, terdiri dari:

### 🔤 Huruf

A – Z

### 🧾 Kata

AKU, APA, AYAH, BAIK, BANTU, BERMAIN, DIA, JANGAN,
KAKAK, KAMU, KAPAN, KEREN, KERJA, MAAF, MARAH,
MINUM, RUMAH, SABAR, SEDIH, SENANG, SUKA

---

## 📊 Performa Model (Hasil Pelatihan)

| Metrik       | Nilai      |
| ------------ | ---------- |
| mAP@0.5      | ±94%       |
| FPS (Local)  | ±30–40 FPS |
| Ukuran Model | ±50 MB     |
| Framework    | YOLO v11   |

> Catatan: Performa dapat berbeda tergantung perangkat dan kondisi pencahayaan.

---

## 🛠️ Troubleshooting

### ❓ Gesture tidak terdeteksi

* Turunkan confidence threshold (0.05 – 0.2)
* Pastikan pencahayaan cukup
* Gunakan pose sesuai data training
* Posisikan tangan di tengah frame

### ❓ Webcam tidak berfungsi di online

* Ini **bukan bug**
* Streamlit Cloud **tidak mendukung webcam realtime**
* Gunakan **mode unggah gambar**

### ❓ Error `cv2` saat deploy

* Pastikan menggunakan `opencv-python-headless`
* Pastikan file `packages.txt` tersedia

---

## 🎓 Konteks Akademik

Aplikasi ini dikembangkan sebagai bagian dari:

> **Tugas Akhir / Skripsi**
> Bidang: *Computer Vision & Deep Learning*
> Studi Kasus: *Pengenalan Bahasa Isyarat Indonesia (BISINDO)*

Pendekatan ini memisahkan:

* **Realtime inference (local)**
* **Demonstrasi sistem (online)**

untuk memastikan performa dan aksesibilitas tetap optimal.

---

## 🏫 Institusi

**Universitas Negeri Semarang (UNNES)**

---

## 📌 Catatan Akhir

* Realtime webcam ➜ **Local (`run_local.py`)**
* Demo & pengujian sistem ➜ **Online (Streamlit)**
* Arsitektur ini dipilih untuk menjaga **kestabilan, performa, dan kejelasan sistem**

---