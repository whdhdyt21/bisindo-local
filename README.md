# 🤟 BISINDO Gesture Detection - Local Demo

Aplikasi deteksi Bahasa Isyarat Indonesia (BISINDO) menggunakan YOLOv11l.

## 📋 Requirements

- Python 3.8+
- Webcam (untuk real-time detection)

## 🚀 Cara Menjalankan

### Langkah 1: Install Dependencies

```bash
pip install ultralytics gradio opencv-python
```

### Langkah 2: Pastikan File Model Ada

Pastikan file `best.pt` ada di folder yang sama dengan `app.py`:

```
bisindo-local/
├── app.py
├── best.pt      <-- Model YOLOv11l
├── run.bat
└── README.md
```

### Langkah 3: Jalankan Aplikasi

**Windows:**
- Double-click `run.bat`
- Atau buka PowerShell: `python app.py`

**Mac/Linux:**
```bash
python app.py
```

### Langkah 4: Buka Browser

Aplikasi akan otomatis membuka browser di: http://localhost:7860

## 🎯 Fitur

| Tab | Fungsi |
|-----|--------|
| 🎥 Real-time Webcam | Deteksi gesture secara real-time dari webcam |
| 📷 Upload Gambar | Upload gambar untuk deteksi gesture |

## 📊 Gesture yang Dapat Dideteksi (47 kelas)

**Huruf:** A-Z

**Kata:** AKU, APA, AYAH, BAIK, BANTU, BERMAIN, DIA, JANGAN, KAKAK, KAMU, KAPAN, KEREN, KERJA, MAAF, MARAH, MINUM, RUMAH, SABAR, SEDIH, SENANG, SUKA

## 📈 Performa Model

| Metric | Value |
|--------|-------|
| mAP@0.5 | 94.46% |
| FPS | 39.04 |
| Model Size | 48.89 MB |

## 🔧 Troubleshooting

**Gesture tidak terdeteksi:**
- Turunkan Confidence Threshold ke 0.1-0.2
- Pastikan pencahayaan cukup
- Posisikan tangan di tengah frame

**Webcam tidak muncul:**
- Pastikan browser mengizinkan akses kamera
- Coba refresh halaman

---

**Universitas Negeri Semarang** - Tugas Akhir/Skripsi
