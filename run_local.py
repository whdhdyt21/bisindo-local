import cv2
import time
from ultralytics import YOLO

# =============================
# LOAD MODEL
# =============================
model = YOLO("best.pt")

# =============================
# CAMERA SETUP
# =============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam tidak terdeteksi")
    exit()

print("✅ Webcam aktif | Tekan 'Q' untuk keluar")

# =============================
# REALTIME LOOP
# =============================
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame")
        break

    # YOLO inference
    results = model(frame, conf=0.3, imgsz=640)

    # Plot bounding box
    annotated_frame = results[0].plot()

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Tampilkan FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Tampilkan window
    cv2.imshow("BISINDO Detection - Realtime", annotated_frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =============================
# CLEANUP
# =============================
cap.release()
cv2.destroyAllWindows()
