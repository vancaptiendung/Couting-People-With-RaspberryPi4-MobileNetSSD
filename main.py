import cv2
import numpy as np
import sys
import threading
import time
from collections import deque

# =====================================================================
# 1. CẤU HÌNH AI (MOBILENET-SSD) VÀ HIỆU NĂNG
# =====================================================================
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

print("[INFO] Đang tải mô hình AI MobileNet-SSD...")
try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
except Exception as e:
    print("[ERROR] Không tìm thấy file mô hình! Vui lòng tải file .prototxt và .caffemodel")
    sys.exit(1)

CONFIDENCE_THRESHOLD = 0.5 

# ---- GIỚI HẠN FPS ----
TARGET_FPS = 6  # 6 khung hình/giây giúp Pi chạy siêu nhẹ mà vẫn đếm tốt

# =====================================================================
# 2. CẤU HÌNH VẠCH CHÉO VÀ BIẾN ĐẾM
# =====================================================================
L1_PT1 = (313, 204)
L1_PT2 = (524, 4)
L2_PT1 = (201, 481)
L2_PT2 = (575, -2)

TOTAL_IN = 0
TOTAL_OUT = 0
PEOPLE_IN_ROOM = 0

MAX_DISAPPEARED = 30     

trackable_objects = {}
next_object_id = 0
WINDOW_NAME = "He thong dem nguoi AI (MobileNet-SSD)"

# =====================================================================
# 3. HÀM BẮT SỰ KIỆN CHUỘT (ĐIỀU CHỈNH SỐ LƯỢNG)
# =====================================================================
def adjust_counters(event, x, y, flags, param):
    global TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM
    if event == cv2.EVENT_LBUTTONDOWN:
        if 400 <= y <= 420:
            if 120 <= x <= 150: TOTAL_IN = max(0, TOTAL_IN - 1)
            elif 160 <= x <= 190: TOTAL_IN += 1
        elif 430 <= y <= 450:
            if 120 <= x <= 150: TOTAL_OUT = max(0, TOTAL_OUT - 1)
            elif 160 <= x <= 190: TOTAL_OUT += 1
        elif 460 <= y <= 480:
            if 120 <= x <= 150: PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1)
            elif 160 <= x <= 190: PEOPLE_IN_ROOM += 1

# =====================================================================
# 4. HÀM TOÁN HỌC XÁC ĐỊNH VÙNG
# =====================================================================
def get_position(cx, cy, pt1, pt2):
    cross_product = (pt2[0] - pt1[0]) * (cy - pt1[1]) - (pt2[1] - pt1[1]) * (cx - pt1[0])
    return "BELOW" if cross_product > 0 else "ABOVE"

def get_zone(cx, cy):
    min_x = min(L1_PT1[0], L1_PT2[0], L2_PT1[0], L2_PT2[0]) - 20
    max_x = max(L1_PT1[0], L1_PT2[0], L2_PT1[0], L2_PT2[0]) + 20
    
    if cx < min_x or cx > max_x: return "IGNORE"

    pos_line1 = get_position(cx, cy, L1_PT1, L1_PT2)
    pos_line2 = get_position(cx, cy, L2_PT1, L2_PT2)

    if pos_line1 == "ABOVE": return "OUTSIDE"
    elif pos_line2 == "BELOW": return "INSIDE"
    else: return "MIDDLE"

# =====================================================================
# 5. KHỞI TẠO CAMERA (HỖ TRỢ LIBCAMERIFY)
# =====================================================================
class FrameGrabber:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30) # Luồng đọc ảnh phần cứng vẫn để 30fps cho mượt
        threading.Thread(target=self._reader, daemon=True).start()

    def isOpened(self): return self.cap.isOpened()
    
    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        time.sleep(0.05)
        self.cap.release()

vs = FrameGrabber(0)
if not vs.isOpened():
    print("[ERROR] Không thể kết nối camera. Hãy kiểm tra lại lệnh libcamerify.")
    sys.exit(1)

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, adjust_counters)

print(f"[INFO] Hệ thống đếm người AI bắt đầu chạy (Giới hạn: {TARGET_FPS} FPS)...")

# =====================================================================
# VÒNG LẶP XỬ LÝ CHÍNH
# =====================================================================
prev_time = 0

while True:
    frame = vs.read()
    if frame is None:
        time.sleep(0.01)
        continue

    # --- BỘ LỌC GIỚI HẠN FPS XỬ LÝ AI ---
    current_time = time.time()
    if (current_time - prev_time) < (1.0 / TARGET_FPS):
        time.sleep(0.005) # Ngủ 5ms nhả CPU
        continue
    
    prev_time = current_time 

    (h, w) = frame.shape[:2]
    current_centroids = []

    # --- CHẠY AI MOBILENET-SSD ---
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            
            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            current_centroids.append((cX, cY, startX, startY, endX, endY))

    # --- LOGIC TRACKING TỐI ƯU CHO FPS THẤP ---
    updated_trackable_objects = dict(trackable_objects)
    seen_ids = set()

    for (cX, cY, startX, startY, endX, endY) in current_centroids:
        matched_id = None
        # ĐÃ TĂNG lên 180 để người đi bộ nhanh không bị rớt ID do tụt FPS
        min_distance = 180  

        for obj_id, (old_cX, old_cY, zone_history, disappeared) in trackable_objects.items():
            d = np.hypot(cX - old_cX, cY - old_cY)
            if d < min_distance:
                min_distance = d
                matched_id = obj_id

        if matched_id is None:
            matched_id = next_object_id
            next_object_id += 1
            zone_history = deque([get_zone(cX, cY)], maxlen=10) 
            disappeared = 0
        else:
            old_cX, old_cY, zone_history, disappeared = trackable_objects[matched_id]
            current_zone = get_zone(cX, cY)
            if len(zone_history) == 0 or zone_history[-1] != current_zone:
                zone_history.append(current_zone)
            disappeared = 0

        compressed = [z for z in zone_history if z != "IGNORE"]
        final_compressed = []
        for z in compressed:
            if not final_compressed or final_compressed[-1] != z:
                final_compressed.append(z)

        if "OUTSIDE" in final_compressed and "INSIDE" in final_compressed:
            idx_out = final_compressed.index("OUTSIDE")
            idx_in = final_compressed.index("INSIDE")
            
            if idx_out < idx_in:
                TOTAL_IN += 1
                PEOPLE_IN_ROOM += 1
                zone_history = deque(["INSIDE"], maxlen=10)
            elif idx_in < idx_out:
                TOTAL_OUT += 1
                PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1) 
                zone_history = deque(["OUTSIDE"], maxlen=10)

        updated_trackable_objects[matched_id] = (cX, cY, zone_history, disappeared)
        seen_ids.add(matched_id)

        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 150, 0), 2)
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        current_zone_str = get_zone(cX, cY)
        text = f"ID {matched_id} | {current_zone_str} | {confidence*100:.0f}%"
        cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)

    for obj_id in list(updated_trackable_objects.keys()):
        if obj_id not in seen_ids:
            cX, cY, zone_history, disappeared = updated_trackable_objects[obj_id]
            disappeared += 1
            if disappeared > MAX_DISAPPEARED:
                del updated_trackable_objects[obj_id]
            else:
                updated_trackable_objects[obj_id] = (cX, cY, zone_history, disappeared)

    trackable_objects = updated_trackable_objects

    # --- VẼ GIAO DIỆN ---
    cv2.line(frame, L1_PT1, L1_PT2, (0, 0, 255), 2)
    cv2.line(frame, L2_PT1, L2_PT2, (0, 0, 255), 2)
    
    cv2.putText(frame, "NGOAI", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "TRONG", (w - 150, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.rectangle(frame, (5, 385), (205, 485), (0, 0, 0), -1)
    cv2.putText(frame, f"Vao: {TOTAL_IN}", (10, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Ra: {TOTAL_OUT}", (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Trong: {PEOPLE_IN_ROOM}", (10, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for y_btn in [400, 430, 460]:
        cv2.rectangle(frame, (120, y_btn), (150, y_btn + 20), (80, 80, 80), -1)
        cv2.putText(frame, "-", (130, y_btn + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (160, y_btn), (190, y_btn + 20), (80, 80, 80), -1)
        cv2.putText(frame, "+", (168, y_btn + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()