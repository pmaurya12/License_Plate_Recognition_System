import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict, deque

# Load models
model = YOLO("models/license_plate_best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# Regex: 2 letters + 2 numbers + 3 letters
plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")


# -------------------- OCR CORRECTION --------------------
def correct_plate_format(ocr_text):
    mapping_num_to_alpha = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
    mapping_alpha_to_num = {"O": "0", "I": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8", "Q": "0", "D": "0", "L": "1"}

    ocr_text = ocr_text.upper().replace(" ", "")

    if len(ocr_text) != 7:
        return ""

    corrected = []

    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4:  # alphabet positions
            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        else:  # numeric positions
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""

    return "".join(corrected)


# -------------------- OCR RECOGNITION --------------------
def recognize_plate(plate_crop):
    if plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plate_resized = cv2.resize(
        thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )

    try:
        ocr_result = reader.readtext(
            plate_resized,
            detail=0,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        if len(ocr_result) > 0:
            candidate = correct_plate_format(ocr_result[0])
            if candidate and plate_pattern.match(candidate):
                return candidate
    except:
        pass

    return ""


# -------------------- STABILIZATION --------------------
plate_history = defaultdict(lambda: deque(maxlen=10))
plate_final = {}


def get_box_id(x1, y1, x2, y2):
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"


def get_stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)

        most_common = max(
            set(plate_history[box_id]),
            key=plate_history[box_id].count
        )

        plate_final[box_id] = most_common

    return plate_final.get(box_id, "")


# -------------------- MAIN PROCESS FUNCTION --------------------
def process_video(input_video, output_video):

    cap = cv2.VideoCapture(input_video)

    # ✅ Fix FPS issue
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ⚠️ This may or may not fully encode H.264 depending on system
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    CONF_THRESH = 0.3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                conf = box.conf.item()

                if conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                plate_crop = frame[y1:y2, x1:x2]

                # OCR
                text = recognize_plate(plate_crop)

                # Stabilization
                box_id = get_box_id(x1, y1, x2, y2)
                stable_text = get_stable_plate(box_id, text)

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # Overlay zoomed plate
                if plate_crop.size > 0:
                    overlay_h, overlay_w = 150, 400
                    plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))

                    oy1 = max(0, y1 - overlay_h - 40)
                    ox1 = x1
                    oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

                    if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                        frame[oy1:oy2, ox1:ox2] = plate_resized

                    # Draw text
                    if stable_text:
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        out.write(frame)

    cap.release()
    out.release()

def generate_live_frames():
    cap = cv2.VideoCapture(0)

    CONF_THRESH = 0.3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                conf = box.conf.item()
                if conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                plate_crop = frame[y1:y2, x1:x2]

                text = recognize_plate(plate_crop)

                box_id = get_box_id(x1, y1, x2, y2)
                stable_text = get_stable_plate(box_id, text)

                # 🔴 SAME RED BOX (unchanged)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)

                if stable_text:
                    cv2.putText(frame, stable_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0,0,0), 6)
                    cv2.putText(frame, stable_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255,255,255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()