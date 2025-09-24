from ultralytics import YOLO
import cv2
import easyocr
import ssl
import os
from datetime import datetime

# -------------------------------
# CONFIGURATION
# -------------------------------
ssl._create_default_https_context = ssl._create_unverified_context
VIDEO_PATH = 'ntpc.mp4'
VEHICLE_MODEL = 'yolov8n.pt'
PLATE_MODEL = '/Users/jayadeep/Documents/NTPC-INT/models/best.pt'
SPEED_LIMIT = 10  # km/h
REAL_DISTANCE_METERS = 3  # actual distance between ENTRY and EXIT lines in meters
ENTRY_LINE_Y = 330
EXIT_LINE_Y = 800
PIXEL_TO_METER = REAL_DISTANCE_METERS / (EXIT_LINE_Y - ENTRY_LINE_Y)  # calibration factor

# -------------------------------
# INIT
# -------------------------------
vehicle_detector = YOLO(VEHICLE_MODEL)
plate_detector = YOLO(PLATE_MODEL)
ocr_reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

# CSV setup
os.makedirs("logs", exist_ok=True)
CSV_PATH = "logs/overspeeding_log.csv"
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w") as f:
        f.write("ID,Type,License Plate,Speed (km/h),Timestamp\n")

# Object dictionary
vehicles = {}

# -------------------------------
# MAIN LOOP
# -------------------------------
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = vehicle_detector.track(frame, persist=True, classes=[2, 5, 7], verbose=False)  # car, bus, truck

    if results[0].boxes.id is None:
        continue

    boxes = results[0].boxes
    for i, box in enumerate(boxes):
        id = int(box.id.item())
        cls = int(box.cls.item())
        label = vehicle_detector.model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw bounding box & ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{id}-{label}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Initialize vehicle state
        if id not in vehicles:
            vehicles[id] = {
                'type': label,
                'entry_y': None,
                'entry_frame': None,
                'exit_y': None,
                'exit_frame': None,
                'logged': False,
                'box': (x1, y1, x2, y2)
            }

        # Update current box
        vehicles[id]['box'] = (x1, y1, x2, y2)

        # Track entry
        if vehicles[id]['entry_y'] is None and cy >= ENTRY_LINE_Y:
            vehicles[id]['entry_y'] = cy
            vehicles[id]['entry_frame'] = frame_idx

        # Track exit
        elif vehicles[id]['entry_y'] and vehicles[id]['exit_y'] is None and cy >= EXIT_LINE_Y:
            vehicles[id]['exit_y'] = cy
            vehicles[id]['exit_frame'] = frame_idx

            delta_pixels = vehicles[id]['exit_y'] - vehicles[id]['entry_y']
            delta_frames = vehicles[id]['exit_frame'] - vehicles[id]['entry_frame']

            if delta_frames > 0:
                distance_m = delta_pixels * PIXEL_TO_METER
                time_sec = delta_frames / fps
                speed_mps = distance_m / time_sec
                speed_kmph = speed_mps * 3.6

                if speed_kmph > SPEED_LIMIT and not vehicles[id]['logged']:
                    # OCR plate
                    x1, y1, x2, y2 = vehicles[id]['box']
                    vehicle_crop = frame[y1:y2, x1:x2]
                    plate_text = "Unknown"
                    plate_results = plate_detector(vehicle_crop)

                    for p in plate_results[0].boxes:
                        px1, py1, px2, py2 = map(int, p.xyxy[0])
                        plate_img = vehicle_crop[py1:py2, px1:px2]
                        ocr = ocr_reader.readtext(plate_img)
                        if ocr:
                            plate_text = ocr[0][1]
                            break

                    # Log to CSV
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(CSV_PATH, "a") as f:
                        f.write(f"{id},{label},{plate_text},{speed_kmph:.1f},{timestamp}\n")
                    print(f"✅ Logged: {id}-{label} | {plate_text} | {speed_kmph:.1f} km/h | {timestamp}")
                    vehicles[id]['logged'] = True

    # Draw lines
    cv2.line(frame, (0, ENTRY_LINE_Y), (frame.shape[1], ENTRY_LINE_Y), (255, 255, 0), 2)
    cv2.line(frame, (0, EXIT_LINE_Y), (frame.shape[1], EXIT_LINE_Y), (0, 0, 255), 2)

    # Display
    cv2.imshow("Overspeeding Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n📄 CSV saved to: {CSV_PATH}")
