from ultralytics import YOLO
from pathlib import Path
import shutil

# ✅ 1) Train your YOLO model
model = YOLO('yolov8n.pt')

results = model.train(
    data='Indian number plate.v2i.yolov8/data.yaml',
    epochs=50,
    imgsz=640
)

# ✅ 2) After training, find the latest run folder
runs_dir = Path("runs/detect")
latest_run = max(runs_dir.glob("train*"), key=lambda p: p.stat().st_mtime)

print(f"📂 Latest run folder: {latest_run}")

# ✅ 3) The best.pt path inside that run
default_best = latest_run / "weights/best.pt"

# ✅ 4) Where you want to save it in your project
destination = Path("models/detect_license.pt")
destination.parent.mkdir(parents=True, exist_ok=True)

# ✅ 5) Copy it safely
if default_best.exists():
    shutil.copy(default_best, destination)
    print(f"✅ Your trained model is saved to: {destination.resolve()}")
else:
    print(f"❌ ERROR: Could not find {default_best}")
