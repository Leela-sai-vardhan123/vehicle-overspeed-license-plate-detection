import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ✅ 1) Paths
root_dir = "dataset"  # change if needed
images_dir = Path(root_dir) / "images"
labels_dir = Path(root_dir) / "labels"

# ✅ 2) Make YOLO folders
for split in ["train", "val"]:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

# ✅ 3) Collect all images and labels from vid-1, vid-2, vid-3
all_images = []
for vid_folder in ["vid-1", "vid-2", "vid-3"]:
    vid_path = Path(root_dir) / vid_folder
    images = list(vid_path.glob("*.jpg"))
    all_images.extend(images)

print(f"Total images found: {len(all_images)}")

# ✅ 4) Split into train/val (80% train, 20% val)
train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)

# ✅ 5) Move images & labels to new folders
def move_pairs(images, split):
    for img in images:
        label = img.with_suffix('.txt')
        # Copy image
        shutil.copy(img, images_dir / split / img.name)
        # Copy label
        if label.exists():
            shutil.copy(label, labels_dir / split / label.name)
        else:
            print(f"Warning: Label for {img.name} not found!")

move_pairs(train_imgs, "train")
move_pairs(val_imgs, "val")

print("✅ Dataset prepared in YOLO format!")

# ✅ 6) Create data.yaml
yaml_content = f"""
path: {root_dir}
train: images/train
val: images/val

nc: 1
names: ["license_plate"]
"""

with open(Path(root_dir) / "data.yaml", "w") as f:
    f.write(yaml_content.strip())

print("✅ data.yaml created!")
