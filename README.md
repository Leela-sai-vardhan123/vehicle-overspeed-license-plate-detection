# vehicle-overspeed-license-plate-detection
Real-time vehicle speed detection and license plate recognition system using YOLOv8, EasyOCR, and OpenCV. Developed during NTPC Internship 2025.

# Vehicle Overspeeding and License Plate Detection 🚗📸

This project was developed as part of my **NTPC Internship (Summer 2025)**.  
It implements a **real-time Vehicle Speed Detection and License Plate Recognition (LPR)** system using **YOLOv8, EasyOCR, and OpenCV**.

---

## 🔍 Problem Statement
Over-speeding is a major cause of road accidents. Traditional speed monitoring is:
- Labor-intensive
- Error-prone
- Not scalable for real-time traffic monitoring

We need an **automated, low-cost, real-time system** that:
- Detects vehicles from video
- Calculates speed
- Identifies over-speeding vehicles
- Extracts license plate numbers
- Logs violations for enforcement

---

## 🛠️ Tech Stack
- **YOLOv8** → Vehicle & License Plate Detection  
- **EasyOCR** → License Plate Text Recognition  
- **OpenCV** → Video processing & speed estimation  
- **Python (NumPy, Pandas)** → Data handling & logging  

---

## 📂 Project Structure
├── notebooks/ # Jupyter notebooks (training/testing)
├── scripts/ # Python scripts for real-time detection
├── utils/ # Helper functions
├── outputs/ # Sample output videos & violation logs
├── requirements.txt # Python dependencies
├── report/ # Internship report (PDF)
└── README.md # Project documentation





---

## ⚙️ Features
✅ Detect vehicles in real-time using YOLOv8  
✅ Estimate vehicle speed using frame timestamps & distance calibration  
✅ Detect and crop license plates from vehicles  
✅ Recognize license plate numbers with EasyOCR (with post-processing for accuracy)  
✅ Log over-speeding violations with:
- Vehicle ID
- Speed (km/h)
- License Plate Number
- Date & Time  
✅ Save results to **CSV/Excel** for further analysis  

---

## 📊 Experimental Results
- Achieved **~100% license plate detection** on high-quality images  
- Speed estimation error within **±2–3 km/h** after calibration  
- Example log format:
vehicle_id,license_plate,speed_kph,overspeed,timestamp
359,AP64DMV,35.8,True,2025-06-30 15:42:12



## 🚀 How to Run
1. Clone the repo 
   git clone https://github.com/<your-username>/vehicle-overspeed-detection.git
   cd vehicle-overspeed-detection
Install dependencies
pip install -r requirements.txt

Run detection on sample video
python scripts/detect_speed_and_plate.py --video sample.mp4 --weights yolov8_best.pt

Output:
Annotated video with speed & license plate overlays
violations.csv file with logged over-speeding events
<img width="917" height="652" alt="Screenshot 2025-09-24 141221" src="https://github.com/user-attachments/assets/3cd60b3a-5482-428c-868d-3418bd0e4a5f" />

<img width="782" height="508" alt="image" src="https://github.com/user-attachments/assets/e7633be2-9867-4ad7-a401-5821f493824f" />


📌 Future Improvements
   Better performance in low-light & weather conditions

   Edge deployment on Jetson Nano/Raspberry Pi

   Multilingual OCR for Indian plates across regions

   Integration with traffic police databases for challan automation

Helmet detection & vehicle type classification

📄 License
       This project is released under the MIT License.

🙌 Acknowledgments
    NTPC Internship for mentorship & guidance

   Ultralytics YOLOv8

   EasyOCR

   SORT Tracking
