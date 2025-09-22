# 🔫 Real-Time Weapon Detection System using YOLOv8 and OpenCV

## 📝 Abstract

The proliferation of weapons in public spaces poses significant challenges to public safety and security. In response to this concern, this project presents a **real-time weapon detection system with dynamic data augmentation**, leveraging **YOLOv8**, **OpenCV**, and **Non-Maximum Suppression (NMS)**.

Key features include:
- **Multi-camera setup** for real-time detection.
- **Stochastic augmentation** for robust model generalization across varying lighting and orientations.
- **OpenCV integration** to capture live feeds and process them efficiently.
- **YOLOv8-based object detection** with high performance.
- **NMS filtering** for precise bounding boxes based on confidence score.

The model achieved **mAP@50 of 0.993** and **accuracy of 9%8 at 0.568**, making it highly effective for **smart surveillance and real-time monitoring**.

---

## 🚀 Features

- 🎥 Real-time detection from multiple camera feeds.
- 🔁 Dynamic stochastic augmentation (flipping, brightness, affine rotation).
- 🎯 High detection accuracy and confidence-based alerting.
- 🖼️ Color-coded bounding boxes with class labels and scores.
- 🧠 Fine-tuned YOLOv8 model on custom weapon dataset.
- 🧹 Non-Maximum Suppression (NMS) to reduce false positives.

---

## 🧠 Methodology

1. **Frame Capture**  
   OpenCV captures real-time video frames from multiple webcam sources.

2. **Stochastic Augmentation**  
   Detected frames are dynamically augmented to simulate varying visual conditions:
   - Horizontal flipping  
   - Brightness alteration  
   - Affine rotation  

3. **Weapon Detection & Tracking**  
   YOLOv8 is used to:
   - Detect objects
   - Extract class ID, confidence, bounding box
   - Visualize with OpenCV and label by class and score

4. **Non-Maximum Suppression (NMS)**  
   To eliminate duplicate bounding boxes:

![comparison_cam1_frame13](https://github.com/user-attachments/assets/aaf11075-168f-4ce3-a7af-14c268ba5760)


![cam1_weapon_20250623_115954_493007](https://github.com/user-attachments/assets/08dc9f5e-115c-41e9-a92f-b33f206c3675)
![res](https://github.com/user-attachments/assets/0d9613bd-9433-4eba-b8d0-2a4b056a80fc)
