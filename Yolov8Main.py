import numpy as np
import cv2
from ultralytics import YOLO
import random
import os
from datetime import datetime

# Create output folder if it doesn't exist
os.makedirs("output1", exist_ok=True)

def predictions(img, Detect_obj, Box_colours, class_list):
    weapon_detected = False  # flag to trigger saving
    DP = Detect_obj[0].numpy()

    if len(DP) != 0:
        for i in range(len(Detect_obj[0])):
            boxes = Detect_obj[0].boxes
            box = boxes[i]
            clsID = int(box.cls.numpy()[0])
            conf = float(box.conf.numpy()[0])
            bb = box.xyxy.numpy()[0]

            # Draw bounding box
            cv2.rectangle(
                img,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                Box_colours[int(clsID)],
                3,
            )

            # Add label
            cv2.putText(
                img,
                class_list[clsID] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2,
            )

            # Check if it's a weapon
            if class_list[clsID].lower() in ['rifle']:
                weapon_detected = True

    return weapon_detected

# Load class names
with open("utils/coco.txt") as f:
    class_list = f.read().split("\n")

# Generate random colors for boxes
Box_colours = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Load model
model = YOLO(r"3339model\weights\best.pt", "v8")

# Start video capture for two cameras
caps = [cv2.VideoCapture(i) for i in range(2)]
for cap in caps:
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 100)

if any(not cap.isOpened() for cap in caps):
    print("Unable to open one or more webcams. Exiting...")
else:
    print("Starting your webcams...")

while True:
    imgs = [cap.read()[1] for cap in caps]
    if any(img is None for img in imgs):
        print("Unable to load frames. Exiting...")
        break

    for i, img in enumerate(imgs):
        Detect_obj = model.track(source=[img], conf=0.50, save=False)
        weapon_found = predictions(img, Detect_obj, Box_colours, class_list)

        # Save if weapon found
        if weapon_found:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = f"output1/cam{i}_weapon_{timestamp}.jpg"

            # Optional: apply noise reduction before saving
            # img_filtered = cv2.GaussianBlur(img, (5, 5), 0)
            # img_filtered = cv2.medianBlur(img_filtered, 5)

            cv2.imwrite(save_path, img)
            print(f"Weapon detected — saved frame to {save_path}")

        cv2.imshow(f'CAM-{i}', img)

    if cv2.waitKey(1) == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
