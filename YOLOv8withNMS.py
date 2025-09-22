import numpy as np
import cv2
from ultralytics import YOLO
import random
import os
from datetime import datetime

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

def non_maximum_suppression(boxes, scores, threshold):
    bboxes = [(int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])) for box in boxes]
    scores = np.array(scores)
    indices = cv2.dnn.NMSBoxes(bboxes, scores.tolist(), threshold, threshold - 0.1)
    filtered_boxes = [boxes[i] for i in indices.flatten()]
    filtered_scores = [scores[i] for i in indices.flatten()]
    return filtered_boxes, filtered_scores


def predictions(img, Detect_obj, Box_colours, class_list):
    weapon_detected = False
    if len(Detect_obj[0]) != 0:
        raw_boxes = []
        raw_scores = []
        raw_cls_ids = []
        for i in range(len(Detect_obj[0])):
            box = Detect_obj[0].boxes[i]
            clsID = int(box.cls.numpy()[0])
            conf = float(box.conf.numpy()[0])
            bb = box.xyxy.numpy()[0]
            raw_boxes.append(bb)
            raw_scores.append(conf)
            raw_cls_ids.append(clsID)

            # Check for weapon classes
            if class_list[clsID].lower() in ['knife', 'gun', 'pistol', 'rifle']:
                weapon_detected = True

            # RED boxes - raw detections
            cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 2)
            # cv2.putText(img, class_list[clsID] + " " + str(round(conf, 2)), (int(bb[0]), int(bb[1]) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            label = f"{class_list[clsID]} {round(conf, 2)}"
            position = (int(bb[0]), int(bb[1]) - 10)

            # Outline (black shadow for contrast)
            cv2.putText(img, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)

            # Main text (white foreground)
            cv2.putText(img, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)


        filtered_boxes, filtered_scores = non_maximum_suppression(raw_boxes, raw_scores, threshold=0.5)

        for box, score in zip(filtered_boxes, filtered_scores):
    # Draw rectangle (unchanged)
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        
            # Draw bold white text with better visibility
            text = f"NMS {round(score, 2)}"
            position = (int(box[0]), int(box[1]) + 20)
        
            # Optional: black outline for bold contrast
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # Shadow/outline
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)  # Main white text


    return weapon_detected


# Load class list
with open("utils/coco.txt") as f:
    class_list = f.read().split("\n")

# Generate random box colors
Box_colours = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_list))]

# Load YOLO model
model = YOLO(r"3339model\weights\best.pt", "v8")

# Open video capture devices
caps = [cv2.VideoCapture(i) for i in range(2)]

# Set video capture properties
for cap in caps:
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 100)

if any(not cap.isOpened() for cap in caps):
    print("Unable to open one or more webcams. Exiting...")
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
else:
    print("Starting your webcams...")
    while True:
        imgs = [cap.read()[1] for cap in caps]
        if any(img is None for img in imgs):
            print("Unable to load frames. Exiting...")
            break

        for i, img in enumerate(imgs):
            Detect_obj = model.track(source=[img], conf=0.65, save=False)
            weapon_found = predictions(img, Detect_obj, Box_colours, class_list)

            if weapon_found:
                # Apply noise filtering to the image before saving
                denoised = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"output/cam{i}_weapon_{timestamp}.jpg"
                cv2.imwrite(filename, denoised)
                print(f"Weapon detected and saved to {filename}")

            cv2.imshow(f'CAM-{i}', img)

        if cv2.waitKey(1) == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
