"""
Quick manual check that Ultralytics YOLO can load and detect a person
from the webcam feed. Draws boxes for class 0 (person) above a confidence
threshold. Press 'q' to quit.
"""

import cv2
from ultralytics import YOLO

CAMERA_INDEX = 0
MODEL_PATH = "yolov8n.pt"   # smallest/fastest YOLOv8 model, downloaded on first run
IMG_SIZE = 320              # smaller = faster, good enough for presence detection
CONFIDENCE = 0.5
PERSON_CLASS_ID = 0         # COCO dataset class id for "person"

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}")
        return

    print("Running YOLO on webcam feed. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        results = model(frame, imgsz=IMG_SIZE, conf=CONFIDENCE, classes=[PERSON_CLASS_ID], verbose=False)
        annotated = results[0].plot()  # draws boxes/labels for us

        cv2.imshow("test_yolo - press q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
