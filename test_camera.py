"""
Quick manual check that OpenCV can open the webcam and read frames.
Run it, confirm a window pops up showing your live camera feed, press 'q' to quit.
"""

import cv2

CAMERA_INDEX = 0  # 0 is usually the built-in laptop webcam

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}")
        return

    print("Camera opened. Press 'q' in the window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        cv2.imshow("test_camera - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
