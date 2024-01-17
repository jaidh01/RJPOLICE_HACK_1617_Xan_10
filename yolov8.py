import cv2
from ultralytics import YOLO



# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model.track(frame)
    cv2.imshow('YOLO Object Detection', results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam capture and close the window
cap.release()
cv2.destroyAllWindows()