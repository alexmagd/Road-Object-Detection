import os
import cv2
from ultralytics import YOLO

# Load model with proper path handling
model_path = r'Thesis\RoadObjectDetection\runs\detect\train2\weights\best.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")

model = YOLO(model_path)  # Load the model

threshold = 0.5

# Specify the desired width and height
width = 640
height = 480

def preprocess_frame(frame, width, height):
    # Resize the frame
    frame_resized = cv2.resize(frame, (width, height))
    return frame_resized

# Capture video from the camera
cap = cv2.VideoCapture(0)  # Change camera index if needed
if not cap.isOpened():
    raise IOError("Cannot access the camera")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    frame_preprocessed = preprocess_frame(frame, width, height)

    # Perform object detection
    results = model(frame_preprocessed)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
