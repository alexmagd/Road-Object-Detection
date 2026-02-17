import os
import cv2
from ultralytics import YOLO

# Load model with proper path handling
model_path = r'Thesis\RoadObjectDetection\runs\detect\train2\weights\best.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")

model = YOLO(model_path)  # Load the model

threshold = 0.5

# Specify the desired width and height for resizing
width = 640
height = 480

def preprocess_frame(frame, width, height):
    # Resize the frame
    frame_resized = cv2.resize(frame, (width, height))
    return frame_resized

# Input video path (change to your actual video file path)
input_video_path = r'Thesis\object_detection1.mp4'  # Path to the input video file
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    raise IOError(f"Cannot open video file {input_video_path}")

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Set up the VideoWriter to save the processed video
output_video_path = r'output_object_detection1.mp4'  # Path for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    frame_preprocessed = preprocess_frame(frame, width, height)

    # Perform object detection
    results = model(frame_preprocessed)[0]

    # Draw bounding boxes and labels
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Write the processed frame to the output video
    out.write(frame)

    # Optionally display the frame (for debugging or visual feedback)
    cv2.imshow('Frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
