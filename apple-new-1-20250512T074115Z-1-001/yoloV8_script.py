import cv2
import numpy as np
from ultralytics import YOLO
import time

def run_yolo_detection():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use the nano model, you can also use 's', 'm', 'l', or 'x' for different sizes
    
    # Open the video capture (0 for webcam, or provide a video file path)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Get the video frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up FPS calculation
    prev_time = 0
    new_time = 0
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Calculate FPS
        new_time = time.time()
        fps = 1 / (new_time - prev_time) if (new_time - prev_time) > 0 else 0
        prev_time = new_time
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Add FPS information
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

def detect_on_image(image_path):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Run YOLOv8 inference on the image
    results = model(img)
    
    # Visualize the results on the image
    annotated_img = results[0].plot()
    
    # Display the annotated image
    cv2.imshow("YOLOv8 Detection", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose whether to run on webcam/video or on a single image
    use_webcam = True  # Set to False to run on a single image
    
    if use_webcam:
        run_yolo_detection()
    else:
        
        # Provide the path to your image
        image_path = "img2.png"
        detect_on_image(image_path)
