import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
import os

class DualCameraYOLOv8SingleObjectTracker:
    def __init__(self, camera1_id=0, camera2_id=1, model_name="D:/openCV/apple_v2/apple-new-1-20250512T074115Z-1-001/apple-new-1/train3/weights/best.pt", 
                 conf_threshold=0.8, target_class=0, output_dir="./frames"):
        self.camera1_id = camera1_id
        self.camera2_id = camera2_id
        self.camera1 = None
        self.camera2 = None
        self.frame1 = None
        self.frame2 = None
        self.running = False
        
        # YOLOv8 parameters
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.target_class = target_class
        self.initialize_model()
        
        # Tracking parameters
        self.tracking_initialized1 = False        # Flag to track if tracking is initialized for camera 1
        self.tracking_initialized2 = False        # Flag to track if tracking is initialized for camera 2
        self.tracking_box1 = None                # Stores the bounding box coordinates for tracked object in camera 1
        self.tracking_box2 = None                # Stores the bounding box coordinates for tracked object in camera 2
        self.detection_counter = 0               # Counter to keep track of detection iterations        
        self.redetection_interval = 1  # Re-detect every frame (can be increased for performance)
        
        # Output directory for saving frames
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def initialize_model(self):
        """Initialize YOLOv8 model"""
        # Load YOLOv8 model
        self.model = YOLO(self.model_name)
        
    def initialize_cameras(self):
        """Initialize both camera streams"""
        self.camera1 = cv2.VideoCapture(self.camera1_id)
        self.camera2 = cv2.VideoCapture(self.camera2_id)
        
        # Check if cameras opened successfully
        if not self.camera1.isOpened() or not self.camera2.isOpened():
            print("Error: Could not open one or both cameras.")
            return False
            
        # Set camera properties if needed
        self.camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        return True
        
    def capture_frames(self):
        """Continuously capture frames from both cameras"""
        while self.running:
            ret1, self.frame1 = self.camera1.read()
            ret2, self.frame2 = self.camera2.read()
            
            if not ret1 or not ret2:
                print("Error: Failed to grab frames from one or both cameras")
                break
                
            time.sleep(0.01)  # Small delay to prevent high CPU usage
    
    def detect_objects(self, frame):
        """Detect objects in a frame using YOLOv8"""
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.conf_threshold)
        
        # Process results
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                # Get confidence and class
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = self.model.names[cls]
                
                # Only add objects of the target class
                if cls == 1 : 
                    detected_objects.append((x, y, w, h, "rotten", conf))
                elif cls == 0 :
                    detected_objects.append((x, y, w, h, "fresh", conf))
                
        return detected_objects
    
     
    def process_frames(self):
        """Process frames from both cameras and track a single object"""
        while self.running:
            if self.frame1 is None or self.frame2 is None:
                time.sleep(0.1)
                continue
                
            # Make copies to avoid threading issues
            frame1_copy = self.frame1.copy()
            frame2_copy = self.frame2.copy()
            
            # Increment detection counter
            self.detection_counter += 1
            
            # Re-detect periodically or if tracking is not initialized
            if (self.detection_counter % self.redetection_interval == 0) or \
            (not self.tracking_initialized1) or (not self.tracking_initialized2):
                
                # Detect objects in both frames
                objects1 = self.detect_objects(frame1_copy)
                objects2 = self.detect_objects(frame2_copy)
                
                # Initialize or update trackers if objects are detected
                if objects1:
                    # Sort by confidence and get the highest confidence detection
                    objects1.sort(key=lambda x: x[5], reverse=True)
                    x, y, w, h, cls1, conf1 = objects1[0]
                    self.tracking_box1 = (x, y, w, h)
                    self.tracking_initialized1 = True
                    
                if objects2:
                    # Sort by confidence and get the highest confidence detection
                    objects2.sort(key=lambda x: x[5], reverse=True)
                    x, y, w, h, cls2, conf2 = objects2[0]
                    self.tracking_box2 = (x, y, w, h)
                    self.tracking_initialized2 = True
            
            # Draw tracking results
            result1 = frame1_copy.copy()
            result2 = frame2_copy.copy()
            
            if self.tracking_initialized1 and self.tracking_box1:
                x, y, w, h = self.tracking_box1
                cv2.rectangle(result1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{cls1}: {conf1}"
                cv2.putText(result1, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if self.tracking_initialized2 and self.tracking_box2:
                x, y, w, h = self.tracking_box2
                cv2.rectangle(result2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{cls2}: {conf2}"
                cv2.putText(result2, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display results instead of saving them
            try:
                cv2.imshow("Camera 1", result1)
                cv2.imshow("Camera 2", result2)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            except cv2.error:
                print("Error: Could not display frames. Your OpenCV installation may not support GUI.")
                print("Falling back to saving frames...")
                # Fallback to saving frames if display fails
                os.makedirs(self.output_dir, exist_ok=True)
                frame_count = getattr(self, 'frame_count', 0) + 1
                self.frame_count = frame_count
                cv2.imwrite(os.path.join(self.output_dir, f"camera1_frame_{frame_count:04d}.jpg"), result1)
                cv2.imwrite(os.path.join(self.output_dir, f"camera2_frame_{frame_count:04d}.jpg"), result2)
                print(f"Saved frame {frame_count}")
                
                # Check for keyboard interrupt less frequently
                if frame_count % 10 == 0:
                    try:
                        if input("Press Enter to stop...") == '':
                            self.running = False
                    except:
                        pass

    def start(self):
        """Start the dual camera tracking system"""
        if not self.initialize_cameras():
            return
            
        self.running = True
        
        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.daemon = True
        capture_thread.start()
        
        print(f"Starting tracking of {self.target_class} objects...")
        print("Press 'q' in the camera window to stop")
        
        # Process frames in the main thread
        self.process_frames()
        
        # Cleanup
        self.running = False
        capture_thread.join(timeout=1.0)
        self.camera1.release()
        self.camera2.release()
        cv2.destroyAllWindows()
        print("Tracking stopped.")

    
if __name__ == "__main__":
    # Create and start the dual camera tracker
    tracker = DualCameraYOLOv8SingleObjectTracker(
        camera1_id=0, 
        camera2_id=1,
        model_name="D:/openCV/apple_v2/apple-new-1-20250512T074115Z-1-001/apple-new-1/train3/weights/best.pt",  # Use yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), or yolov8l.pt (large)
        conf_threshold=0.8,
        target_class=0,  # Change this to the class you want to track
        output_dir="./frames"  # Directory to save output frames
    )
    tracker.start()
