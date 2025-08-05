import cv2  
import numpy as np 
import threading 
import time 
from ultralytics import YOLO 
import os 
import logging

## working properly on actual apple
class DualCameraYOLOv8SingleObjectTracker:  # Define a class for tracking objects using two cameras and YOLOv8
    def __init__(self, camera1_id=1, camera2_id=0, model_name="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/train3/weights/best.pt",
                 conf_threshold=0.5, image_path="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/imageYellow.png", img_result="D:/openCV/apple_v2/apple-new-1-20250512T074115Z-1-001/apple-new-1/"):  # Initialize the class with default parameters
        self.camera1_id = camera1_id  
        self.camera2_id = camera2_id  
        self.camera1 = None 
        self.camera2 = None 
        self.frame1 = None 
        self.frame2 = None  
        self.running = False 
        self.image_path = image_path 
        # YOLOv8 parameters
        self.model_name = model_name  
        self.conf_threshold = conf_threshold  
        self.initialize_model() 
        
        # Tracking parameters
        self.tracking_initialized1 = False 
        self.tracking_initialized2 = False  
        self.tracking_box1 = None  
        self.tracking_box2 = None  
        self.detection_counter = 0 
        self.redetection_interval =  1 
        
        # Output directory for saving frames
        self.img_result = img_result  # Commented out: would store directory path for saving frames

    
    def initialize_model(self):
        """Initialize YOLOv8 model"""
        self.model = YOLO(self.model_name)  # Load the YOLO model from the specified path
    
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
        prev_time = time.time()  
        
        while self.running:
            ret1, self.frame1 = self.camera1.read()  
            ret2, self.frame2 = self.camera2.read() 
            
            if not ret1 or not ret2:  # Check if either frame reading failed
                print("Error: Failed to grab frames from one or both cameras")  
                break  # Exit the loop
            
            time.sleep(0.01)  
        
    def process_frames(self):
        """Process frames from both cameras with YOLOv8"""
        prev_time = time.time()
        
        while self.running and self.frame1 is not None and self.frame2 is not None:  
            # Calculate FPS
            new_time = time.time()  # Get the current time
            fps = 1 / (new_time - prev_time) if (new_time - prev_time) > 0 else 0  # Calculate frames per second
            prev_time = new_time  # Update previous time for next iteration
            
            # Create copies of frames to avoid modification during capture
            frame1_copy = self.frame1.copy() 
            frame2_copy = self.frame2.copy()  
            
            # Run YOLOv8 inference on the frames
            logging.getLogger("ultralytics").setLevel(logging.WARNING)  
            results1 = self.model(frame1_copy, conf=self.conf_threshold) 
            results2 = self.model(frame2_copy, conf=self.conf_threshold)  
            # Visualize the results on the frames
            annotated_frame1 = results1[0].plot()  
            annotated_frame2 = results2[0].plot() 

            box1 = results1[0].boxes 
            box2 = results2[0].boxes
            print(box1.cls)
            if len(box1.cls) == 0  or len(box2.cls) == 0:
                if len(box1.cls) > 0 and box1.cls[0] == 2:
                    print("Apple detected in rotten apple")
                if len(box2.cls) > 0 and box2.cls[0] == 2:
                    print("Apple detected in rotten apple")
                elif len(box1.cls) == 0 or len(box2.cls) == 0:
                    pass

            elif len(box1.cls) > 0 and len(box2.cls) > 0:
                if box1.cls[0] == 2 or box2.cls[0] == 2:
                    print("apple is rotten")
                elif box1.cls[0] != box2.cls[0]:
                    print("apple is fresh")
                elif box1.cls[0] == 1: 
                    print("apple is red")
                elif box1.cls[0] == 0:
                    print("apple is green")

# Do the same for frame 2 if needed

            # Add FPS information
            cv2.putText(annotated_frame1, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Add FPS text to frame1
            cv2.putText(annotated_frame2, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Add FPS text to frame2
            
            # Display the annotated frames
            cv2.imshow("Camera 1 - YOLOv8 Detection", annotated_frame1)  # Show the annotated frame1 in a window
            cv2.imshow("Camera 2 - YOLOv8 Detection", annotated_frame2)  # Show the annotated frame2 in a window

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if the 'q' key was pressed
                self.running = False 
                break  # Exit the loop
           
    def start(self):
        """Start the dual camera tracking system"""
        if not self.initialize_cameras(): 
            return 
        
        self.running = True  # Set running flag to True
        
        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_frames)  
        capture_thread.daemon = True 
        capture_thread.start()  
        
        # Allow time for cameras to initialize
        time.sleep(1.0) 
        
        # Start processing frames
        self.process_frames()  # Call the method to process frames (this runs in the main thread)
        
        # Cleanup
        self.running = False  # Set running flag to False
        capture_thread.join(timeout=1.0)  
        self.camera1.release()  
        self.camera2.release() 
        cv2.destroyAllWindows()  
        print("Tracking stopped.")  

    def calculate_apple_color_areas(self, frame, box):
        """
        Calculate the areas of green, red, and rotten parts within a detected apple
        
        Args:
            frame: The input image frame
            box: The bounding box of the detected apple
        
        Returns:
            Dictionary containing the areas of green, red, and rotten parts
        """
        # Extract the apple region using the bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
        apple_region = frame[y1:y2, x1:x2]
        
        if apple_region.size == 0:
            return {"green_area": 0, "red_area": 0, "rotten_area": 0}
        
        # Convert to HSV color space for better color segmentation
        hsv = cv2.cvtColor(apple_region, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for green, red, and brown (rotten) parts
        # These ranges may need adjustment based on lighting conditions
        yellowish_green_lower = np.array([15, 50, 50])
        yellowish_green_upper = np.array([65, 255, 255])

        
        # Red has two ranges in HSV (wraps around 0/180)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        # Brown/rotten parts
        rotten_lower = np.array([10, 50, 20])
        rotten_upper = np.array([30, 255, 150])
        
        # Create masks for each color
        green_mask = cv2.inRange(hsv, yellowish_green_lower, yellowish_green_upper)
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        rotten_mask = cv2.inRange(hsv, rotten_lower, rotten_upper)
        
        # Calculate areas (pixel counts)
        green_area = cv2.countNonZero(green_mask)
        red_area = cv2.countNonZero(red_mask)
        rotten_area = cv2.countNonZero(rotten_mask)
        
        # Calculate total area and percentages
        total_area = apple_region.shape[0] * apple_region.shape[1]
        green_percentage = (green_area / total_area) * 100 if total_area > 0 else 0
        red_percentage = (red_area / total_area) * 100 if total_area > 0 else 0
        rotten_percentage = (rotten_area / total_area) * 100 if total_area > 0 else 0
        
        return {
            "green_area": green_area,
            "red_area": red_area,
            "rotten_area": rotten_area,
            "green_percentage": green_percentage,
            "red_percentage": red_percentage,
            "rotten_percentage": rotten_percentage
        }



    def detect_on_image(self):
        """Detect objects in an image and return the annotated image"""
        # Load the YOLO model
        self.model = YOLO(self.model_name)
        
        # Read the image
        img = cv2.imread(self.image_path)
        cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
        if img is None:
            print(f"Error: Could not read image from {self.image_path}")
            return
        
        # Run YOLOv8 inference on the image
        logging.getLogger("ultralytics").setLevel(logging.WARNING)  # Set logging level for YOLOv8
        results = self.model(img)
        # Visualize the results on the image
        annotated_img = results[0].plot()
        box = results[0].boxes
        apple_result = int(box[0].cls[0]) # for value of apple_result (0: greenapple, 1: red apple, 2: rotten apple)

        if len(box.cls) > 0:
            # For each detected apple in frame 1
            for i in range(len(box.cls)):
                # Calculate color areas for this apple
                areas = self.calculate_apple_color_areas(img, box[i])
                
                # Display the results on the frame
                text_y_pos = 100 + i * 60  # Position text below FPS counter
                cv2.putText(annotated_img, f"Green: {areas['green_percentage']:.1f}%", 
                        (20, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_img, f"Red: {areas['red_percentage']:.1f}%", 
                        (20, text_y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(annotated_img, f"Rotten: {areas['rotten_percentage']:.1f}%", 
                        (20, text_y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 69, 19), 2)

                green_detected = areas['green_percentage']
                red_detected = areas['red_percentage']
                rotten_detected = areas['rotten_percentage']
                
                print(green_detected)
                print(red_detected)
                print(rotten_detected)
                if green_detected > 50: 
                    print(f"Green: {areas['green_percentage']:.1f}%")
                    print("Apple is green")
                if red_detected > 50:
                    
                    print(f"Red: {areas['red_percentage']:.1f}%")
                    print("Apple is red")
                if rotten_detected > 15:
                    print(f"Rotten: {areas['rotten_percentage']:.1f}%")
                    print("Apple is rotten")
            

        # Display the annotated image
        cv2.imshow("YOLOv8 Detection", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__": 
    use_cam = True 
    if use_cam:
        tracker = DualCameraYOLOv8SingleObjectTracker( 
            camera1_id=0, 
            camera2_id=1,  
            model_name="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/train3/weights/best.pt",  # Set path to the YOLO model
            conf_threshold=0.5,
              # Set confidence threshold to 0.5
       
        )
        tracker.start()  # Start the tracking system
        
    else:
        tracker = DualCameraYOLOv8SingleObjectTracker(  # Create an instance of the tracker class
        camera1_id=0, 
        camera2_id=1, 
        model_name="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/train3/weights/best.pt",  
        conf_threshold=0.5,
            # Set confidence threshold to 0.5
  
        image_path="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/imageYellow.png"
    )
        tracker.detect_on_image()




