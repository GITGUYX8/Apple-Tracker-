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
        self.img_result = img_result  
    
    def initialize_model(self):
        """Initialize YOLOv8 model"""
        self.model = YOLO(self.model_name) 
        self.class_names = self.model.names  
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
        
        return True  # Return True to indicate successful initialization
    
    def capture_frames(self):
        """Continuously capture frames from both cameras"""
        prev_time = time.time() 
        
        while self.running:  
            ret1, self.frame1 = self.camera1.read() 
            ret2, self.frame2 = self.camera2.read()  
            
            if not ret1 or not ret2:  
                print("Error: Failed to grab frames from one or both cameras")  
                break 
            
            time.sleep(0.01) 

    def get_apple_type_from_model(self,class_id, confidence):
        # Get class name from model
        class_name = self.class_names[int(class_id)]
        
        # Map class names to display format and colors
        class_mapping = {
            'green apple': ("GREEN", (0, 255, 0)),
            'red apple': ("RED", (0, 0, 255)),
            'rotten apple': ("ROTTEN", (0, 165, 255)),
            'apple': ("APPLE", (255, 255, 0)),  
        }
        
        # Get the mapped type and color, or use the original class name
        if class_name.lower() in class_mapping:
            apple_type, type_color = class_mapping[class_name.lower()]
        else:
            # If class name not in mapping, use it as is
            apple_type = class_name.upper()
            type_color = (255, 255, 255)  # White color for unknown types
        
        return apple_type, confidence * 100, type_color
    def display_apple_results(self, frame, areas, apple_index, camera_name, class_id, model_confidence):
        """
        Display apple classification results based on model prediction
        
        Args:
            frame: The frame to draw on
            areas: Dictionary containing color percentages (for display only)
            apple_index: Index of the apple (for multiple apples)
            camera_name: Name of the camera for identification
            class_id: Class ID from model prediction
            model_confidence: Confidence score from model
        """
        # Get apple classification from model
        apple_type, confidence, type_color = self.get_apple_type_from_model(class_id, model_confidence)
        
        # Create a results panel
        panel_height = 200
        panel_width = 320
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 10 + (apple_index * (panel_height + 10))
        
        # Draw semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw panel border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Display camera and apple info
        y_offset = panel_y + 25
        cv2.putText(frame, f"{camera_name} - Apple #{apple_index + 1}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Display apple type with colored background (from model)
        y_offset += 30
        type_bg_color = type_color
        cv2.rectangle(frame, (panel_x + 10, y_offset - 20), 
                     (panel_x + panel_width - 10, y_offset + 5), 
                     type_bg_color, -1)
        cv2.putText(frame, f"TYPE: {apple_type}", 
                   (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Display model confidence
        y_offset += 35
        cv2.putText(frame, f"Model Confidence: {confidence:.1f}%", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # Display color breakdown (for reference only)
        y_offset += 25
        cv2.putText(frame, "Color Analysis (Reference):", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.45, (200, 200, 200), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Green: {areas['green_percentage']:.1f}%", 
                   (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (0, 255, 0), 1)
        
        y_offset += 15
        cv2.putText(frame, f"Red: {areas['red_percentage']:.1f}%", 
                   (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (0, 0, 255), 1)
        
        y_offset += 15
        cv2.putText(frame, f"Rotten: {areas['rotten_percentage']:.1f}%", 
                   (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (0, 165, 255), 1)
        
        # Print to console for logging
        print(f"\n{camera_name} - Apple #{apple_index + 1}:")
        print(f"  Model Prediction: {apple_type} ({confidence:.1f}%)")
        print(f"  Class ID: {class_id}")
        print(f"  Color Analysis (Reference):")
        print(f"    Green: {areas['green_percentage']:.1f}%")
        print(f"    Red: {areas['red_percentage']:.1f}%")
        print(f"    Rotten: {areas['rotten_percentage']:.1f}%")
        
        return apple_type
       
    def process_frames(self):
        """Process frames from both cameras with YOLOv8"""
        prev_time = time.time()  # Store the current time for FPS calculation
        
        while self.running and self.frame1 is not None and self.frame2 is not None: 
            # Calculate FPS
            new_time = time.time() 
            fps = 1 / (new_time - prev_time) if (new_time - prev_time) > 0 else 0  
            prev_time = new_time 
            
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
            
            # Process Camera 1 results
            if len(box1.cls) > 0:
                for i in range(len(box1.cls)):
                    areas = self.calculate_apple_color_areas(frame1_copy, box1[i])
                    class_id = box1.cls[i].item()
                    model_confidence = box1.conf[i].item()
                    apple_type = self.display_apple_results(annotated_frame1, areas, i, "Camera 1", class_id, model_confidence)

            # Process Camera 2 results
            if len(box2.cls) > 0:
                for i in range(len(box2.cls)):
                    areas = self.calculate_apple_color_areas(frame2_copy, box2[i])
                    class_id = box2.cls[i].item()
                    model_confidence = box2.conf[i].item()
                    apple_type = self.display_apple_results(annotated_frame2, areas, i, "Camera 2", class_id, model_confidence)

            # Add FPS information
            cv2.putText(annotated_frame1, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Add FPS text to frame1
            cv2.putText(annotated_frame2, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Add FPS text to frame2
            
            # Display the annotated frames
            cv2.imshow("Camera 1 - YOLOv8 Detection", annotated_frame1) 
            cv2.imshow("Camera 2 - YOLOv8 Detection", annotated_frame2)  

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if the 'q' key was pressed
                self.running = False 
                break 
           
    def start(self):
        """Start the dual camera tracking system"""
        if not self.initialize_cameras():  
            return  
        
        self.running = True  # Set running flag to True
        
        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_frames)  # Create a thread for capturing frames
        capture_thread.daemon = True  # Set thread as daemon so it exits when the main program exits
        capture_thread.start()  # Start the capture thread
        
        # Allow time for cameras to initialize
        time.sleep(1.0)  # Wait for 1 second to ensure cameras are ready
        
        # Start processing frames
        self.process_frames()  # Call the method to process frames (this runs in the main thread)
        
        # Cleanup
        self.running = False  
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
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        apple_region = frame[y1:y2, x1:x2]
        
        if apple_region.size == 0:
            return {"green_area": 0, "red_area": 0, "rotten_area": 0}
        
        # Convert to HSV color space for better color segmentation
        hsv = cv2.cvtColor(apple_region, cv2.COLOR_BGR2HSV)
      
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
      
        
        
        if len(box.cls) > 0:
            print("\n" + "="*50)
            print("APPLE DETECTION RESULTS")
            print("="*50)
            
            # For each detected apple
            for i in range(len(box.cls)):
                # Get model predictions
                class_id = box.cls[i].item()
                model_confidence = box.conf[i].item()
                
                # Calculate color areas for reference
                areas = self.calculate_apple_color_areas(img, box[i])
                
                # Get apple classification from model
                apple_type, confidence, type_color = self.get_apple_type_from_model(class_id, model_confidence)
                
                # Display results in proper section on image
                self.display_apple_results(annotated_img, areas, i, "Image Analysis", class_id, model_confidence)
                
                print(f"\nApple #{i+1} Analysis:")
                print(f"  Model Classification: {apple_type}")
                print(f"  Model Confidence: {confidence:.1f}%")
                print(f"  Class ID: {class_id}")
                print(f"  Class Name: {self.class_names[int(class_id)]}")
                print(f"  Color Analysis (Reference):")
                print(f"    - Green: {areas['green_percentage']:.1f}%")
                print(f"    - Red: {areas['red_percentage']:.1f}%")
                print(f"    - Rotten: {areas['rotten_percentage']:.1f}%")
                print(f"  Final Decision: ", end="")
                
                if apple_type == "ROTTEN":
                    print("  REJECT - Apple is rotten (Model Prediction)")
                elif apple_type in ["GREEN", "RED"]:
                    print("ACCEPT - Apple is good quality (Model Prediction)")
                else:
                    print(f" REVIEW - {apple_type} (Model Prediction)")
            
            print("\n" + "="*50)
        else:
            print("No apples detected in the image.")
            cv2.putText(annotated_img, "No apples detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Display the annotated image
        cv2.imshow("YOLOv8 Detection", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":  # Check if this script is being run directly
    use_cam = True  
    if use_cam:
        tracker = DualCameraYOLOv8SingleObjectTracker(  # Create an instance of the tracker class
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
        model_name="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/train3/weights/best.pt",  # Set path to the YOLO model
        conf_threshold=0.5,
          
        # output_dir="./frames"  # Commented out: would set the output directory for saving frames
        image_path="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/imageYellow.png"
    )
        tracker.detect_on_image()




