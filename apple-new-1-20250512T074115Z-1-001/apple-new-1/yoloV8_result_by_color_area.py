import cv2  # Import OpenCV library for computer vision tasks
import numpy as np  # Import NumPy for numerical operations
import threading  # Import threading module for parallel execution
import time  # Import time module for timing operations
from ultralytics import YOLO  # Import YOLO model from ultralytics
import os  # Import os module for file and directory operations
import logging

## working properly on actual apple
class DualCameraYOLOv8SingleObjectTracker:  # Define a class for tracking objects using two cameras and YOLOv8
    def __init__(self, camera1_id=1, camera2_id=0, model_name="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/train3/weights/best.pt",
                 conf_threshold=0.5, image_path="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/imageYellow.png", img_result="D:/openCV/apple_v2/apple-new-1-20250512T074115Z-1-001/apple-new-1/"):  # Initialize the class with default parameters
        self.camera1_id = camera1_id  # Store the ID for the first camera
        self.camera2_id = camera2_id  # Store the ID for the second camera
        self.camera1 = None  # Initialize camera1 object as None
        self.camera2 = None  # Initialize camera2 object as None
        self.frame1 = None  # Initialize frame1 as None to store images from camera1
        self.frame2 = None  # Initialize frame2 as None to store images from camera2
        self.running = False  # Set running flag to False initially
        self.image_path = image_path  # Store the path to the image directory
        # YOLOv8 parameters
        self.model_name = model_name  # Store the path to the YOLOv8 model
        self.conf_threshold = conf_threshold  # Store the confidence threshold for detections
        self.initialize_model()  # Call method to initialize the YOLO model
        
        # Tracking parameters
        self.tracking_initialized1 = False  # Flag to check if tracking is initialized for camera1
        self.tracking_initialized2 = False  # Flag to check if tracking is initialized for camera2
        self.tracking_box1 = None  # Variable to store the tracking box for camera1
        self.tracking_box2 = None  # Variable to store the tracking box for camera2
        self.detection_counter = 0  # Counter for detection frames
        self.redetection_interval =  1 # Interval for re-detection
        
        # Output directory for saving frames
        self.img_result = img_result  # Commented out: would store directory path for saving frames
        # os.makedirs(self.output_dir, exist_ok=True)  # Commented out: would create the output directory if it doesn't exist
    
    def initialize_model(self):
        """Initialize YOLOv8 model"""
        self.model = YOLO(self.model_name)  # Load the YOLO model from the specified path
    
    def initialize_cameras(self):
        """Initialize both camera streams"""
        self.camera1 = cv2.VideoCapture(self.camera1_id)  # Initialize the first camera with its ID
        self.camera2 = cv2.VideoCapture(self.camera2_id)  # Initialize the second camera with its ID
        
        # Check if cameras opened successfully
        if not self.camera1.isOpened() or not self.camera2.isOpened():  # Check if either camera failed to open
            print("Error: Could not open one or both cameras.")  # Print error message
            return False  # Return False to indicate failure
        
        # Set camera properties if needed
        self.camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width of camera1 to 640 pixels
        self.camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height of camera1 to 480 pixels
        self.camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width of camera2 to 640 pixels
        self.camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height of camera2 to 480 pixels
        
        return True  # Return True to indicate successful initialization
    
    def capture_frames(self):
        """Continuously capture frames from both cameras"""
        prev_time = time.time()  # Store the current time for FPS calculation
        
        while self.running:  # Loop as long as the running flag is True
            ret1, self.frame1 = self.camera1.read()  # Read a frame from camera1
            ret2, self.frame2 = self.camera2.read()  # Read a frame from camera2
            
            if not ret1 or not ret2:  # Check if either frame reading failed
                print("Error: Failed to grab frames from one or both cameras")  # Print error message
                break  # Exit the loop
            
            time.sleep(0.01)  # Small delay to prevent high CPU usage

    def classify_apple_type(self, areas):
        """
        Classify apple type based on color percentages
        
        Args:
            areas: Dictionary containing color percentages
            
        Returns:
            tuple: (apple_type, confidence, color)
        """
        green_pct = areas['green_percentage']
        red_pct = areas['red_percentage']
        rotten_pct = areas['rotten_percentage']
        
        # Classification logic
        if rotten_pct > 15:
            return "ROTTEN", rotten_pct, (0, 165, 255)  # Orange color for rotten
        elif green_pct > 50:
            return "GREEN", green_pct, (0, 255, 0)  # Green color
        elif red_pct > 50:
            return "RED", red_pct, (0, 0, 255)  # Red color
        else:
            # Mixed or unclear classification
            dominant_color = max(green_pct, red_pct, rotten_pct)
            if dominant_color == green_pct:
                return "MIXED (Green)", green_pct, (0, 255, 128)
            elif dominant_color == red_pct:
                return "MIXED (Red)", red_pct, (128, 0, 255)
            else:
                return "UNCLEAR", dominant_color, (128, 128, 128)

    def display_apple_results(self, frame, areas, apple_index, camera_name):
        """
        Display apple classification results in a proper section
        
        Args:
            frame: The frame to draw on
            areas: Dictionary containing color percentages
            apple_index: Index of the apple (for multiple apples)
            camera_name: Name of the camera for identification
        """
        # Get apple classification
        apple_type, confidence, type_color = self.classify_apple_type(areas)
        
        # Create a results panel
        panel_height = 180
        panel_width = 300
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
        
        # Display apple type with colored background
        y_offset += 30
        type_bg_color = type_color
        cv2.rectangle(frame, (panel_x + 10, y_offset - 20), 
                     (panel_x + panel_width - 10, y_offset + 5), 
                     type_bg_color, -1)
        cv2.putText(frame, f"TYPE: {apple_type}", 
                   (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Display confidence
        y_offset += 35
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # Display color breakdown
        y_offset += 25
        cv2.putText(frame, "Color Analysis:", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
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
        print(f"  Type: {apple_type} ({confidence:.1f}%)")
        print(f"  Green: {areas['green_percentage']:.1f}%")
        print(f"  Red: {areas['red_percentage']:.1f}%")
        print(f"  Rotten: {areas['rotten_percentage']:.1f}%")
        
        return apple_type
        
    def process_frames(self):
        """Process frames from both cameras with YOLOv8"""
        prev_time = time.time()  # Store the current time for FPS calculation
        
        while self.running and self.frame1 is not None and self.frame2 is not None:  # Loop while running and frames exist
            # Calculate FPS
            new_time = time.time()  # Get the current time
            fps = 1 / (new_time - prev_time) if (new_time - prev_time) > 0 else 0  # Calculate frames per second
            prev_time = new_time  # Update previous time for next iteration
            
            # Create copies of frames to avoid modification during capture
            frame1_copy = self.frame1.copy()  # Create a copy of frame1 to avoid modifying the original
            frame2_copy = self.frame2.copy()  # Create a copy of frame2 to avoid modifying the original
            
            # Run YOLOv8 inference on the frames
            logging.getLogger("ultralytics").setLevel(logging.WARNING)  # Set logging level for YOLOv8 "to stop spamming in terminal"
            results1 = self.model(frame1_copy, conf=self.conf_threshold)  # Run YOLO detection on frame1
            results2 = self.model(frame2_copy, conf=self.conf_threshold)  # Run YOLO detection on frame2
            # Visualize the results on the frames
            annotated_frame1 = results1[0].plot()  # Create a visualization of the detection results for frame1
            annotated_frame2 = results2[0].plot()  # Create a visualization of the detection results for frame2

            box1 = results1[0].boxes 
            box2 = results2[0].boxes
            
            
            # Process Camera 1 results
            if len(box1.cls) > 0:
                for i in range(len(box1.cls)):
                    areas = self.calculate_apple_color_areas(frame1_copy, box1[i])
                    apple_type = self.display_apple_results(annotated_frame1, areas, i, "Camera 1")

            # Process Camera 2 results
            if len(box2.cls) > 0:
                for i in range(len(box2.cls)):
                    areas = self.calculate_apple_color_areas(frame2_copy, box2[i])
                    apple_type = self.display_apple_results(annotated_frame2, areas, i, "Camera 2")

            # Add FPS information
            cv2.putText(annotated_frame1, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Add FPS text to frame1
            cv2.putText(annotated_frame2, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Add FPS text to frame2
            
            # Display the annotated frames
            cv2.imshow("Camera 1 - YOLOv8 Detection", annotated_frame1)  # Show the annotated frame1 in a window
            cv2.imshow("Camera 2 - YOLOv8 Detection", annotated_frame2)  # Show the annotated frame2 in a window

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if the 'q' key was pressed
                self.running = False  # Set running flag to False to stop the program
                break  # Exit the loop
           
    def start(self):
        """Start the dual camera tracking system"""
        if not self.initialize_cameras():  # Initialize cameras and check if successful
            return  # Return early if camera initialization failed
        
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
        self.running = False  # Set running flag to False
        capture_thread.join(timeout=1.0)  # Wait for the capture thread to finish (with timeout)
        self.camera1.release()  # Release camera1 resources
        self.camera2.release()  # Release camera2 resources
        cv2.destroyAllWindows()  # Close all OpenCV windows
        print("Tracking stopped.")  # Print message indicating tracking has stopped

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
        
        if len(box.cls) > 0:
            print("\n" + "="*50)
            print("APPLE DETECTION RESULTS")
            print("="*50)
            
            # For each detected apple
            for i in range(len(box.cls)):
                # Calculate color areas for this apple
                areas = self.calculate_apple_color_areas(img, box[i])
                
                # Get apple classification
                apple_type, confidence, type_color = self.classify_apple_type(areas)
                
                # Display results in proper section on image
                self.display_apple_results(annotated_img, areas, i, "Image Analysis")
                
                # Create a summary box on the image
                summary_y = 50 + (i * 120)
                cv2.rectangle(annotated_img, (10, summary_y - 30), (400, summary_y + 80), (0, 0, 0), -1)
                cv2.rectangle(annotated_img, (10, summary_y - 30), (400, summary_y + 80), type_color, 3)
                
                # Main classification result
                cv2.putText(annotated_img, f"Apple #{i+1}: {apple_type}", 
                           (20, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, type_color, 2)
                
                # Confidence
                cv2.putText(annotated_img, f"Confidence: {confidence:.1f}%", 
                           (20, summary_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Color breakdown
                cv2.putText(annotated_img, f"G:{areas['green_percentage']:.0f}% R:{areas['red_percentage']:.0f}% Rot:{areas['rotten_percentage']:.0f}%", 
                           (20, summary_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Print detailed console output
                print(f"\nApple #{i+1} Analysis:")
                print(f"  Classification: {apple_type}")
                print(f"  Confidence: {confidence:.1f}%")
                print(f"  Color Breakdown:")
                print(f"    - Green: {areas['green_percentage']:.1f}%")
                print(f"    - Red: {areas['red_percentage']:.1f}%")
                print(f"    - Rotten: {areas['rotten_percentage']:.1f}%")
                print(f"  Status: ", end="")
                
                if apple_type == "ROTTEN":
                    print("⚠️  REJECT - Apple is rotten")
                elif apple_type in ["GREEN", "RED"]:
                    print("✅ ACCEPT - Apple is good quality")
                else:
                    print("❓ REVIEW - Mixed or unclear classification")
            
            print("\n" + "="*50)
        else:
            print("No apples detected in the image.")
            cv2.putText(annotated_img, "No apples detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the annotated image
        cv2.imshow("YOLOv8 Apple Detection & Classification", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":  # Check if this script is being run directly
    use_cam = True  # Set use_cam to True to access the camera and False to use an image
    if use_cam:
        tracker = DualCameraYOLOv8SingleObjectTracker(  # Create an instance of the tracker class
            camera1_id=0,  # Set camera1 ID to 0
            camera2_id=1,  # Set camera2 ID to 1
            model_name="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/train3/weights/best.pt",  # Set path to the YOLO model
            conf_threshold=0.5,
              # Set confidence threshold to 0.5
            # output_dir="./frames"  # Commented out: would set the output directory for saving frames
        )
        tracker.start()  # Start the tracking system
        
    else:
        tracker = DualCameraYOLOv8SingleObjectTracker(  # Create an instance of the tracker class
        camera1_id=0,  # Set camera1 ID to 0
        camera2_id=1,  # Set camera2 ID to 1
        model_name="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/train3/weights/best.pt",  # Set path to the YOLO model
        conf_threshold=0.5,
            # Set confidence threshold to 0.5
        # output_dir="./frames"  # Commented out: would set the output directory for saving frames
        image_path="D:/openCV/Apple-Tracker-/apple-new-1-20250512T074115Z-1-001/apple-new-1/imageYellow.png"
    )
        tracker.detect_on_image()
