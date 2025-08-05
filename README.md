# Apple Detection and Classification with YOLOv8

This project provides a Python script for real-time apple detection and classification using the YOLOv8 model. It supports both dual-camera live tracking and static image analysis, classifying apples as green, red, or rotten, and provides color analysis for further reference.

## Features
- **Dual Camera Support:** Simultaneous detection and classification from two camera feeds.
- **YOLOv8 Integration:** Utilizes Ultralytics YOLOv8 for robust object detection.
- **Apple Type Classification:** Distinguishes between green, red, and rotten apples.
- **Color Area Analysis:** Calculates the percentage of green, red, and rotten areas within detected apples.
- **Real-Time Visualization:** Annotates frames with detection results, confidence scores, and color breakdowns.
- **Image Mode:** Supports detection and analysis on static images.

## Requirements
- Python 3.7+
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- OpenCV (`cv2`)
- NumPy

Install dependencies with:
```bash
pip install ultralytics opencv-python numpy
```

## Usage

### 1. Real-Time Dual Camera Tracking
By default, the script runs in camera mode. It will open two camera feeds, perform detection, and display annotated results.
There are three script with diffrent purposes such as by model which directly classify them by pretrained model. Secondly, classify by color which calculate green brown and yellow areas to classify. Thirdly, single detection for single apple (it uses algorithm to give a final result according to both feed)  

```bash
python yoloV8_result_by_model.py
```
or 
```bash
python yoloV8_result_by_color_area.py
```
or 
```bash
python yolov8_singleDetection.py
```

- **Camera IDs:**
  - `camera1_id=0` (default)
  - `camera2_id=1` (default)
- **Model Path:**
  - Update the `model_name` parameter in the script to point to your trained YOLOv8 weights (e.g., `train3/weights/best.pt`).

### 2. Static Image Detection
To run detection on a single image, set `use_cam = False` in the `__main__` section of the script and specify the `image_path`.

```python
if __name__ == "__main__":
    use_cam = False
    tracker = DualCameraYOLOv8SingleObjectTracker(
        camera1_id=0,
        camera2_id=1,
        model_name="path/to/your/best.pt",
        conf_threshold=0.5,
        image_path="path/to/your/image.png"
    )
    tracker.detect_on_image()
```

## File Structure
- `yoloV8_result_by_model.py` — Main script for detection and tracking
- `README.md` — Project documentation
- `train3/weights/best.pt` — (You must provide your own trained YOLOv8 model weights)
- Example images (optional)

## Notes
- Ensure your camera devices are connected and accessible by OpenCV.
- Update all file paths in the script to match your local setup.
- The script prints detailed results to the console and displays annotated images in OpenCV windows.

## Example Output
- Annotated video/image windows showing detected apples, their type (green, red, rotten), confidence scores, and color area breakdowns.
- Console output with detailed classification and color analysis for each detected apple.

## License
This project is for research and educational purposes. Please cite the original YOLOv8 authors if you use this code in your work.

## Credits
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy

