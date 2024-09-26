# Object Tracking Application

This application uses the YOLOv8 model to perform real-time object detection on video files. The app allows users to upload a video and visually track objects with bounding boxes and labels drawn on each frame.

## Features

- **Object Detection**: Detects and tracks objects in uploaded video files using the YOLOv8 model.
- **Bounding Boxes**: Draws bounding boxes around detected objects with labels and confidence scores.
- **Supported Video Formats**: Accepts video files in `.mp4`, `.avi`, and `.mov` formats.

## Requirements

Make sure you have the following installed:

- Python 3.7+
- Streamlit
- OpenCV (`cv2`)
- Numpy
- PIL (Pillow)
- ultralytics (for YOLOv8)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
