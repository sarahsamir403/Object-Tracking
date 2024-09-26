import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

#load model
model = YOLO('yolov8n.pt')

#define function to process frames
def Process_Frame(frame):
  #resize frames
  frame_resized = cv2.resize(frame, (640,480))

  #perform object detection
  results = model(frame_resized, stream = True)

  #draw bounding box and label
  for result in results:
    if len(result.boxes) > 0:
      for box in result.boxes:
        x1, y1, x2,y2 = box.xyxy[0].int().tolist()
        conf = box.conf[0].item() #score
        cls = int(box.cls[0].item()) #label
        label = model.names[cls]
        color = (0,255,0)
        frame_resized = cv2.rectangle(frame_resized, (x1,y1), (x2,y2), color , 2)
        frame_resized = cv2.putText(frame_resized, f"{label} {conf: 2f}", (x1,y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    #resize frame
    frame_resized = cv2.resize(frame_resized, (frame.shape[1], frame.shape[0]))
    return frame_resized

def main():
  st.title('Object Tracking Application')
  video_file = st.file_uploader('Upload a video', type = ['mp4', 'avi', 'mov'])

  if video_file:
    with tempfile.NamedTemporaryFile(delete = False, suffix = '.mp4') as tmp_file:
      tmp_file.write(video_file.read())
      tmp_path = tmp_file.name

    captures = cv2.VideoCapture(tmp_path)

    stframe = st.empty()

    while True:
      ret, frame = captures.read()
      if not ret:
        st.write("Videp Processing Completed")
        break

      frame = Process_Frame(frame)
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      stframe.image(rgb_frame, channels = 'RGB', use_column_width= True)

    captures.release()
    os.remove(tmp_path)

if __name__ == '__main__':
  main()