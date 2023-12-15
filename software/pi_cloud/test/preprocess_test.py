import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_frame(frame):
    """
    Preprocess the frame for object detection.
    """
    # Convert the color space from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the frame to 320x320
    frame_resized = cv2.resize(frame_rgb, (320, 320))

    # Normalize pixel values to [0, 1]
    frame_normalized = frame_resized / 255.0

    return frame_normalized

# RTSP URL
rtsp_url = "rtsp://192.168.254.78:8554/unicast"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream was opened successfully
if not cap.isOpened():
    print("Error: Stream could not be opened.")
    exit()

ret, frame = cap.read()

# Release the capture
cap.release()

if ret:
    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Display the original and processed frames
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Original Frame')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed_frame)
    plt.title('Processed Frame (320x320, Normalized)')
    plt.axis('off')

    plt.show()
else:
    print("Error: Frame could not be read from the stream.")
