import cv2
import numpy as np
import tensorflow as tf

import os
import cv2

# Set FFmpeg preferences to use UDP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# Now open your RTSP stream
cap = cv2.VideoCapture("rtsp://192.168.254.78:8554/unicast", cv2.CAP_FFMPEG)


class_names = ['Convertible', 'Crossover', 'Fastback', 'Hardtop Convertible', 'Hatchback', 'MPV', 'Minibus', 'Pickup Truck', 'SUV', 'Sedan', 'Sports', 'Wagon']

def load_tf_model(model_path):
    print("Loading TensorFlow model...")
    model = tf.saved_model.load(model_path)
    print("Model loaded. Model output structure:")
    print(model.signatures['serving_default'].structured_outputs)
    return model

def preprocess_image(image, input_size):
    # Resize the image
    resized = cv2.resize(image, input_size)
    
    # Normalize the image
    normalized = resized / 255.0

    # Convert to float32
    float_data = normalized.astype(np.float32)

    # Change from HWC to CHW format (Height x Width x Channels to Channels x Height x Width)
    chw_format = np.transpose(float_data, (2, 0, 1))

    # Add a batch dimension (BCHW format)
    batch_data = np.expand_dims(chw_format, axis=0)
    
    return batch_data

def draw_boxes(frame, results, class_names):
    print("Drawing boxes on frame...")
    detections = results['output0'].numpy()[0]  # Convert to numpy and remove batch dimension
    for detection in detections:
        x_center, y_center, width, height, obj_score, *class_probs = detection[:5 + len(class_names)]
        if obj_score > 0.9:  # Threshold check
            xmin = int(x_center - width / 2)
            xmax = int(x_center + width / 2)
            ymin = int(y_center)
            ymax = int(y_center - (2 * height / 3))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            class_id = np.argmax(class_probs)
            class_label = class_names[class_id]
            cv2.putText(frame, f"{class_label} {obj_score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    print("Boxes drawn.")

def main():
    rtsp_url = "rtsp://192.168.254.78:8554/unicast"
    model_path = '/Users/tilboon/Desktop/320x320/best_tf'
    input_size = (320, 320)

    print("Initializing video capture from URL:", rtsp_url)
    #cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Unable to open video stream.")
        return

    print("Loading model...")
    model = load_tf_model(model_path)

    while True:
        print("Reading frame...")
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from stream.")
            break

        print("Processing frame...")
        input_data = preprocess_image(frame, input_size)
        print("********************************")
        results = model.signatures['serving_default'](tf.constant(input_data))
        print("********************************")
        draw_boxes(frame, results, class_names)

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream and windows closed.")

if __name__ == "__main__":
    main()
