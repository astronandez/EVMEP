import cv2
import numpy as np
import tensorflow as tf

from simple_weight_estimation import *

class_names = ['Convertible', 'Crossover', 'Fastback', 'Hardtop Convertible', 'Hatchback', 'MPV', 'Minibus', 'Pickup Truck', 'SUV', 'Sedan', 'Sports', 'Wagon']

def load_tf_model(model_path):
    # Load the TensorFlow model
    model = tf.saved_model.load(model_path)
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

'''
def print_detections(results, class_names):
    detections = results['output0'].numpy()[0]  # Convert to numpy and remove batch dimension

    for detection in detections:
        x_center, y_center, width, height, obj_score, *class_probs = detection[:5 + len(class_names)]
        if obj_score > 0.9:  # Adjust threshold as needed
            class_id = np.argmax(class_probs)
            class_label = class_names[class_id]

            # Calculate bounding box coordinates
            # Assuming coordinates are normalized and in the format [x_center, y_center, width, height]
            xmin = (x_center - width / 2)
            ymin = (y_center - height / 2)
            xmax = (x_center + width / 2)
            ymax = (y_center + height / 2)

            print(f"Detected: {class_label} with confidence {obj_score:.2f}")
            print(f"Bounding Box: [xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}]")
'''

def print_detections(results, class_names):
    detections = results['output0'].numpy()[0]  # Convert to numpy and remove batch dimension

    for detection in detections:
        x_center, y_center, width, height, obj_score, *class_probs = detection[:5 + len(class_names)]
        if obj_score > 0.9:  # Adjust threshold as needed
            class_id = np.argmax(class_probs)
            class_label = class_names[class_id]
            weight = estimate_vechicle_weight(class_label)
            print(f"    Class: {class_label}, Probability: {class_probs[class_id]:.2f}, lbs: {weight}")
            #print(f"Detected: {class_label} with confidence {obj_score:.2f}")

def main():
    rtsp_url = "rtsp://10.1.1.89:554"
    #rtsp_url =  "rtsp://192.168.1.188:554"
    model_path = '/Users/tilboon/Desktop/content/yolov5/best_tf'
    input_size = (320, 320)  # Adjust to your model's input size

    cap = cv2.VideoCapture(rtsp_url)
    model = load_tf_model(model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_data = preprocess_image(frame, input_size)
        results = model.signatures['serving_default'](tf.constant(input_data)) 

        # Print detected objects and their confidence
        print_detections(results, class_names)

        # Break loop with a key press (optional)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    main()

