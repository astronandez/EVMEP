import cv2
import numpy as np
import tensorflow as tf
import os
import cv2

# Set FFmpeg preferences to use UDP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

class_names = ['Convertible', 'Crossover', 'Fastback', 'Hardtop Convertible', 'Hatchback', 'MPV', 'Minibus', 'Pickup Truck', 'SUV', 'Sedan', 'Sports', 'Wagon']

def load_tf_model(model_path):
    # Load the TensorFlow model
    model = tf.saved_model.load(model_path)

    # Find model ouput structure
    #print(model.signatures['serving_default'].structured_outputs)

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
To test locations of bounding boxes
def draw_boxes(frame, results, class_names):
    detections = results['output0'].numpy()[0]  # Convert to numpy and remove batch dimension

    for detection in detections:
        x_center, y_center, width, height, obj_score, *class_probs = detection[:5 + len(class_names)]

        if obj_score > 0.8:  # Adjust threshold as needed
            # Convert center coordinates to corner coordinates
            xmin = int(x_center - width / 2)
            xmax = int(x_center + width / 2)
            ymin = int(y_center)
            ymax = int(y_center - (2 * height / 3))

            # Debugging print statements
            print("Detection found:")
            print(f"    Object Score: {obj_score:.2f}")
            print(f"    Bounding Box: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
            print(f"    Width: {width}, Height: {height}")
            print(f"    x_center: {x_center}, y_center: {y_center}")

            class_id = np.argmax(class_probs)
            class_label = class_names[class_id]
            print(f"    Class: {class_label}, Probability: {class_probs[class_id]:.2f}")

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw class label and confidence
            cv2.putText(frame, f"{class_label} {obj_score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)
'''

def draw_boxes(frame, results, class_names):
    detections = results['output0'].numpy()[0]  # Convert to numpy and remove batch dimension

    for detection in detections:
        x_center, y_center, width, height, obj_score, *class_probs = detection[:5 + len(class_names)]

        if obj_score > 0.8:  # Adjust threshold as needed
            # Convert center coordinates to corner coordinates
            xmin = int(x_center - width / 2)
            xmax = int(x_center + width / 2)
            ymin = int(y_center)
            ymax = int(y_center - (2 * height / 3))

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            class_id = np.argmax(class_probs)
            class_label = class_names[class_id]

            # Draw class label and confidence
            cv2.putText(frame, f"{class_label} {obj_score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)
                
def main():
    #rtsp_url = "rtsp://10.1.1.89:554"
    #rtsp_url =  "rtsp://192.168.1.188:554"
    # rtsp_url = "rtsp://172.91.64.236:8554/unicast"

    rtsp_url = "rtsp://192.168.254.78:8554/unicast"
    model_path = '/home/tilboon/Documents/320x320/best_tf'
    input_size = (320, 320)  # Adjust to your model's input size

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    model = load_tf_model(model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_data = preprocess_image(frame, input_size)
        # Adjust 'serving_default' and input/output tensor names as per your model
        results = model.signatures['serving_default'](tf.constant(input_data)) 
        
        # Print the keys and shape of each output for inspection
        '''
        for key in results:
            print(f"{key}: shape {results[key].shape}")
            print(f"First element of {key}: {results[key][0]}")
        '''

        draw_boxes(frame, results, class_names)

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



