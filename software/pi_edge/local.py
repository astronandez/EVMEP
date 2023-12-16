import cv2
import numpy as np
import tensorflow as tf

from simple_weight_estimation import *

class_names = ['Convertible', 'Crossover', 'Fastback', 'Hardtop Convertible', 'Hatchback', 'MPV', 'Minibus', 'Pickup Truck', 'SUV', 'Sedan', 'Sports', 'Wagon']

def load_tf_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def preprocess_image(image, input_size):
    resized = cv2.resize(image, input_size)
    normalized = resized / 255.0
    float_data = normalized.astype(np.float32)
    chw_format = np.transpose(float_data, (2, 0, 1))
    batch_data = np.expand_dims(chw_format, axis=0)
    return batch_data

def draw_boxes(frame, results, class_names):
    detections = results['output0'].numpy()[0]  
    for detection in detections:
        x_center, y_center, width, height, obj_score, *class_probs = detection[:5 + len(class_names)]
        if obj_score > 0.8: 
            xmin = int(2*(x_center - width / 2))
            xmax = int(2*(x_center + width / 2))
            ymin = int(2*(y_center))
            ymax = int(2*(y_center - (2 * height / 3)))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            class_id = np.argmax(class_probs)
            class_label = class_names[class_id]
            weight = estimate_vechicle_weight(class_label)
            print(f"    Class: {class_label}, Probability: {class_probs[class_id]:.2f}, lbs: {weight}")
            cv2.putText(frame, f"{class_label} {obj_score:.2f} lbs: {weight}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
            #cv2.putText(frame, f"{class_label} {obj_score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv2.imshow('Object Detection', frame)

def main():
    model_path = '/home/tilboon/Documents/320x320/best_tf'
    input_size = (320, 320)

    # Initialize the Raspberry Pi Camera
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    model = load_tf_model(model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_data = preprocess_image(frame, input_size)
        results = model.signatures['serving_default'](tf.constant(input_data))
        draw_boxes(frame, results, class_names)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


