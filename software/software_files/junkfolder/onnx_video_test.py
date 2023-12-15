import cv2
import sys

import numpy
import numpy as np
import onnxruntime as rt
import onnx
from onnx.checker import check_model
from annotating_functions import *
import time

def function_to_evaluate(image):
    print(image, file=sys.stderr)
    image = cv2.resize(image, (320, 320))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, 0).astype('uint8')
    images_bchw = np.transpose(image, [0, 3, 1, 2])
    print(images_bchw.shape)
    inp = {inname[0], images_bchw}
    res = session.run(outname, inp)
    show_predictions_from_flat_format_no_save(image, res)


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
    test_images = []
    class_names = ['Convertible', 'Crossover', 'Fastback', 'Hardtop Convertible', 'Hatchback', 'MPV', 'Minibus',
                   'Pickup Truck', 'SUV', 'Sedan', 'Sports', 'Wagon']

    onnx_model = onnx.load('yolo_nas_s_int8_with_calibration_v2.onnx')
    session = rt.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    inname = [o.name for o in session.get_inputs()]
    outname = [o.name for o in session.get_outputs()]

    cam = cv2.VideoCapture("singlecarvideo.mp4")
    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))
    out = cv2.VideoWriter('outputsinglecarvideo_test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (frame_width, frame_height))

    count = 0
    container = []

    input_size = (320, 320)  # Adjust to your model's input size

    while (cam.isOpened()):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        try:
            ret, frame = cam.read()
            if not ret:
                break
        except Exception as e:
            print(e)
            continue

        input_data = preprocess_image(frame, input_size)
        result = session.run(None, {inname[0], input_data.astype(numpy.float32)})[0]
        draw_boxes(frame, result, class_names)


    cam.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()