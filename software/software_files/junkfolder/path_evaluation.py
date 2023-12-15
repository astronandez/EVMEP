import torch
import cv2
import numpy as np

def load_pytorch_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def preprocess_image(image, input_size=(320, 320)):
    image = cv2.resize(image, input_size)
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image)
    return image

def draw_boxes(frame, results, class_names):
    frame_height, frame_width, _ = frame.shape
    detections = results[0]

    for detection in detections:
        # Assuming YOLO format: x_center, y_center, width, height, obj_score, class_probs...
        x_center, y_center, width, height, obj_score, *class_probs = detection
        if obj_score > 0.5:
            # Convert from center coordinates to bounding box corners
            xmin = int((x_center - width / 2) * frame_width)
            xmax = int((x_center + width / 2) * frame_width)
            ymin = int((y_center - height / 2) * frame_height)
            ymax = int((y_center + height / 2) * frame_height)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            class_id = np.argmax(class_probs)
            class_label = class_names[class_id]
            cv2.putText(frame, f"{class_label} {obj_score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

def main():
    pytorch_model_path = 'Z:/CompCarsYOLO/model/content/checkpoints/final_car_type_identifier_qat/RUN_20231211_203659_170405/ckpt_best.pth'
    class_names = ['MPV', 'SUV', 'sedan', 'hatchback', 'minibus', 'fastback', 'estate',
                   'pickup', 'hardtop convertible', 'sports', 'crossover', 'convertible']

    model = load_pytorch_model(pytorch_model_path)
    cap = cv2.VideoCapture('singlecarvideo.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_image = preprocess_image(frame)
        with torch.no_grad():
            results = model(preprocessed_image)

        draw_boxes(frame, results, class_names)
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()