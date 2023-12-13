



import onnxruntime as ort
import numpy as np
import cv2

def preprocess_image(image_path, input_size=(320, 320)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_size)
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))  # CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def run_inference_onnx(model_path, image):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: image})
    return result

def main():
    test_image_path = '/Users/tilboon/Desktop/content/yolov5/Vehicle_Body_Style_Dataset/train/images/ff423094eafdf1_jpg.rf.e029553239c0480b8f2cbd12c61c4161.jpg'
    onnx_model_path = '/Users/tilboon/Desktop/content/yolov5/runs/train/exp/weights/best.onnx'

    preprocessed_image = preprocess_image(test_image_path)
    results = run_inference_onnx(onnx_model_path, preprocessed_image)
    print("Raw model output:", results)

if __name__ == "__main__":
    main()
