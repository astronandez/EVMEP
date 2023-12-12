import cv2
import sys
import numpy as np
import torch
from super_gradients.training.utils.media.image import load_image
import onnxruntime as rt
import onnx
from onnx.checker import check_model
from visualization_model_predictions import show_predictions_from_flat_format
from imutils.object_detection import non_max_suppression

image = load_image('Z:/CompCarsYOLO/model/content/1s_image.jpg')
image = cv2.resize(image, (320, 320))
image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))
# print(image_bchw, file=sys.stderr)
onnx_model = onnx.load('yolo_nas_s_int8_with_calibration_v2.onnx')

session = rt.InferenceSession(onnx_model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inname = [o.name for o in session.get_inputs()]
outname = [o.name for o in session.get_outputs()]
inp = {inname[0]: image_bchw}
result = session.run(outname, inp)

show_predictions_from_flat_format(image, result)