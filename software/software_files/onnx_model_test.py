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

onnx_model = onnx.load('yolo_nas_s_int8_with_calibration_v2.onnx')

session = rt.InferenceSession(onnx_model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inname = [o.name for o in session.get_inputs()]
outname = [o.name for o in session.get_outputs()]
inp = {inname[0]: image_bchw}
result = session.run(outname, inp)

# output = torch.from_numpy(result)
# out = [non_max_suppression(output, overlapThresh=0.8)]

# for i,(x0,y0,x1,y1,score,cls_id) in enumerate(out):
#       box = np.array([x0,y0,x1,y1])
#       box -= np.array(dwdh*2)
#       box /= ratio
#       box = box.round().astype(np.int32).tolist()
#       cls_id = int(cls_id)
#       score = round(float(score),3)
#       name = names[cls_id]
#       color = colors[name]
#       name += ' '+str(score)
#       cv2.rectangle(img,box[:2],box[2:],color,2)
#       cv2.putText(img,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)

show_predictions_from_flat_format(image, result)