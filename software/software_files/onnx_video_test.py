import cv2
import sys
import numpy as np
import onnxruntime as rt
import onnx
from onnx.checker import check_model
from annotating_functions import *
import time

test_images = []

onnx_model = onnx.load('yolo_nas_s_int8_with_calibration_v2.onnx')
session = rt.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
inname = [o.name for o in session.get_inputs()]
outname = [o.name for o in session.get_outputs()]

cam = cv2.VideoCapture("singlecarvideo.mp4")
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
out = cv2.VideoWriter('outputsinglecarvideo_test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

count = 0
container = []

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

    cv2.imshow('frame', frame)

    # function_to_evaluate(img)


cam.release()
out.release()
cv2.destroyAllWindows()