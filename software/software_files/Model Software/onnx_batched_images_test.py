import os
import time
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

path_to_images = 'C:/Users/Marc Hernandez/Documents/UCLA/ECE 202A/EVMEP/software/software_files/Test Images/'
file_images = os.listdir(path_to_images)
file_images = [f"{file_name}" for file_name in file_images]
original_images = [cv2.imread(f"{path_to_images}/{file}") for file in file_images]

for image in range(len(original_images)):
    original_images[image] = cv2.cvtColor(original_images[image], cv2.COLOR_BGR2RGB)
    test_images.append(np.transpose(np.expand_dims(original_images[image], 0), (0, 3, 1, 2)))

def function_to_eval():
    for image in range(len(test_images)):
        inp = {inname[0]: test_images[image]}
        res = session.run(outname, inp)
        show_predictions_from_flat_format(original_images[image], res, file_images[image])

if __name__=='__main__':
    start = time.time()
    function_to_eval()
    end = time.time() - start
    print('{:.6f}s for the calculation'.format(end), file=sys.stderr)