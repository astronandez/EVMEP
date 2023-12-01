#
#   basic_object_detection_webcam.py
#       Contains the program functions needed to use a pretrained model to detect objects in real-time using the users
#       primary webcam.
#       Authors: Marc Hernandez & Tilboon Elberier
#

import cv2
import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models

yolo_nas_s = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
yolo_nas_s = yolo_nas_s.to("cuda" if torch.cuda.is_available() else "cpu")

# Press "q" on keyboard to exit the webcam tool
yolo_nas_s.predict_webcam(conf=0.5)
