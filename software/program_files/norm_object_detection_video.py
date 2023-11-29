import cv2
import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models
import numpy as np
import math

cap = cv2.VideoCapture("Videos/bikes.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

yolo_nas_s = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
yolo_nas_s = yolo_nas_s.to("cuda" if torch.cuda.is_available() else "cpu")

count = 0
object_types = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

out = cv2.VideoWriter('OutputVideos/outputbikesdetect.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    count += 1
    if ret:
        detects = list(yolo_nas_s.predict(frame, conf=0.35))[0]
        bbox_xyxys = detects.prediction.bboxes_xyxy.tolist()
        confidences = detects.prediction.confidence
        labels = detects.prediction.labels.tolist()
        for (bbox_xyxy, conf, label) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_name = object_types[int(label)]
            class_conf = math.ceil((conf * 100)) / 100
            name_plate = f'{class_name}{class_conf}'
            t_size = cv2.getTextSize(name_plate, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(frame, name_plate, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            resize_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        out.write(frame)
        cv2.imshow("Frame", resize_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()