import cv2
from yolo_nas_s import yolo_nas_s

# # Initialize video
# cap = cv2.VideoCapture("input.mp4")

video = 'singlecarvideo.mp4'
# cap = cap_from_youtube(videoUrl, resolution='720p')
cap = cv2.VideoCapture(video)
start_time = 1 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (3840, 2160))

# Initialize YOLOv7 model
model_path = "yolo_nas_s_int8_with_calibration_v4.onnx"
yolo_nas_s_detector = yolo_nas_s(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolo_nas_s_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)
    # out.write(combined_img)

# out.release()