import cv2

def test_stream(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Error: Cannot open video stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read video frame")
            break

        cv2.imshow('Video Stream Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rtsp_url = "rtsp://192.168.254.78:8554/unicast"
    test_stream(rtsp_url)
