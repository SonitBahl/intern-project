import cv2
from ultralytics import YOLO
import threading

def read_frames(cap, results, model):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results.append(model(frame, conf=0.65, task='segment'))

model = YOLO('yolov8l-seg.pt')

cap = cv2.VideoCapture(0)

results = []

thread = threading.Thread(target=read_frames, args=(cap, results, model))
thread.start()

while True:
    if results:
        cv2.imshow('FEED', results[0][0].plot())
        results.clear()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
