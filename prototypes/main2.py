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

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

results1 = []
results2 = []

thread1 = threading.Thread(target=read_frames, args=(cap1, results1, model))
thread2 = threading.Thread(target=read_frames, args=(cap2, results2, model))

thread1.start()
thread2.start()

while True:
    if results1 and results2:
        cv2.imshow('FEED1', results1[0][0].plot())
        cv2.imshow('FEED2', results2[0][0].plot())
        results1.clear()
        results2.clear()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()