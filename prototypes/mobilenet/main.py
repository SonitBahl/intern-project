import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the pre-trained MobileNetV2 SSD model from TensorFlow Hub
model = hub.load("https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/ssd-mobilenet-v2/1")

# Function to run object detection
def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

# Function to draw bounding boxes on the image
def draw_boxes(image, detections, class_names, threshold=0.7):
    height, width, _ = image.shape
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    
    for i in range(len(scores)):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            class_name = class_names[classes[i]-1]  # Adjust index by -1 since class IDs are 1-indexed
            label = f"{class_name}, Score: {scores[i]:.2f}"
            cv2.putText(image, label, (int(left), int(top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Load class names from coco.names
class_names_path = 'coco.names'
with open(class_names_path, 'r') as f:
    class_names = f.read().strip().split('\n')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect objects in the frame
    detections = detect_objects(frame_rgb)
    
    # Draw bounding boxes on the frame
    frame_with_boxes = draw_boxes(frame, detections, class_names)
    
    # Display the frame
    cv2.imshow('Object Detection', frame_with_boxes)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
