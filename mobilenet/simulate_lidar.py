import numpy as np
import cv2
import time


def non_max_suppression(boxes, confidences, threshold):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=threshold)
    return indices


def detect_vehicles(frame, net, output_layers, classes):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    conf_threshold = 0.5
    vehicle_classes = {1, 2, 3, 5, 7}  
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id in vehicle_classes:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = non_max_suppression(boxes, confidences, threshold=0.4)
    vehicles = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            vehicles.append((x, y, x + w, y + h, class_ids[i]))
    return vehicles


def estimate_distances(vehicles, frame_shape, focal_length, real_vehicle_widths, camera_height, camera_angle):
    distances = []
    for vehicle in vehicles:
        x1, y1, x2, y2, class_id = vehicle
        vehicle_width_in_pixels = x2 - x1
        if vehicle_width_in_pixels > 0:  
            real_width = real_vehicle_widths[class_id]
            distance = (real_width * focal_length) / vehicle_width_in_pixels
            
            adjusted_distance = distance * np.cos(np.radians(camera_angle)) + camera_height * np.sin(np.radians(camera_angle))
            distances.append(adjusted_distance)
        else:
            distances.append(float('inf')) 
    return distances


def draw_lidar_on_frame(frame, vehicles, distances, user_position, classes):
    close_threshold = 10.0  

    for vehicle, distance in zip(vehicles, distances):
        x1, y1, x2, y2, class_id = vehicle
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        
        if distance < close_threshold:
            color = (0, 0, 255)  
        else:
            color = (0, 255, 0)  
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{classes[class_id]}: {distance:.2f} meters', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.line(frame, user_position, (center_x, center_y), (0, 0, 255), 2)
    return frame


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()


output_layers = ['yolo_82', 'yolo_94', 'yolo_106']


cap = cv2.VideoCapture("D:\\Headgear\\5927704-hd_1920_1080_30fps.mp4")


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


user_positions = {
    "A": (50, frame_height - 50),
    "B": (frame_width // 4, frame_height - 50),
    "C": (frame_width // 2, frame_height - 50),
    "D": (3 * frame_width // 4, frame_height - 50)
}


user_input = input("Enter your position (A, B, C, D): ").strip().upper()
if user_input in user_positions:
    user_position = user_positions[user_input]
else:
    print("Invalid position entered. Using default position A.")
    user_position = user_positions["A"]


output_path = 'output_with_lidar11.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


time_interval = 1/6
last_update_time = time.time()


focal_length = 615  
real_vehicle_widths = {
    1: 0.5,  
    2: 2.0,  
    3: 0.7, 
    5: 2.5,  
    7: 2.5,  
}
camera_height = 5 
camera_angle = 0.0  

last_processed_vehicles = []
last_processed_distances = []


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    last_processed_vehicles = detect_vehicles(frame, net, output_layers, classes)

    current_time = time.time()
    
    if current_time - last_update_time >= time_interval:
        last_processed_distances = estimate_distances(last_processed_vehicles, frame.shape, focal_length, real_vehicle_widths, camera_height, camera_angle)
        last_update_time = current_time

    
    frame_with_lidar = draw_lidar_on_frame(frame.copy(), last_processed_vehicles, last_processed_distances, user_position, classes)
    
    
    out.write(frame_with_lidar)
    
 
    cv2.imshow('Frame with LiDAR', frame_with_lidar)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
