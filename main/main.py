import cv2
import numpy as np
import time
import json
from concurrent.futures import ThreadPoolExecutor
import wave
import pyaudio
import asyncio
from tts import StateMachine, generate_speech, generate_all_vehicles_announcement
from dehaze import dehaze_frame

# Function to load configuration from JSON file
def load_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

# Function for non-maximum suppression
def non_max_suppression(boxes, confidences, threshold):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=threshold)
    return indices

# Function to detect objects using YOLO
def detect_objects(frame, net, output_layers, classes, conf_threshold, nms_threshold):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = non_max_suppression(boxes, confidences, threshold=nms_threshold)
    objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            objects.append((x, y, x + w, y + h, class_ids[i]))
    return objects

# Function to estimate distances to detected vehicles
def estimate_distances(vehicles, frame_shape, focal_length, real_vehicle_widths, camera_height, camera_angle):
    distances = []
    for vehicle in vehicles:
        x1, y1, x2, y2, class_id = vehicle
        vehicle_width_in_pixels = x2 - x1
        if vehicle_width_in_pixels > 0:
            real_width = real_vehicle_widths[str(class_id)] 
            distance = (real_width * focal_length) / vehicle_width_in_pixels
            
            adjusted_distance = distance * np.cos(np.radians(camera_angle)) + camera_height * np.sin(np.radians(camera_angle))
            distances.append(adjusted_distance)
        else:
            distances.append(float('inf'))
    return distances

# Function to draw LiDAR information on the frame
def draw_lidar_on_frame(frame, objects, distances, user_position, classes, vehicle_classes, close_threshold):
    for obj in objects:
        x1, y1, x2, y2, class_id = obj
        color = (0, 255, 0)  # Default color for non-vehicle objects
        label = f'{classes[class_id]}'
        
        if class_id in vehicle_classes:
            if distances:
                distance = distances.pop(0)
                if distance < close_threshold:
                    color = (0, 0, 255)  # Red for close vehicles
                label = f'{classes[class_id]}: {distance:.2f} meters'
        
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.line(frame, user_position, (center_x, center_y), (0, 0, 255), 2)
    return frame

async def main():
    # Load configuration
    config = load_config("config.json")
    
    net = cv2.dnn.readNet("yolofiles/yolov3.weights", "yolofiles/yolov3.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    classes = []
    with open("yolofiles/coco.names", "r") as f:
        classes = f.read().splitlines()

    output_layers = ['yolo_82', 'yolo_94', 'yolo_106']

    cap = cv2.VideoCapture(config["input_video"])

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    user_positions = config["user_positions"]

    if config["user_position"] in user_positions:
        user_position = tuple(user_positions[config["user_position"]])
    else:
        print("Invalid position in config. Using default position A.")
        user_position = tuple(user_positions["A"])

    output_path = config["output_video"]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_skip = 5  
    time_interval = 1/6
    last_update_time = time.time()
    announce_interval = 5  # Interval to announce all vehicles (in seconds)
    last_announce_time = time.time() - announce_interval  # Initialize to ensure first announcement

    focal_length = config["focal_length"]
    real_vehicle_widths = config["real_widths"]
    camera_height = config["camera_height"]
    camera_angle = config["camera_angle"]

    last_processed_objects = []
    last_processed_distances = []

    vehicle_classes = set(config["vehicle_classes"])
    close_threshold = config["close_threshold"]

    dehaze_flag = config["dehaze_flag"]

    audio_output_path = config.get("audio_output", "output/output.wav")
    state_machine = StateMachine(audio_output_path)

    with ThreadPoolExecutor() as executor:
        future_objects = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            if frame_id % frame_skip == 0:
                if future_objects:
                    last_processed_objects = future_objects.result()
                
                if dehaze_flag == 1:
                    frame = dehaze_frame(frame)  # Dehaze the frame if flag is set to 1

                future_objects = executor.submit(detect_objects, frame, net, output_layers, classes, config["threshold"], config["threshold"])
            
            current_time = time.time()
            
            if current_time - last_update_time >= time_interval:
                if future_objects:
                    last_processed_objects = future_objects.result()
                
                vehicles = [obj for obj in last_processed_objects if obj[4] in vehicle_classes]
                last_processed_distances = estimate_distances(vehicles, frame.shape, focal_length, real_vehicle_widths, camera_height, camera_angle)
                
                # Generate speech
                speech_text = await generate_speech(last_processed_objects, last_processed_distances, classes, vehicle_classes, close_threshold)
                await state_machine.handle_state(speech_text)

                if current_time - last_announce_time >= announce_interval:
                    # Announce all vehicles periodically
                    all_vehicles_text = await generate_all_vehicles_announcement(last_processed_objects, last_processed_distances, classes, vehicle_classes)
                    await state_machine.handle_state(all_vehicles_text)
                    last_announce_time = current_time
                
                last_update_time = current_time
            
            frame_with_lidar = draw_lidar_on_frame(frame.copy(), last_processed_objects, last_processed_distances.copy(), user_position, classes, vehicle_classes, close_threshold)
            
            out.write(frame_with_lidar)
            
            cv2.imshow('Frame with LiDAR', frame_with_lidar)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    await state_machine.save_to_file()

if __name__ == "__main__":
    asyncio.run(main())