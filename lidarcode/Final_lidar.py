import numpy as np
import cv2
import time
import pyttsx3
import sounddevice as sd
import speech_recognition as sr
from scipy.io.wavfile import write
import json


def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config


config = load_config('config.json')

def get_threshold_from_audio():
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()

    
    engine.say("Please say the threshold value.")
    engine.runAndWait()

   
    sample_rate = 44100
    duration = 5 
    print("Recording...")
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete.")

    
    write("output.wav", sample_rate, audio_data)

    
    audio_data = sr.AudioFile("output.wav")
    
    with audio_data as source:
        audio = recognizer.record(source)
    
    
    try:
        threshold_text = recognizer.recognize_google(audio, language="en-US")
        threshold = float(threshold_text)
        print("Threshold value received:", threshold)
        return threshold
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("Error retrieving audio: {0}".format(e))
        return None
    except ValueError:
        print("Invalid threshold value received.")
        return None


threshold = get_threshold_from_audio()
if threshold is None:
    print(f"Using default threshold value of {config['threshold']}")
    threshold = config['threshold']


def non_max_suppression(boxes, confidences, threshold):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=threshold)
    return indices


def detect_objects(frame, net, output_layers, classes, class_dict):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    conf_threshold = 0.5
    object_classes = set(config['vehicle_classes'] + config['animal_classes'] + [config['human_class']])
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id in object_classes:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = non_max_suppression(boxes, confidences, threshold=threshold)
    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detected_objects.append((x, y, x + w, y + h, class_ids[i]))
    return detected_objects


def estimate_distances(objects, frame_shape, focal_length, real_widths, camera_height, camera_angle):
    distances = []
    for obj in objects:
        x1, y1, x2, y2, class_id = obj
        obj_width_in_pixels = x2 - x1
        if obj_width_in_pixels > 0:  
            real_width = real_widths[str(class_id)]
            distance = (real_width * focal_length) / obj_width_in_pixels
            
            adjusted_distance = distance * np.cos(np.radians(camera_angle)) + camera_height * np.sin(np.radians(camera_angle))
            distances.append((adjusted_distance, class_id, (x1 + x2) // 2))
        else:
            distances.append((float('inf'), class_id, (x1 + x2) // 2))   
    return distances


def draw_lidar_on_frame(frame, objects, distances, user_position, class_dict, threshold):
    for obj, (distance, class_id, obj_center_x) in zip(objects, distances):
        x1, y1, x2, y2, _ = obj
        center_x, center_y = obj_center_x, (y1 + y2) // 2
        
       
        if distance < threshold:
            color = (0, 0, 255)  
        else:
            color = (0, 255, 0)  
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        direction = "left" if obj_center_x < user_position[0] else "right"
        cv2.putText(frame, f'{class_dict[str(class_id)]}: {distance:.2f} meters {direction}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.line(frame, user_position, (center_x, center_y), (0, 0, 255), 2)
    return frame


tts_engine = pyttsx3.init()


def announce_close_objects(distances, class_dict, user_position, threshold):
   
    distances_sorted = sorted(distances, key=lambda x: x[0])
    for distance, class_id, obj_center_x in distances_sorted:
        if distance < threshold:  
            direction = "left" if obj_center_x < user_position[0] else "right"
            tts_engine.say(f"{class_dict[str(class_id)]} approaching from the {direction}, {distance:.2f} meters away")
    tts_engine.runAndWait()


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()


output_layers = ['yolo_82', 'yolo_94', 'yolo_106']


cap = cv2.VideoCapture(config['input_video'])


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


user_positions = config['user_positions']


user_input = input("Enter your position (A, B, C, D): ").strip().upper()
if user_input in user_positions:
    user_position = tuple(user_positions[user_input])
else:
    print("Invalid position entered. Using default position A.")
    user_position = tuple(user_positions["A"])


output_path = config['output_video']
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


time_interval = 1 / 6
last_update_time = time.time()


focal_length = config['focal_length']
real_widths = config['real_widths']
camera_height = config['camera_height']
camera_angle = config['camera_angle']

last_processed_objects = []
last_processed_distances = []


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    last_processed_objects = detect_objects(frame, net, output_layers, classes, config['class_labels'])

    current_time = time.time()
    
    if current_time - last_update_time >= time_interval:
        last_processed_distances = estimate_distances(last_processed_objects, frame.shape, focal_length, real_widths, camera_height, camera_angle)
        last_update_time = current_time
        
        announce_close_objects(last_processed_distances, config['class_labels'], user_position, threshold)

   
    frame_with_lidar = draw_lidar_on_frame(frame.copy(), last_processed_objects, last_processed_distances, user_position, config['class_labels'], threshold)
    
    
    out.write(frame_with_lidar)
    
    
    cv2.imshow('Frame with LiDAR', frame_with_lidar)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
