import numpy as np
import cv2
import time
import pyttsx3
import sounddevice as sd
import speech_recognition as sr
from scipy.io.wavfile import write

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
    print("Using default threshold value of 0.4")
    threshold = 0.4

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
    vehicle_classes = {0, 1, 2, 3, 5, 7, 8}  # Person, Bicycle, Car, Motorbike, Bus, Truck, Auto rickshaw
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
    
    indices = non_max_suppression(boxes, confidences, threshold=threshold)
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
            distances.append((adjusted_distance, class_id, (x1 + x2) // 2))
        else:
            distances.append((float('inf'), class_id, (x1 + x2) // 2))
    return distances

def draw_lidar_on_frame(frame, vehicles, distances, user_position, classes):
    close_threshold = 10.0

    for vehicle, (distance, class_id, vehicle_center_x) in zip(vehicles, distances):
        x1, y1, x2, y2, _ = vehicle
        center_x, center_y = vehicle_center_x, (y1 + y2) // 2
        
        if distance < close_threshold:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        direction = "left" if vehicle_center_x < user_position[0] else "right"
        cv2.putText(frame, f'{classes[class_id]}: {distance:.2f} meters {direction}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.line(frame, user_position, (center_x, center_y), (0, 0, 255), 2)
    return frame

tts_engine = pyttsx3.init()

def announce_risks(distances, classes, user_position):
    risks = [d for d in distances if d[0] < 5.0]
    if risks:
        closest_risk = min(risks, key=lambda x: x[0])
        distance, class_id, vehicle_center_x = closest_risk
        direction = "left" if vehicle_center_x < user_position[0] else "right"
        tts_engine.say(f"Warning! {classes[class_id]} crash {distance:.2f} meters away to your {direction}")
        tts_engine.runAndWait()
        return True
    return False

def announce_close_vehicles(distances, classes, user_position):
    distances_sorted = sorted(distances, key=lambda x: x[0])
    multiple_vehicles = len(distances_sorted) > 1 and all(d[0] < threshold for d in distances_sorted)
    if multiple_vehicles:
        direction = "left" if distances_sorted[0][2] < user_position[0] else "right"
        tts_engine.say(f"Multiple vehicles under the threshold value on your {direction}, please be careful")
        tts_engine.runAndWait()
        time.sleep(5)
    else:
        for distance, class_id, vehicle_center_x in distances_sorted:
            if distance < threshold:
                direction = "left" if vehicle_center_x < user_position[0] else "right"
                tts_engine.say(f"{classes[class_id]} approaching from the {direction}, {distance:.2f} meters away")
                tts_engine.runAndWait()

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

output_layers = ['yolo_82', 'yolo_94', 'yolo_106']

cap = cv2.VideoCapture("D:\\PYQ\\video-output-7CBBABAB-542D-423D-BBE4-9739B81F2969.mp4")

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

output_path = 'output_with_lidar22.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

time_interval = 1/6
last_update_time = time.time()

focal_length = 615
real_vehicle_widths = {
    0: 0.5,  # Average width of a person in meters
    1: 0.5,  # Average width of a bicycle in meters
    2: 2.0,  # Average width of a car in meters
    3: 0.7,  # Average width of a motorbike in meters
    5: 2.5,  # Average width of a bus in meters
    7: 2.5,  # Average width of a truck in meters
    8: 1.5,  # Average width of an auto rickshaw in meters
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

        if not announce_risks(last_processed_distances, classes, user_position):
            announce_close_vehicles(last_processed_distances, classes, user_position)

    frame_with_lidar = draw_lidar_on_frame(frame.copy(), last_processed_vehicles, last_processed_distances, user_position, classes)
    out.write(frame_with_lidar)
    cv2.imshow('Frame with LiDAR', frame_with_lidar)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
