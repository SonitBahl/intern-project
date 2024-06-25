import pyttsx3
import asyncio

class StateMachine:
    def __init__(self, output_audio_path):
        self.output_text = ""
        self.output_audio_path = output_audio_path
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    async def handle_state(self, speech_text):
        if speech_text:
            self.output_text += speech_text + " "

    async def save_to_file(self):
        self.engine.save_to_file(self.output_text, self.output_audio_path)
        self.engine.runAndWait()

async def generate_speech(objects, distances, classes, vehicle_classes, close_threshold):
    speech_text = ""
    vehicles = [obj for obj in objects if obj[4] in vehicle_classes]
    vehicle_distances = [distances[i] for i in range(len(vehicles))]

    if len(vehicles) != len(vehicle_distances):
        print(f"Error: Mismatch between number of vehicles ({len(vehicles)}) and distances ({len(vehicle_distances)})")
        return ""

    close_vehicles = [(classes[vehicle[4]], distance) for vehicle, distance in zip(vehicles, vehicle_distances) if distance < close_threshold]

    if len(close_vehicles) > 0:
        speech_text = "Vehicle close by. "
    elif len(vehicles) > 1:
        speech_text = "Multiple vehicles detected. "

    return speech_text

async def generate_all_vehicles_announcement(objects, distances, classes, vehicle_classes):
    speech_text = "Multiple vehicles detected. "
    vehicles = [obj for obj in objects if obj[4] in vehicle_classes]
    vehicle_distances = []

    if len(vehicles) != len(distances):
        print(f"Warning: Mismatch between number of vehicles ({len(vehicles)}) and distances ({len(distances)})")

    for obj in vehicles:
        class_id = obj[4]
        if class_id in vehicle_classes:
            if distances and len(distances) > vehicles.index(obj):
                distance = distances[vehicles.index(obj)]
                speech_text += f"{classes[class_id]} ahead at {distance:.2f} meters. "

    return speech_text