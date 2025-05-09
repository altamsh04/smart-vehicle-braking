import cv2
import numpy as np
import time
import requests
import json
import os
import datetime

ADAFRUIT_IO_URL = "ADAFRUIT_IO_URL"
ADAFRUIT_IO_KEY = "ADAFRUIT_IO_KEY"

DISTANCE_THRESHOLD = 700  # 7 cm

CRASH_DIR = "crashed_vehicles"
if not os.path.exists(CRASH_DIR):
    os.makedirs(CRASH_DIR)
    print(f"Created directory: {CRASH_DIR}")

LOG_FILE = "logs.txt"

# https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
# https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
print("Loading YOLO model...")
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("CUDA is available. Using GPU acceleration.")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    print("CUDA is not available. Using CPU.")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

KNOWN_CAR_HEIGHT_CM = 150  # Average car height in cm
KNOWN_CAR_WIDTH_CM = 180   # Average car width in cm
FOCAL_LENGTH = 800         # Adjusted focal length for better distance calculation

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not connect to laptop camera")
    exit()

print("Connected to laptop camera successfully!")
print("Press 'q' to exit")

target_classes = {
    "vehicles": ["car", "truck", "bus", "motorbike"],
    "persons": ["person"],
    "animals": ["dog", "cat", "horse", "sheep", "cow", "elephant", "bear", 
                "zebra", "giraffe", "bird", "mouse", "rabbit"]
}

target_class_list = []
for category in target_classes.values():
    target_class_list.extend(category)

def write_log(vehicle_type, image_path):
    """Write detection log to the log file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | {vehicle_type} | {image_path}\n"
    
    try:
        with open(LOG_FILE, "a") as log_file:
            log_file.write(log_entry)
        print(f"Log entry added: {log_entry.strip()}")
    except Exception as e:
        print(f"Error writing to log file: {e}")

def save_vehicle_image(frame, vehicle_type):
    """Save detected vehicle image to the crashed_vehicles folder"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{vehicle_type}_{timestamp}.jpg"
    file_path = os.path.join(CRASH_DIR, filename)
    
    try:
        cv2.imwrite(file_path, frame)
        print(f"Saved vehicle image: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving vehicle image: {e}")
        return None

def send_to_adafruit(value, frame=None, vehicle_type=None):
    """Send detection data to Adafruit IO"""
    try:
        headers = {
            'Content-Type': 'application/json',
            'X-AIO-Key': ADAFRUIT_IO_KEY
        }
        payload = {'value': value}
        
        response = requests.post(
            ADAFRUIT_IO_URL,
            headers=headers,
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            print(f"Successfully sent data to Adafruit IO: {value}")
            
            # Save the vehicle image and log the event
            if frame is not None and vehicle_type is not None:
                image_path = save_vehicle_image(frame, vehicle_type)
                if image_path:
                    write_log(vehicle_type, image_path)
        else:
            print(f"Failed to send data to Adafruit IO. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Error sending data to Adafruit IO: {e}")

def calibrate_distance(pixel_width, pixel_height):
    """
    Calculate distance with non-linear correction for more accuracy.
    Uses both width and height for better accuracy.
    """
    if pixel_width <= 0 or pixel_height <= 0:
        return 0
    
    width_distance = (KNOWN_CAR_WIDTH_CM * FOCAL_LENGTH) / pixel_width
    
    height_distance = (KNOWN_CAR_HEIGHT_CM * FOCAL_LENGTH) / pixel_height
    
    raw_distance = (width_distance * 0.4) + (height_distance * 0.6)
    
    if pixel_height > 200:
        return raw_distance * 0.8
    elif pixel_height > 100:
        return raw_distance * 0.9
    else:  # Far objects
        return raw_distance * 1.0

last_sent_time = 0

# Minimum interval between Adafruit IO updates (in seconds)
SEND_INTERVAL = 60 # 1 min

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame from camera")
            break
            
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        
        start_time = time.time()
        outputs = net.forward(output_layers)
        end_time = time.time()
        
        fps = 1 / (end_time - start_time)
        
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                class_name = classes[class_id] if class_id < len(classes) else "unknown"
                
                if class_name in target_class_list and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = max(0, int(center_x - w / 2))
                    y = max(0, int(center_y - h / 2))

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        counts = {"vehicles": 0, "persons": 0, "animals": 0}
        
        vehicles_within_threshold = False
        closest_vehicle_distance = float('inf')
        closest_vehicle_type = None
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                class_name = str(classes[class_ids[i]])
                confidence = confidences[i]
                
                if class_name not in target_class_list:
                    continue
                
                if class_name in target_classes["vehicles"]:
                    color = (0, 0, 255)  # Red
                    counts["vehicles"] += 1
                    
                    distance_cm = calibrate_distance(w, h)
                    
                    if distance_cm < closest_vehicle_distance:
                        closest_vehicle_distance = distance_cm
                        closest_vehicle_type = class_name
                    
                    if distance_cm < DISTANCE_THRESHOLD:
                        vehicles_within_threshold = True
                        color = (0, 165, 255)  # Orange
                    
                    if distance_cm > 100:
                        distance_text = f"Distance: {distance_cm/100:.1f}m"
                    else:
                        distance_text = f"Distance: {int(distance_cm)}cm"
                    
                    cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x, y - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, distance_text, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                elif class_name in target_classes["persons"]:
                    color = (0, 255, 0)  # Green
                    counts["persons"] += 1
                    cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                elif class_name in target_classes["animals"]:
                    color = (255, 0, 0)  # Blue
                    counts["animals"] += 1
                    cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        threshold_status = "ALERT: Vehicle within threshold!" if vehicles_within_threshold else "No vehicles within threshold"
        cv2.putText(frame, threshold_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if vehicles_within_threshold else (0, 255, 0), 2)

        count_text = f"Vehicles: {counts['vehicles']} | Persons: {counts['persons']} | Animals: {counts['animals']}"
        cv2.putText(frame, count_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        current_time = time.time()
        if vehicles_within_threshold and current_time - last_sent_time > SEND_INTERVAL:
            # Pass the frame and vehicle type to save the image
            send_to_adafruit("1", frame.copy(), closest_vehicle_type)
            last_sent_time = current_time

        cv2.imshow("Object Detection with Distance", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated")