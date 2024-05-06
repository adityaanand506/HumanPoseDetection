#Accuracy Rate
#THUNDER WITH SMS AND ACCURACY RATE

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
import json
import base64

# Load the MoveNet interpreter
interpreter = tf.lite.Interpreter(model_path="C:\\Users\\coola\\OneDrive\\Desktop\\thunder.tflite")
interpreter.allocate_tensors()

# Define edges and rendering functions (assuming you have them implemented)
def draw_connections(frame, keypoints, edges, confidence_threshold):
    # Implementation of draw_connections function
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)


def draw_keypoints(frame, keypoints, confidence_threshold):
    # Implementation of draw_keypoints function
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)
   

# Define edges and rendering parameters (assuming you have them defined)
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
# Define camera index
camera_index = 0

# Open the camera
cap = cv2.VideoCapture(camera_index)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Failed to open camera.")
    exit()

# Define your ClickSend API credentials
username = "adi12345"
password = "32479DA3-FA60-6329-86C5-4D6EAF89A0D0"

# Base URL for ClickSend API
base_url = "https://rest.clicksend.com/v3"

# Define your sender ID
sender_id = "adi12345"

# Construct the authentication header
auth_header = {
    "Authorization": "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
}

# Function to send SMS using ClickSend API
def send_sms(message, recipient):
    payload = {
        "messages": [
            {
                "source": "sdk",
                "from": sender_id,
                "body": message,
                "to": recipient
            }
        ]
    }
    endpoint = "/sms/send"
    response = requests.post(base_url + endpoint, headers=auth_header, json=payload)
    if response.status_code == 200:
        print(f"SMS sent successfully to {recipient}.")
    else:
        print(f"Failed to send SMS to {recipient}. Error:", response.text)

def detect_pose(keypoints):
    for person_keypoints in keypoints:
        nose_keypoint = person_keypoints[0]  # Accessing the first keypoint for each person
        nose_confidence = nose_keypoint[2]  # Accessing the confidence score of the nose keypoint
        
        # Check if confidence score for nose keypoint is above threshold for standing
        if nose_confidence > 0.4:
            # Check if the person is running (assuming legs keypoints are present)
            if person_keypoints[15][2] > 0.4 and person_keypoints[16][2] > 0.4:
                # Check if arms keypoints are not present to distinguish running from standing
                if person_keypoints[5][2] < 0.1 and person_keypoints[6][2] < 0.1 and \
                   person_keypoints[11][2] < 0.1 and person_keypoints[12][2] < 0.1:
                    return "Running", np.mean([person_keypoints[i][2] for i in range(17)])
            
            # Check if the person is falling (assuming shoulders and hips keypoints are present)
            if (person_keypoints[5][2] > 0.4 and person_keypoints[6][2] > 0.4) and \
               (person_keypoints[11][2] < 0.1 and person_keypoints[12][2] < 0.1):
                # Check if legs keypoints are not present to distinguish falling from standing
                if person_keypoints[15][2] < 0.1 and person_keypoints[16][2] < 0.1:
                    return "Falling", np.mean([person_keypoints[i][2] for i in range(17)])
            
            # If not running or falling, classify as standing
            return "Standing", np.mean([person_keypoints[i][2] for i in range(17)])
    
    # Return "Unknown" if no person's nose keypoint meets the confidence threshold
    return "Unknown", 0.0

# Initialize a flag to keep track of whether an SMS has been sent
sms_sent = False

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)  # Resize to 256x256
    input_image = tf.cast(img, dtype=tf.float32)
    
    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Rendering 
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    
    # Detect pose and calculate accuracy rate
    pose, accuracy = detect_pose(keypoints_with_scores[0])
    cv2.putText(frame, f"Pose: {pose}, Accuracy: {accuracy:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # If falling and SMS has not been sent yet, send SMS to emergency number
    if pose == "Falling" and not sms_sent:
        emergency_number = "+61411111111"
        message = "Emergency: Person detected falling!"
        send_sms(message, emergency_number)
        sms_sent = True  # Set the flag to True to indicate that SMS has been sent
    
    cv2.imshow('MoveNet Thunder', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

