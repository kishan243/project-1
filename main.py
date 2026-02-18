import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# --- Setup for Visualization (from the documentation) ---
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
        # Draw the hand landmarks using the Drawing Utils
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image

# --- Your Original Setup ---
cap = cv2.VideoCapture(0)
model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

latest_result = None

# Callback for Live-Stream mode
def print_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_hand_detection_confidence = 0.5,
    num_hands=2,
    result_callback=print_result)

landmarker = HandLandmarker.create_from_options(options)

while True:
    success, img = cap.read()
    if not success: break

    # Convert BGR (OpenCV) to RGB (MediaPipe)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    # DRAWING LOGIC:
    # If we have a result from the callback, draw it on the current frame
    if latest_result:
        img = draw_landmarks_on_image(img, latest_result)

    cv2.imshow("Hand Landmarker", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()