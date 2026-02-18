import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

cap = cv2.VideoCapture(0)

# --- CHANGE 1: Update the model path to the Gesture Recognizer model ---
# Make sure you download 'gesture_recognizer.task' and put it in the same folder!
model_path = "gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
# --- CHANGE 2: Use GestureRecognizer instead of HandLandmarker ---
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
GestureRecognizerResult = vision.GestureRecognizerResult
VisionRunningMode = vision.RunningMode

latest_result = None

# --- CHANGE 3: Update Callback to handle GestureRecognizerResult ---
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

recognizer = GestureRecognizer.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        # Hand Landmark Drawing
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
        # Gesure recognition and data visualization
        if detection_result.gestures:
            gesture_category = detection_result.gestures[idx][0]
            gesture_name = gesture_category.category_name
            score = gesture_category.score
    
            cv2.putText(annotated_image, 
                        f"{gesture_name} ({score:.2f})", 
                        (10, 50 + (idx * 50)), # Position text
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    return annotated_image

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    # Convert BGR (OpenCV) to RGB (MediaPipe)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    timestamp_ms = int(time.time() * 1000)
    
    recognizer.recognize_async(mp_image, timestamp_ms)

    if latest_result:
        img = draw_landmarks_on_image(img, latest_result)

    cv2.imshow("Gesture Recognizer", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

recognizer.close()
cap.release()
cv2.destroyAllWindows()