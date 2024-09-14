import cv2
import numpy as np
import mediapipe as mp
import pygame

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize Pygame to play sounds
pygame.mixer.init()

# Load piano key sounds (Ensure these files are in the same directory as your script)
piano_sounds = {
    'C': pygame.mixer.Sound("C.wav"),
    'D': pygame.mixer.Sound("D.wav"),
    'E': pygame.mixer.Sound("E.wav"),
    'F': pygame.mixer.Sound("F.wav"),
    'G': pygame.mixer.Sound("G.wav"),
    'A': pygame.mixer.Sound("A.wav"),
    'B': pygame.mixer.Sound("B.wav")
}

# Define the virtual piano key positions
key_positions = {
    'C': (50, 400, 100, 500),
    'D': (100, 400, 150, 500),
    'E': (150, 400, 200, 500),
    'F': (200, 400, 250, 500),
    'G': (250, 400, 300, 500),
    'A': (300, 400, 350, 500),
    'B': (350, 400, 400, 500)
}

# Open webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Helper function to check if a point is inside a key area
def is_finger_on_key(x, y, key_position):
    x1, y1, x2, y2 = key_position
    return x1 <= x <= x2 and y1 <= y <= y2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip frame horizontally to avoid mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw virtual piano keys on the screen
    for key, pos in key_positions.items():
        cv2.rectangle(frame, (pos[0], pos[1]), (pos[2], pos[3]), (255, 255, 255), 2)
        cv2.putText(frame, key, (pos[0] + 10, pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Process hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the fingertip position of the index finger (landmark 8)
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

            # Check if the fingertip is over any piano key
            for key, pos in key_positions.items():
                if is_finger_on_key(x, y, pos):
                    # Play the corresponding piano sound
                    piano_sounds[key].play()

                    # Highlight the pressed key
                    cv2.rectangle(frame, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), -1)
                    break

    # Display the video frame
    cv2.imshow("AI Piano", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()