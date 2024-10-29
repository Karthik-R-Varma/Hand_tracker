import cv2
import mediapipe as mp

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    success, img = cap.read()
    if not success:
        break

    # Convert the image to RGB format (MediaPipe requires RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image to detect hands
    results = hands.process(img_rgb)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the original image
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Print landmark positions
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"ID: {id}, X: {cx}, Y: {cy}")

    # Display the image with the hand landmarks
    cv2.imshow("Hand Tracker", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
