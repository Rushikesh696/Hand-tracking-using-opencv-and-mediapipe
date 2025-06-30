import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.6)
mpDraw = mp.solutions.drawing_utils

# Start webcam feed
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

# For FPS calculation (optional)
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from webcam.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw hand landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Optional: print landmark coordinates
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Tracking - Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
