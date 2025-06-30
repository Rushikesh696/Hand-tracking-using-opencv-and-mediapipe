import cv2
import mediapipe as mp 
import time

# Load video
video_path = 'rushivideo.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = 'output_video.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

# For calculating FPS
ptime = 0

while True:
    success, img = cap.read()
    if not success:
        break  # Exit loop if video ends or there's an error

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # If hands detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw each landmark
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:  # Highlight wrist point
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            # Draw hand connections
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calculate FPS
    ctime = time.time()
    fps_display = 1 / (ctime - ptime)
    ptime = ctime

    # Display FPS on frame
    cv2.putText(img, f'FPS: {int(fps_display)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Show image
    cv2.imshow('Hand Tracking', img)

    # Save frame to output video
    out.write(img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
