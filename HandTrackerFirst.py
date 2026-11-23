import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils


# Open web
cap = cv2.VideoCapture(0)

# Import Mediapipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO
)

# Hand detector
hands = HandLandmarker.create_from_options(options)


while True:

    # Capture frames
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    results = hands.detect_for_video(mpImage, int(time.time() * 1000))

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            drawing_utils.draw_landmarks(img, hand_landmarks, mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            )
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()