import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Import Mediapipe
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        # Hand detector
        self.hands = HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.hands.detect_for_video(mpImage, int(time.time() * 1000))

        if self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                if draw:
                    drawing_utils.draw_landmarks(img, hand_landmarks, mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.hand_landmarks:
            myHand = self.results.hand_landmarks[handNo]

            for id, Lm in enumerate(myHand):
                h, w, c = img.shape
                cx, cy = int(Lm.x * w), int(Lm.y * h)
                lmList.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    pTime = 0
    cTime = 0

    while True:

        # Capture frames
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        print(lmList[12]) if lmList else None
        # Frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()