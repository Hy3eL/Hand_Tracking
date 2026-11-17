import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()


    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()