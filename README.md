# Hand Tracker

A real-time hand tracking application built with Python that detects your hand through a webcam and draws its landmarks on screen.

# Preview

![Hand tracking preview](<HandTrackingPreview.png>)

## Features

* Opens your webcam and tracks hands in real time
* Detects up to 2 hands at once
* Draws 21 landmark points and the connections between them
* Shows the current FPS on screen
* Returns landmark positions in pixel coordinates
* Reusable HandDetector module

## Technologies Used

* Python
* OpenCV
* MediaPipe (Hand Landmarker)
* NumPy

## How It Works

1. The webcam is opened and frames are captured one by one.
2. Each frame is converted from BGR to RGB and passed to MediaPipe's Hand Landmarker.
3. The model returns 21 landmarks per detected hand.
4. The landmarks and their connections are drawn on the frame.
5. The frame rate is calculated and displayed in the corner.
6. Press q to close the window.

## HandDetector Module

findHands(img, draw=True)
Detects hands in a frame and optionally draws the landmarks. Returns the image.

findPosition(img, handNo=0, draw=True)
Returns a list of (id, x, y) for every landmark of the chosen hand, in pixels.

## Hand Landmarks

Each hand has 21 points, numbered 0 to 20.

* 0: Wrist
* 4: Thumb tip
* 8: Index fingertip
* 12: Middle fingertip
* 16: Ring fingertip
* 20: Pinky tip

## Requirements

* A webcam
* hand_landmarker.task model file placed in the project folder ([download here](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker))