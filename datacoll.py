import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
import time  # For auto-clicking photos after a set interval

# Initialize the camera and hand detector
cap = cv2.VideoCapture(0)  # Use 1 for Iriun Webcam, 0 for default
if not cap.isOpened():
    print("Error: Could not open Iriun Webcam.")
    exit()

detector = HandDetector(maxHands=2)
offset = 20
imgsize = 300
counter = 0

# Folder to save images
folder = "E:/sign language/data/hello" 
if not os.path.exists(folder):
    os.makedirs(folder)

last_save_time = time.time()
photo_interval = 0.01  # Interval in seconds (1 second)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img, draw=False) 
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgcrop = img[y1:y2, x1:x2]

        if imgcrop.size == 0:
            continue

        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        aspectratio = h / w

        if aspectratio > 1:
            k = imgsize / h
            wcal = math.ceil(k * w)
            imgresize = cv2.resize(imgcrop, (wcal, imgsize))
            wgap = math.ceil((imgsize - wcal) / 2)
            imgwhite[:, wgap:wcal + wgap] = imgresize
        else:
            k = imgsize / w
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize, hcal))
            hgap = math.ceil((imgsize - hcal) / 2)
            imgwhite[hgap:hcal + hgap, :] = imgresize

        cv2.imshow("Cropped Image", imgcrop)
        cv2.imshow("White Background Image", imgwhite)

        if time.time() - last_save_time >= photo_interval:
            counter += 1
            cv2.imwrite(f"{folder}/image{counter}.jpg", imgwhite)
            print(f"Image {counter} saved.")
            last_save_time = time.time() 

    cv2.imshow("Original Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
