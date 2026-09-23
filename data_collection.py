import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
 
offset = 50
imgSize = 400
 
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']

        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        x1=max(0,x-offset)
        y1=max(0,y-offset)
        x2=min(x+w+offset,img.shape[1])
        y2=min(y+h+offset,img.shape[0])
        imgCrop=img[y1:y2,x1:x2]

        imgCropShape=imgCrop.shape
        imgWhite[0:imgCrop.shape[0],0:imgCrop.shape[1]]=imgCrop

        aspectRatio = h/w

        if aspectRatio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap]=imgResize

        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]=imgResize

        if imgCrop.size>0:
            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)