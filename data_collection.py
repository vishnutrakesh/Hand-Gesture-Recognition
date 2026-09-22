import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
 
offset = 50
 
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']
        x1=max(0,x-offset)
        y1=max(0,y-offset)
        x2=min(x+w+offset,img.shape[1])
        y2=min(y+h+offset,img.shape[0])
        imgCrop=img[y1:y2,x1:x2]

        if imgCrop.size>0:
            cv2.imshow("Image Crop", imgCrop)

    cv2.imshow("Image", img)
    cv2.waitKey(1)