import cv2
import numpy as numpy
import imutils

Tr = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.legacy.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.legacy.TrackerTLD_create,
        "medianflow": cv2.legacy.TrackerMedianFlow_create,
        "mosse": cv2.legacy.TrackerMOSSE_create
    }

tracker = Tr['kcf']()
#tracker = cv2.TrackerCSRT_create()

camera = True #True for Webcam, else its video

if camera:
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture('#10.mp4')
_,frame = video.read()
frame = imutils.resize(frame,width=600)
cv2.imshow('Frame', frame)

BB = cv2.selectROI('Frame',frame)
tracker.init(frame, BB)
while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=600)
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(a) for a in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 2)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
video.release()
cv2.destroyAllwindows()

