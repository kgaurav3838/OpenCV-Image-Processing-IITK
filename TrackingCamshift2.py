# Object tracking with Camshift
# Detecting Green color T-Shirt
import cv2
import numpy as np
img = cv2.imread("images/gaurav8.jpg") # Choose your own image ..Here i have picked green color T-Shirt
roi = img[250: 450, 150: 400]  # [Height vs Width Range] 411x700...picking the region of interest
cv2.imshow('image1',roi)
a = 150
b = 250
width = 400 - a
height = 450 - b
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #  Conversion from BGR color space to HSV color space
# HSV have 3 channels
cv2.imshow("h", hsv_roi[:,:,0])
#cv2.imshow("s", hsv_roi[:,:,1])
#cv2.imshow("v", hsv_roi[:,:,2])

roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180]) # Here we have pick only one channel i.e. Hue and computing Histogram(None: No mask1)
#HSV have 180 values.
cap = cv2.VideoCapture(0)
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # To convert frame taken from camera into HSV pattern
    mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1) # Computing Back projection to find histogram of roi on the frame
    ret, track_window = cv2.CamShift(mask, (a, b, width, height), term_criteria)
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
    cv2.imshow("mask", mask)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()