import sys
import cv2 as cv
import numpy as np
from camera import RealSenseCamera

cam = RealSenseCamera()
try:
    img = cam.stream()
except KeyboardInterrupt:
    print('Exited.')
    sys.exit(0)
finally:
    cv.destroyAllWindows()



image = cv.bilateralFilter(img, 15, 70, 70)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30)
 
circles = np.uint16(np.around(circles))
cimg = np.copy(gray)

for i in circles[0,:]:
    cv.circle(img, (i[0], i[1]), i[2], (255,0,0),2)
    cv.circle(img, (i[0], i[1]), 2, (255,255,255), 3)
    cv.circle(cimg, (i[0], i[1]), i[2], (255,0,0),2)
    cv.circle(cimg, (i[0], i[1]), 2, (255,255,255), 3)

cv.imshow('circles_bgr', img)
cv.imshow('grayscale', gray)
cv.imshow('bilateral', image)
cv.imshow('circles_gray', cimg)
cv.waitKey(0)
cv.destroyAllWindows()