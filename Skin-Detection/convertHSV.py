import cv2
import numpy

hsvLOW = numpy.array([0, 48, 80],numpy.uint8)
hsvUP = numpy.array([20,255,255],numpy.uint8)

img = cv2.imread("SkinDetection/tulip.jpg")
#Konversi RGB ke HSV
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#Pembentukan mask dengan batasan yang dibentuk
mask = cv2.inRange(imgHSV,hsvLOW,hsvUP)
#Masking
result = cv2.bitwise_and(img,img, mask = mask)

cv2.imwrite('prosesc1HSV.jpg',imgHSV)
cv2.imwrite('prosesc2SV.jpg',mask)
cv2.imwrite('prosesc3SV.jpg',result)
