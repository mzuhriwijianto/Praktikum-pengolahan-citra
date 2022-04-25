import cv2
import numpy

min_YCC = numpy.array([0,133,77],numpy.uint8)
max_YCC = numpy.array([255,173,127],numpy.uint8)

img = cv2.imread("SkinDetection/tulip.jpg")
#Konversi RGB ke YCbCr
imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
#Pembentukan mask dengan batasan yang dibentuk
mask = cv2.inRange(imgYCC,min_YCC,max_YCC)
#Masking
result = cv2.bitwise_and(img, img, mask = mask)

cv2.imwrite('ProsesYC1.jpg',imgYCC)
cv2.imwrite('ProsesYC2.jpg',mask)
cv2.imwrite('ProsesYC3.jpg',result)
