import cv2

img = cv2.imread("SkinDetection/tulip.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('gbGray.jpg',imgGray)
