# Python code to read image
import cv2

img = cv2.imread("img/p1.png", cv2.IMREAD_COLOR)
cv2.imshow('Pinarak Bojonegoro', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
