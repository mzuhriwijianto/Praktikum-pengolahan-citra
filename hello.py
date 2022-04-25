import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from array import *

# Main Function

def load_image():
     img01=cv2.imread('img/p1.png')
     img02=cv2.imread('img/p2.png')
     img03=cv2.imread('img/p3.png')
     img04=cv2.imread('img/p4.png')
     
     plt.subplot(2,2,1),plt.imshow(img01)
     plt.title('satu'), plt.xticks([]), plt.yticks([])
     plt.subplot(2,2,2),plt.imshow(img02)
     plt.title('dua'), plt.xticks([]), plt.yticks([])
     plt.subplot(2,2,3),plt.imshow(img03)
     plt.title('tiga'), plt.xticks([]), plt.yticks([])
     plt.subplot(2,2,4),plt.imshow(img04)
     plt.title('empat'), plt.xticks([]), plt.yticks([])
     
     plt.show()
     plt.close()
     return