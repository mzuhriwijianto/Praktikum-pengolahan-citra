# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:15:41 2022

@author: DELL
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import time
from array import *

### -------------main function-------------------
def load_image():
    img01=cv2.imread('gambar/gojo.png')
    img01=cv2.cvtColor(img01, cv2.COLOR_BGR2RGB)
    img02=cv2.imread('gambar/kaneki.jpg')
    img02=cv2.cvtColor(img02, cv2.COLOR_BGR2RGB)
    img03=cv2.imread('gambar/yamanbagiri.jpg')
    img03=cv2.cvtColor(img03, cv2.COLOR_BGR2RGB)
    img04=cv2.imread('gambar/liam.jpg')
    img04=cv2.cvtColor(img04, cv2.COLOR_BGR2RGB)

    plt.subplot(2,2,1),plt.imshow(img01)
    plt.title('gojo'),plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(img02)
    plt.title('kaneki'),plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(img03)
    plt.title('yamanbagiri'),plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(img04)
    plt.title('william'),plt.xticks([]), plt.yticks([])
    
    plt.show()
    plt.close()
    return

def load_image1(): #masih error 
    img=cv2.imread('gambar/kuda.jpg')
    frame_resize=rescaleFrame(img,scale=.5)
    cv2.imshow('kuda',gambar)
    cv2.imshow('kuda resize',frame_resize)
    cv2.waitKey(0)
    return
    
def create_image():
    # create a black image
    img = np.zeros((512, 512,3), np.uint8)
    img = cv2.rectangle(img,(10,10),(500,500),(255,255,255),-1)
    img = cv2.rectangle(img,(50,50),(110,110),(0,0,255),3)
    img = cv2.rectangle(img,(150,150),(210,210),(0,0,255),-1)
    img = cv2.circle(img,(300,100),50,(255,0,0),-1)
    img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
    cv2.imshow('Gambar',img)
    cv2.waitKey(0)
    

def drawing_shape(): 
    blank = np.zeros((500,500,3), dtype='uint8')
    cv2.imshow('Blank', blank)
    
    # 1. paint the iamge certain colour
    #blank[100:300, 100:400] = 0,0,255 # BGR
    #cv2.imshow('Red', blank) 
    
    #2. draw a rectangle
    #cv2.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1) #BGR
    #cv2.imshow('Green Rectangle', blank)
    
    #3. draw a circle
    #cv2.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,255,0), thickness=-1)
    #cv2.imshow('Circle', blank)
    
    #3. draw a line
    #cv2.line(blank, (100,250), (300,400), (255,255,255), thickness=3)
    #cv2.imshow('Line', blank)
    
    #3. write text
    cv2.putText(blank, 'Watashi wa hikmi desu', (0,255), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), 1)
    cv2.imshow('Text', blank)
    
    
    cv2.waitKey(0)
    return
    
def basic_function():
        img = cv2.imread('gambar/kazoku.jpg')
        #cv2.imshow('Color', img)
       
        #converting to grayscale
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Gray', gray)   
        
        #blur
        blur = cv2.GaussianBlur(img, (7,7), cv2.BORDER_DEFAULT)
        #cv2.imshow('Blur', blur) 
        
        #edge cascade
        canny = cv2.Canny(blur, 125, 175)
        #cv2.imshow('Canny Edges', canny)
        
        #dilatinng the image
        dilated = cv2.dilate(canny, (7,7), iterations=3)
        #cv2.imshow('Dilated', dilated)
        
        #eroding
        eroded = cv2.erode(dilated, (7,7), iterations=3)
        cv2.imshow('Eroded', eroded)
        
        #resize
        resized = cv2.resize(img, (500,500), interpolation=cv2.INTER_CUBIC)
        #cv2.imshow('Resized', resized)
        
        #cropping
        cropped = img[50:200, 200:400]
        cv2.imshow('Cropped', cropped)
       
        
        cv2.waitKey(0)
        return
    
def gray_image():
    ori = mpimg.imread("gambar/liam.jpg")
    img=cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
    ret,biner = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    plt.subplot(3,1,1),plt.imshow(ori,cmap='gray')
    plt.title('Original'),plt.xticks([]), plt.yticks([])
    plt.subplot(3,1,2), plt.imshow(img,cmap= 'gray')
    plt.title('Grayscale'),plt.xticks([]), plt.yticks([])
    plt.subplot(3,1,3),plt.imshow(biner,cmap= 'gray')
    plt.title('Biner'),plt.xticks([]), plt.yticks([])
    return
    
def access_image():
    img01 =cv2.imread('gambar/akashi.jpg')
    row1,col1,n=img01.shape
    print (row1,col1)
    img02 = np.zeros((row1,col1,3), np.uint8)
    img03 = np.zeros((140,200,3), np.uint8)
    img04 = np.zeros((140,200,3), np.uint8)
    
    #img02 = cv2.cvtcolor(img01, cv2.color_bgr2rgb)
    #img03 = img02.copy()
    
    #color=(0, 0,255)#sample color 
    #img04 = np.full((140,200,3), color, np.uint8)
    #row4,col4,n=img04.shape
    #print (row4,col4)
    
    for y1 in range(0,col1-1):
        for x1 in range(0,row1-1):
            R,G,B=img01[x1,y1]
            img02[x1,y1]=[B,G,R]
            
    for y1 in range(0,70-1):
        for x1 in range(0,200-1):
            img03[y1,x1]=[255,0,0]
            
    for y1 in range(70,140-1):
        for x1 in range(0,200-1):
            img03[y1,x1]=[255,255,255]
                
    for y1 in range(0,140-1):
        for x1 in range(0,70-1):
            img04[y1,x1]=[255,0,0]   
                
    for y1 in range(0,140-1):
        for x1 in range(70,140-1):
            img04[y1,x1]=[0,255,0]
            
    for y1 in range(0,140-1):
        for x1 in range(140,200-1):
               img04[y1,x1]=[0,0,255]
    
    plt.subplot(2,2,1),plt.imshow(img01)
    plt.title('Gambar 01'),plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(img02)
    plt.title('Gambar 02'),plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(img03)
    plt.title('Gambar 03'),plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(img04)
    plt.title('Gambar 04'),plt.xticks([]), plt.yticks([])
    plt.show()
    return
    
    
def flip_image():
    img=mpimg.imread("gambar/liam.jpg")
    horizontal_img = cv2.flip(img,1)
    vertical_img = cv2.flip(img,0)    
    both_img = cv2.flip(img,-1)
    
    (h,w) = img.shape[:2]
    print (h,w)
    center = (w / 2, h / 2)
    angle90 = 90
    angle180 = 180
    angle270 = 270
    
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle90, scale)
    rotated90 = cv2.warpAffine(img, M, (w, h))
    horizontal_img =rotated90
    
    M = cv2.getRotationMatrix2D(center, angle180, scale)
    rotated180 = cv2.warpAffine(img, M, (w, h))
    vertical_img =rotated180
    
    M = cv2.getRotationMatrix2D(center, angle270, scale)
    rotated270 = cv2.warpAffine(img, M, (w, h))
    both_img =rotated270
    
    
    plt.subplot(2,2,1),plt.imshow(img)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(horizontal_img)
    plt.title('Flip Horizontal'), plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(vertical_img)
    plt.title('Flip Vertical'), plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(both_img)
    plt.title('Flip Both'), plt.xticks([]),plt.yticks([])
    plt.show()
    return

def thresholding1():
    img=mpimg.imread("gambar/akashi.jpg")
    row,col,n=img.shape
    img1 = np.zeros((row,col,3), np.uint8)
    img2 = np.zeros((row,col,3), np.uint8)
    img3 = np.zeros((row,col,3), np.uint8)
    img4 = np.zeros((row,col,3), np.uint8)
    img5 = np.zeros((row,col,3), np.uint8)

    a=64 #threshold
    b=int(256/a)
    print ("a1=",b)
    for y in range(0,col-1):
        for x in range(0,row-1):
            R,G,B=img[x,y]
            b=int(256/a)
            R=b*int(R/b)
            G=b*int(G/b)
            B=b*int(B/b)
            img1[x,y]=[R,G,B]

    a=32 #threshold
    b=int(256/a)
    print ("a2=",b)
    for y in range(0,col-1):
        for x in range(0,row-1):
            R,G,B=img[x,y]
            b=int(256/a)
            R=b*int(R/b)
            G=b*int(G/b)
            B=b*int(B/b)
            img2[x,y]=[R,G,B]

    a=16 #threshold
    b=int(256/a)
    print ("a3=",b)
    for y in range(0,col-1):
        for x in range(0,row-1):
            R,G,B=img[x,y]
            b=int(256/a)
            R=b*int(R/b)
            G=b*int(G/b)
            B=b*int(B/b)
            img3[x,y]=[R,G,B]

    a=8 #threshold
    b=int(256/a)
    print ("a4=",b)
    for y in range(0,col-1):
        for x in range(0,row-1):
            R,G,B=img[x,y]
            b=int(256/a)
            R=b*int(R/b)
            G=b*int(G/b)
            B=b*int(B/b)
            img4[x,y]=[R,G,B]

    a=2 #threshold
    b=int(256/a)
    print ("a5=",b)
    for y in range(0,col-1):
        for x in range(0,row-1):
            R,G,B=img[x,y]
            b=int(256/a)
            R=b*int(R/b)
            G=b*int(G/b)
            B=b*int(B/b)
            img5[x,y]=[R,G,B]

    titles = ['Original Image','TH=64','TH=32','TH=16','TH=8','TH=2']
    images = [img, img1, img2, img3, img4,img5]
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=0)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    return

def enhancement():
    img=mpimg.imread("gambar/shoto.jpg")
    row,col,n=img.shape
    img1 = np.zeros((row,col,3), np.uint8)
    img2 = np.zeros((row,col,3), np.uint8)
    img3 = np.zeros((row,col,3), np.uint8)
    img4 = np.zeros((row,col,3), np.uint8)
    img5 = np.zeros((row,col,3), np.uint8)

    th=50
    for y in range(0,col-1):
        for x in range(0,row-1):
            R,G,B=img[x,y]
            if (R+th) >255 :
                R=255
            else :
                R=R+th
            if (G+th) >255 :
                G=255
            else :
                G=G+th
            if (R+th) >255 :
                R=255
            else :
                R=R+th
            img1[x,y]=[R,G,B]

    th=4
    for y in range(0,col-1):
        for x in range(0,row-1):
            R,G,B=img[x,y]
            if (R*th) >255 :
                R=255
            else :
                R=R*th
            if (G+th) >255 :
                G=255
            else :
                G=G*th
            if (R+th) >255 :
                R=255
            else :
                R=R*th
            img2[x,y]=[R,G,B]
    
    xmax=0
    xmin=300

    for y in range(0,col-1):
        for x in range(0,row-1):
            R,G,B=img[x,y]
            gray=int((R+G+B)/3)
            if(gray>xmax):
                xmax=gray
            if(gray<xmin):
                xmin=gray

    d=xmax-xmin
    for y in range(0,col-1):
        for x in range(0,row-1):
            R,G,B=img[x,y]
            gray=int((R+G+B)/3)
            gray=int((255/d)*gray-xmin)
            img3[x,y]=[gray,gray,gray]

    print ("xmax=", xmax)
    print ("xmin=", xmin)

    titles = ['Original Image','BRIGHTNESS', 'CONTRAST','AUTO SCALE']
    images = [img, img1, img2, img3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show
    return    
   
def convolution2D():
    img1 = cv2.imread('gambar/shoto.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    kernel = np.ones((3, 3), np.float32) / 9
    #print (kernel)
    img2 = cv2.filter2D(img1, -1, kernel)

    kernel = np.array([[0,  -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img3 = cv2.filter2D(img1, -1, kernel)

    img4 = cv2.blur(img1, (5,5))
    img5 = cv2.GaussianBlur(img1, (3,3), 0)
    img6 = cv2.medianBlur(img1, 3)

    titles = ['Original Image','Filter 1/9','Sharpen','Blur','Gaussian','Median Blur']
    images = [img1,img2,img3,img4,img5,img6]
    for i in range(6):
        plt.subplot(3,2,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    return
         
def dilatation():
    img1 = cv2.imread('gambar/oikawa.jpg')

    # convert to black and white
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    r ,img1 = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY)
    #create kernel
    kernel = np.ones((5,5), np.uint8)
    img2 = cv2.erode(img1, kernel)
    img3 = cv2.dilate(img1, kernel)
    img4 = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel)

    img5 = cv2.GaussianBlur(img1, (3,3), 0)
    img6 = cv2.medianBlur(img1, 3)

    titles = ['Original Image','Erosion','Dilatation','morphologyEx','Gaussian','Median Blur']
    images = [img1,img2,img3,img4,img5,img6]
    for i in range(6):
        plt.subplot(3,2,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    return

def filtering():
    img1=cv2.imread('gambar/oik.jpg')
    kernel= np.array([[1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]])

    kernel=kernel/25
    img2= cv2.filter2D(img1, -1, kernel)
    kernel= np.array([[0.0, -1.0, 0.0],
                    [-1.0, 4.0, -1.0],
                    [0.0, -1.0, 0.0]])

    kernel=kernel/(np.sum(kernel) if np.sum(kernel) !=0 else 1)
    img3= cv2.filter2D(img1, -1, kernel)
    kernel= np.array([[0.0, -1.0, 0.0],
                    [-1.0, 5.0, -1.0],
                    [0.0, -1.0, 0.0]])
    kernel=kernel/(np.sum(kernel) if np.sum(kernel) !=0 else 1)
    img4= cv2.filter2D(img1, -1, kernel)

    # img4= cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel)        
    # img5= cv2.GaussianBlur(img1, (3,3), 0)
    # img6= cv2.medianBlur(img1, 3)

    kernel= np.array([[-1.0, -1.0,],
                    [2.0, 2.0],
                    [-1.0, -1.0 ]])
    kernel=kernel/(np.sum(kernel) if np.sum(kernel) !=0 else 1)
    img5= cv2.filter2D(img1, -1, kernel)

    titles= ['original image', 'low pass', 'high pass', 'high pass', 'cusom kernel', 'normal']
    images= [img1, img2, img3, img4, img5, img1]

    for i in range(6):
        plt.subplot(3,2,i+1), plt.imshow(images[i], 'gray', vmin=-0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    return

def spektrum():
    img=cv2.imread('gambar/kuda.jpg',0)
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])) 
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray') 
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    return

def spektrum2():
    img=cv2.imread('gambar/kuda.jpg',0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift)) 
    plt.subplot(121),plt.imshow(img, cmap = 'gray') 
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray') 
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([]) 
    plt.show()
    return

def afterhpfjet():
    img=cv2.imread('gambar/kuda.jpg',0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    rows, cols = img.shape
    crow,ccol = int(rows/2) , int(cols/2)
    print (crow,ccol)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift) 
    img_back = np.abs(img_back)

    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(132),plt.imshow(img_back, cmap = 'gray') 
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(133),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([]) 
    plt.show()
    return

def spektrum3():
    img=cv2.imread('gambar/kuda.jpg',0)
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow,ccol = int(rows/2) , int(cols/2)
    # create a mask first, center square is 1, remaining all zeros 
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    # apply mask and inverse DF1
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1]) 
    plt.subplot(121),plt.imshow(img, cmap = 'gray') 
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_back, cmap = 'gray') 
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([]) 
    plt.show()
    return

def lapsobel():
    img=cv2.imread("gambar/kuda.jpg",0)
    laplacian = cv2. Laplacian(img,cv2.CV_64F) 
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) 
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray') 
    plt.title('Original'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray') 
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray') 
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray') 
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([]) 
    plt.show()
    return

def hpffilter():
    # simple averaging filter without scaling parameter
    mean_filter = np.ones((3,3))
    # creating a guassian filter 
    x = cv2.getGaussianKernel(5,10) 
    gaussian = x*x.T

    # different edge detecting
    # scharr in x-direction
    scharr = np.array([[-3, 0, 3],
                       [-10,0,10],
                       [-3, 0, 3]])
    # sobel in x direction
    sobel_x= np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    # sobel in y direction 
    sobel_y= np.array([[-1,-2,-1],
                       [0, 0, 0],
                       [1, 2, 1]])
    # :Laplacian
    laplacian=np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr] 
    filter_name = ['mean filter', 'gaussian','laplacian', 'sobel_x', \
                   'sobel_y', 'scharr_x']
    fft_filters = [np.fft.fft2(x) for x in filters]
    fft_shift = [np.fft.fftshift(y) for y in fft_filters]
    mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray') 
        plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

    plt.show()
    return
       
#--------------main menu------------------
"""
praktikum 1

"""
#load_image()
#load_image1()
#create_image()
#drawing_shape()
#basic_function()   
#gray_image()

"""
praktikum 2

"""
#access_image()
#flip_image()

"""
praktikum 3

"""
#thresholding1()
#enhancement()

"""
praktikum 4

"""
#convolution2D()
#dilatation()
#filtering()

"""
praktikum 5

"""
#spektrum()
#spektrum2()
#spektrum3()
#afterhpfjet()
#lapsobel()
#hpffilter()