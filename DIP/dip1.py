# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:57:48 2018

@author: victor cortez
"""

import numpy as np
import cv2

import time

VID_DIRECTORY = "D:\\Users\\f202897\\Desktop\\vm-master\\Videos\\"

cap = cv2.VideoCapture(VID_DIRECTORY+'dia.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
#detectShadows=False

timeslist=[]
while(1):
    
    millis1 = time.time()
    
    ret, frame = cap.read()
    
    if frame is None:
        print("none")
        break
    
    #kernels ==================================================================
    ksize=11
    OPEN_KERNEL = np.ones((ksize,ksize),np.uint8)
    OPEN_KERNEL[0,0] = 0
    OPEN_KERNEL[0,1] = 0
    OPEN_KERNEL[0,ksize-1] = 0
    OPEN_KERNEL[0,ksize-2] = 0
    
    OPEN_KERNEL[1,0] = 0
    OPEN_KERNEL[1,ksize-1] = 0
    
    OPEN_KERNEL[ksize-2,0] = 0
    OPEN_KERNEL[ksize-2,ksize-1] = 0
    
    OPEN_KERNEL[ksize-1,0] = 0
    OPEN_KERNEL[ksize-1,1] = 0
    OPEN_KERNEL[ksize-1,ksize-1] = 0
    OPEN_KERNEL[ksize-1,ksize-2] = 0
    #kernel = np.ones((5,5),np.uint8)
    #kernel[0,0] = 0
    #kernel[0,4] = 0
    #kernel[4,0] = 0
    #kernel[4,4] = 0
    ksize=31
    DIL_KERNEL=np.ones((ksize,ksize),np.uint8)
    DIL_KERNEL[0,0] = 0
    DIL_KERNEL[0,1] = 0
    DIL_KERNEL[0,ksize-1] = 0
    DIL_KERNEL[0,ksize-2] = 0
    
    DIL_KERNEL[1,0] = 0
    DIL_KERNEL[1,ksize-1] = 0
    
    DIL_KERNEL[ksize-2,0] = 0
    DIL_KERNEL[ksize-2,ksize-1] = 0
    
    DIL_KERNEL[ksize-1,0] = 0
    DIL_KERNEL[ksize-1,1] = 0
    DIL_KERNEL[ksize-1,ksize-1] = 0
    DIL_KERNEL[ksize-1,ksize-2] = 0
    #==========================================================================
    
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (5,5), 8)
    mblur = cv2.medianBlur(frame,5)
    
    fgmask=fgbg.apply(frame)
    #fgmaskblur=fgbg.apply(blur)
    fgmaskmblur=fgbg.apply(mblur)

    #fgmask[np.where(fgmask<=254)] = [0]
    #ret,fgmaskblur = cv2.threshold(fgmaskblur,127,255,cv2.THRESH_BINARY)
    ret,fgmaskmblur = cv2.threshold(fgmaskmblur,127,255,cv2.THRESH_BINARY)
    ret,fgmask = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)
    
    
    
    #erosion = cv2.morphologyEx(fgmask,cv2.MORPH_ERODE, kernel)
    
    #removes false positives from background
    #opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #openingblur = cv2.morphologyEx(fgmaskblur, cv2.MORPH_OPEN, kernel)
    openingmblur = cv2.morphologyEx(fgmaskmblur, cv2.MORPH_OPEN, OPEN_KERNEL)
    
    #removes false negarives on the contour
    #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #shows pixels removed by opening
    #tophat = cv2.morphologyEx(fgmask, cv2.MORPH_TOPHAT, kernel)
    #shows pixels added by closing
    #blackhat = cv2.morphologyEx(fgmask, cv2.MORPH_BLACKHAT, kernel)
    #dilation = cv2.morphologyEx(opening,cv2.MORPH_DILATE, np.ones([7,7], np.uint8))
    #dilationblur = cv2.morphologyEx(openingblur,cv2.MORPH_DILATE, np.ones([7,7], np.uint8))
    dilationmblur = cv2.morphologyEx(openingmblur,cv2.MORPH_DILATE, DIL_KERNEL)
    
    #cannyer = cv2.Canny(erosion, 100, 200)
    #cannyop = cv2.Canny(opening, 100, 200)
    
    
    #filling image
    h, w = openingmblur.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill=openingmblur.copy()
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = openingmblur | im_floodfill_inv
    
    #filling image (mblur)
    h, w = dilationmblur.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill=dilationmblur.copy()
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    #=========================================
    # Combine the two images to get the foreground.
    im_outdil = dilationmblur | im_floodfill_inv
    #=========================================
    
    image22, contours, hierachy = cv2.findContours(im_outdil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagehull = image22.copy()
    hull = [cv2.convexHull(c) for c in contours]
    
    #converting image back to 3 channels to display border colours
    image22=cv2.cvtColor(image22,cv2.COLOR_GRAY2BGR)
    imagehull=cv2.cvtColor(imagehull,cv2.COLOR_GRAY2BGR)
    
    #drawing contours
    image22 = cv2.drawContours(image22,contours,-1,(0,255,0))
    imagehull = cv2.drawContours(image22,hull,-1,(0,255,0))
    
    
    #cv2.imshow('fgmask', frame)
    #cv2.imshow('frame', cnts)
    

    cv2.imshow('filled', im_outdil)
    cv2.imshow('contour', imagehull)
    #cv2.imshow('dil', im_outdil)
    #cv2.imshow('can', canny)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    millis2 = time.time()
    millis=millis2 - millis1
    timeslist.append(millis*1000)

cap.release()
cv2.destroyAllWindows()

mean = sum(timeslist)/len(timeslist)
