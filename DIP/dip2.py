# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:15:38 2018

@author: f202897
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
    
    kernel = np.ones((7,7),np.uint8)
    kernel[0,0] = 0
    kernel[0,6] = 0
    kernel[6,0] = 0
    kernel[6,6] = 0
    
    #gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (3,3), 0)
    
    #fgmask=fgbg.apply(frame)
    fgmaskblur=fgbg.apply(blur)

    #fgmask[np.where(fgmask<=254)] = [0]
    ret,fgmaskblur = cv2.threshold(fgmaskblur,127,255,cv2.THRESH_BINARY)
    #ret,fgmask = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)

    #image2, contours, hierachy = cv2.findContours(fgmaskblur.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image22, contours2, hierachy = cv2.findContours(fgmaskblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image2hull = image22.copy()
    #image2hull2 = image22.copy()
    hull = [cv2.convexHull(c) for c in contours2]
    #hull2 = cv2.convexHull(contours)
    
    #image2 = cv2.drawContours(image2,contours,-1,(0,255,0))
    
    #filling image
    h, w = fgmaskblur.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill=fgmaskblur.copy()
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = fgmaskblur | im_floodfill_inv
    
    #erosion = cv2.morphologyEx(fgmask,cv2.MORPH_ERODE, kernel)
    
    #removes false positives from background
    #openingblur = cv2.morphologyEx(fgmaskblur, cv2.MORPH_OPEN, kernel)
    #opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #removes false negarives on the contour
    #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #shows pixels removed by opening
    #tophat = cv2.morphologyEx(fgmask, cv2.MORPH_TOPHAT, kernel)
    #shows pixels added by closing
    #blackhat = cv2.morphologyEx(fgmask, cv2.MORPH_BLACKHAT, kernel)
    #dilation = cv2.morphologyEx(opening,cv2.MORPH_DILATE, np.ones([9,9], np.uint8))
    #dilationblur = cv2.morphologyEx(openingblur,cv2.MORPH_DILATE, np.ones([9,9], np.uint8))
    
    #cannyer = cv2.Canny(erosion, 100, 200)
    #cannyop = cv2.Canny(opening, 100, 200)
    
    
    #cv2.imshow('fgmask', frame)
    #cv2.imshow('frame', cnts)
    #image2=cv2.cvtColor(image2,cv2.COLOR_GRAY2BGR)
    image22=cv2.cvtColor(image22,cv2.COLOR_GRAY2BGR)
    image2hull=cv2.cvtColor(image2hull,cv2.COLOR_GRAY2BGR)
    #image2hull2=cv2.cvtColor(image2hull,cv2.COLOR_GRAY2BGR)
    
    #image2 = cv2.drawContours(image2,contours,-1,(0,255,0))
    image22 = cv2.drawContours(image22,contours2,-1,(0,255,0))
    image2hull = cv2.drawContours(image22,hull,-1,(0,255,0))
    #image2hull2 = cv2.drawContours(image2,hull,-1,(0,255,0))
    
    #image2hull2 = cv2.drawContours(image2,hull2,-1,(0,255,0))
        
    #cv2.imshow('image2', image2)
    cv2.imshow('external', im_out)
    cv2.imshow('hull', image2hull)
    #cv2.imshow('hull2', hull2)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    millis2 = time.time()
    millis=millis2 - millis1
    timeslist.append(millis*1000)

cap.release()
cv2.destroyAllWindows()

mean = sum(timeslist)/len(timeslist)
