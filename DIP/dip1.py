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
    
    kernel = np.ones((7,7),np.uint8)
    kernel[0,0] = 0
    kernel[0,6] = 0
    kernel[6,0] = 0
    kernel[6,6] = 0
    
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (5,5), 8)
    
    fgmask=fgbg.apply(frame)
    fgmaskblur=fgbg.apply(blur)

    #fgmask[np.where(fgmask<=254)] = [0]
    ret,fgmaskblur = cv2.threshold(fgmaskblur,127,255,cv2.THRESH_BINARY)
    ret,fgmask = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)


    
    
    
    
    #erosion = cv2.morphologyEx(fgmask,cv2.MORPH_ERODE, kernel)
    
    #removes false positives from background
    openingblur = cv2.morphologyEx(fgmaskblur, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #removes false negarives on the contour
    #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #shows pixels removed by opening
    #tophat = cv2.morphologyEx(fgmask, cv2.MORPH_TOPHAT, kernel)
    #shows pixels added by closing
    #blackhat = cv2.morphologyEx(fgmask, cv2.MORPH_BLACKHAT, kernel)
    dilation = cv2.morphologyEx(opening,cv2.MORPH_DILATE, np.ones([9,9], np.uint8))
    dilationblur = cv2.morphologyEx(openingblur,cv2.MORPH_DILATE, np.ones([9,9], np.uint8))
    
    #cannyer = cv2.Canny(erosion, 100, 200)
    #cannyop = cv2.Canny(opening, 100, 200)
    
    
    #cv2.imshow('fgmask', frame)
    #cv2.imshow('frame', cnts)
    

    cv2.imshow('dil', dilation)
    cv2.imshow('dilblur', dilationblur)
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
