# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:57:48 2018

@author: victor cortez
"""

import numpy as np
import cv2

cap = cv2.VideoCapture('dia.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
#detectShadows=False

while(1):
    ret, frame = cap.read()
    
    kernel = np.ones((5,5),np.uint8)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    if frame is None:
        break
    
    fgmask=fgbg.apply(frame)
    blur = cv2.GaussianBlur(fgmask, (5,5), 8)
    
    #cv2.imshow('fgmask', frame)
    #cv2.imshow('frame', blur)
    
    #removes false positives from background
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #removes false negarives on the contour
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    #shows pixels removed by opening
    #tophat = cv2.morphologyEx(fgmask, cv2.MORPH_TOPHAT, kernel)
    #shows pixels added by closing
    #blackhat = cv2.morphologyEx(fgmask, cv2.MORPH_BLACKHAT, kernel)
    cv2.imshow('open', opening)
    cv2.imshow('close', closing)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
