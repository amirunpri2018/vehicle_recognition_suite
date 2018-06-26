# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:57:48 2018

@author: victor cortez
"""

import numpy as np
import cv2

import time

# functions ===================================================================
def predict_next_position(pointpositionhistory):
    numofpos=len(pointpositionhistory)
    if numofpos == 1:
        predicted_next_position=1
    return predicted_next_position

def point_existed(p,points_history):
    #se distancia entre ponto e algum dos elementos de lasthull.predict for menor que 10, True
    distlist=[]
    if len(points_history)==0:
        return False
    for el in points_history:
        distlist.append(np.linalg.norm(np.subtract(p,el[len(el)-1])))
    minval=min(distlist)
    if minval > 40:
        return False
    else:
        return True
    
def get_point_pos(p,points_history):
    #se distancia entre ponto e algum dos elementos de lasthull.predict for menor que 10, True
    distlist=[]
    for el in points_history:
        distlist.append(np.linalg.norm(np.subtract(p,el[len(el)-1])))
    minpos=distlist.index(min(distlist))
    return minpos

def point_crossed_line_and_deleted(points_history,line,orientation):
    if len(points_history) <= 1:
        return False
    up=False
    down=False
    if orientation == 'vertical':
        for point_history in points_history:
            up=False
            down=False
            for point in point_history:
                if point[1] > line[0][1]:
                    up=True
                else:
                    down=True
                if up and down:
                    points_history.remove(point_history)
                    return True
    return False
#==============================================================================

#VID_DIRECTORY = "D:\\Users\\f202897\\Desktop\\vm-master\\Videos\\"
VID_DIRECTORY = "C:\\Users\\victor\\Desktop\\vm-master\\Videos\\"
#area of interest
RECT_Y1 = 180
RECT_Y2 = 720
RECT_X1 = 160
RECT_X2 = 1120
# points = x,y (l-r, t-b)
ROI_CORNERS = np.array([[(800,180),(400,180), (40,720), (1200,720)]], dtype=np.int32)
LINE_POINTS = [(100,600),(1000,600)]
ORIENTATION='vertical'

cap = cv2.VideoCapture(VID_DIRECTORY+'dia.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
#detectShadows=False

timeslist=[]
framecount=0

#each element contains the history of each center detected
points_history=[]
while(1):
    
    millis1 = time.time()
    
    ret, frame = cap.read()
    
    if frame is None:
        print("none")
        break
    framecount+=1
    
    #getting ROI ==============================================================
    # mask defaulting to black for 3-channel and transparent for 4-channel
    mask = np.zeros(frame.shape, dtype=np.uint8)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillConvexPoly(mask, ROI_CORNERS, ignore_mask_color)
    # apply the mask
    roi = cv2.bitwise_and(frame, mask)
    #==========================================================================

    #kernels ==================================================================
    ksize=5
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
    OPEN_KERNEL=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    #kernel = np.ones((5,5),np.uint8)
    #kernel[0,0] = 0
    #kernel[0,4] = 0
    #kernel[4,0] = 0
    #kernel[4,4] = 0
    ksize=23
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
    DIL_KERNEL=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
    
    KERNEL5=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    KERNEL7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    KERNEL11=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    KERNEL19=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
    KERNEL23=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23))
    KERNEL27=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(27,27))
    KERNEL29=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(29,29))
    #==========================================================================
    
    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(roi, (5,5), 8)
    mblur = cv2.medianBlur(roi,5)
    
    fgmask=fgbg.apply(roi)
    #fgmaskblur=fgbg.apply(blur)
    fgmaskmblur=fgbg.apply(mblur)

    #fgmask[np.where(fgmask<=254)] = [0]
    #ret,fgmaskblur = cv2.threshold(fgmaskblur,127,255,cv2.THRESH_BINARY)
    ret,fgmaskmblur = cv2.threshold(fgmaskmblur,127,255,cv2.THRESH_BINARY)
    ret,fgmask = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)
    
    # morphology manipulation  ================================================
    #removing false positives =====
    #erosion = cv2.morphologyEx(fgmask,cv2.MORPH_ERODE, kernel)
    #removes false positives from background
    #opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #openingblur = cv2.morphologyEx(fgmaskblur, cv2.MORPH_OPEN, kernel)
    openingmblur = cv2.morphologyEx(fgmaskmblur, cv2.MORPH_OPEN, KERNEL7)
    #blobedimg = cv2.morphologyEx(openingmblur,cv2.MORPH_ERODE,KERNEL5)
    #============================
    
    #removing false negarives ====
    #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #shows pixels removed by opening
    #tophat = cv2.morphologyEx(fgmask, cv2.MORPH_TOPHAT, kernel)
    #shows pixels added by closing
    #blackhat = cv2.morphologyEx(fgmask, cv2.MORPH_BLACKHAT, kernel)
    #dilation = cv2.morphologyEx(opening,cv2.MORPH_DILATE, np.ones([7,7], np.uint8))
    #dilationblur = cv2.morphologyEx(openingblur,cv2.MORPH_DILATE, np.ones([7,7], np.uint8))
    dilationmblur = cv2.morphologyEx(openingmblur,cv2.MORPH_DILATE, KERNEL29)
    #closingmblur = cv2.morphologyEx(openingmblur,cv2.MORPH_CLOSE,KERNEL7)
    #dilationmblur = cv2.morphologyEx(closingmblur,cv2.MORPH_DILATE, KERNEL27)
    morphresult = dilationmblur
    #==========================================================================
    
    
    #filling blobs ============================================================
    
    #openingblur
    h, w = openingmblur.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill=openingmblur.copy()
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    openingfilled = openingmblur | im_floodfill_inv
    
    #mblur
    h, w = morphresult.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill=morphresult.copy()
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    filledimg = morphresult | im_floodfill_inv
    
    fillingresult = filledimg
    #==========================================================================
    
    # contours ================================================================
    image22, contours, hierachy = cv2.findContours(fillingresult.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagehull = image22.copy()
    hull = [cv2.convexHull(c) for c in contours]
    #clearing small blobs
    for index, el in enumerate(hull):
        area = cv2.contourArea(el)
        if area < 1500:
            del hull[index]
    
    blank=np.zeros(image22.copy().shape,np.uint8)
    #converting image back to 3 channels to display border colours
    image22=cv2.cvtColor(image22,cv2.COLOR_GRAY2BGR)
    #image23=cv2.cvtColor(image23,cv2.COLOR_GRAY2BGR)
    imagehull=cv2.cvtColor(imagehull,cv2.COLOR_GRAY2BGR)
    
    #drawing contours
    image22 = cv2.drawContours(image22,contours,-1,(0,255,0))
    blank=cv2.drawContours(blank,hull,-1,255,cv2.FILLED)
    #image23 = cv2.drawContours(image23,contours2,-1,(0,255,0))
    imagehull = cv2.drawContours(imagehull,hull,-1,(0,255,0),cv2.FILLED)
    contouringresult=blank
    #==========================================================================
    
    # counting ================================================================
    #para cada contorno, checar se ele ja existia no frame anterior. se nao,
    #aloca-lo na lista, se sim, prever o proximo passo dele.
    
    if framecount > 2:
        # tracking resulting blobs
        for index, el in enumerate(hull):
            M=cv2.moments(el)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            p=(cx,cy)
            cv2.circle(roi, p, 7, (0, 255, 255), -1)
            
            if point_existed(p,points_history):
                pos = get_point_pos(p,points_history)
                points_history[pos].append(p)
                    
            else:
                points_history.append([p])
    
    
    if point_crossed_line_and_deleted(points_history,LINE_POINTS,ORIENTATION):
        cv2.line(frame,LINE_POINTS[0],LINE_POINTS[1],(0,255,0),5)
    else:
        cv2.line(frame,LINE_POINTS[0],LINE_POINTS[1],(255,0,0),5)    
    #==========================================================================
    
    
    # display images ==========================================================
    #cv2.imshow('fgmask', frame)
    #cv2.imshow('frame', cnts)

    cv2.imshow('blob', contouringresult)
    cv2.imshow('frame', frame)
    cv2.imshow('morphresult', morphresult)
    #cv2.imshow('can', canny)
    #==========================================================================
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    millis2 = time.time()
    millis=millis2 - millis1
    timeslist.append(millis*1000)

cap.release()
cv2.destroyAllWindows()

mean = sum(timeslist)/len(timeslist)