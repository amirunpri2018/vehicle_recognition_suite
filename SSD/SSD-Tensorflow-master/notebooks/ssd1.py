import numpy as np
import cv2
import time

import os
import math
import random

import tensorflow as tf

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization


# functions ===================================================================
# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes
#==============================================================================

# global constants ============================================================
#VID_DIRECTORY = "D:\\Users\\f202897\\Desktop\\vm-master\\Videos\\"
VID_DIRECTORY = "C:\\Users\\victor\\Desktop\\vm-master\\Videos\\"
ROI_CORNERS = np.array([[(1000,180),(300,180), (50,720), (1100,720)]], dtype=np.int32)

#==============================================================================

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)
# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)



cap = cv2.VideoCapture(VID_DIRECTORY+'tarde.mp4')

#benchmarking
timeslist=[]

framecount=0

while(1):
    millis1 = time.time()
    # Test on some demo image and visualize output.
    #path = '../demo/'
    #image_names = sorted(os.listdir(path))
    ret, frame = cap.read()
    width, heigth, channels = frame.shape
    
    if frame is None:
        print("none")
        break
    
    framecount+=1
    framecopy=frame.copy()
    
    #getting ROI ==============================================================
    # mask defaulting to black for 3-channel and transparent for 4-channel
    mask = np.zeros(frame.shape, dtype=np.uint8)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillConvexPoly(mask, ROI_CORNERS, ignore_mask_color)
    # apply the mask
    roi = cv2.bitwise_and(frame, mask)
    rectangle=cv2.boundingRect(ROI_CORNERS)
    #cv2.rectangle(roi,(rectangle[0],rectangle[1]),(rectangle[0]+rectangle[2],rectangle[1]+rectangle[3]),(0,0,255),3)
    x1=rectangle[0]
    x2=rectangle[0]+rectangle[2]
    y1=rectangle[1]
    y2=rectangle[1]+rectangle[3]
    roi=roi[y1:y2,x1:x2]
    roi=cv2.resize(roi, (heigth//2,width//2))

    #==========================================================================
    
    #img = mpimg.imread(path + image_names[-4])
    #img = mpimg.imread(path + 'orig.jpg')
    
    rclasses, rscores, rbboxes =  process_image(roi)
    
    # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    frameresult = visualization.plt_bboxes_cv(roi, rclasses, rscores, rbboxes)
    cv2.imshow('result', frameresult)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    millis2 = time.time()
    millis=millis2 - millis1
    timeslist.append(millis*1000)
    
cap.release()
cv2.destroyAllWindows()
mean = sum(timeslist)/len(timeslist)