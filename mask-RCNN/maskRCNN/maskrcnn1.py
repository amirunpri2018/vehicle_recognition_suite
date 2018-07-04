import os
import sys
#import skimage.io
#import matplotlib.pyplot as plt
import cv2
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize_cv
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

import numpy as np
import time

# classes =====================================================================
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
#==============================================================================

# global constants ============================================================
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
VID_DIRECTORY = "C:\\Users\\victor\\Desktop\\vm-master\\Videos\\"
#VID_DIRECTORY = "D:\\Users\\f202897\\Desktop\\vm-master\\Videos\\"
# points = x,y (l-r, t-b)
ROI_CORNERS = np.array([[(800,180),(400,180), (40,720), (1200,720)]], dtype=np.int32)

#==============================================================================


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]

#filename = os.path.join(IMAGE_DIR, 'van.png')

#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#image = skimage.io.imread(filename)

cap = cv2.VideoCapture(VID_DIRECTORY+'tarde.mp4')

#benchmarking
timeslist=[]

framecount=0

while(1):
    millis1 = time.time()

    ret, frame = cap.read()
    
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
    #==========================================================================


    # Run detection
    results = model.detect([roi], verbose=1)
    
    # Visualize results
    r = results[0]
    frameresult=visualize_cv.display_instances(roi, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])
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