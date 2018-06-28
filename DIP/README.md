# Vehicle detection, counting and classification using simple digital image processing techniques, python and openCV3

### Victor Cortez Trigueiro de Oliveira
### victor.cto@hotmail.com

## Introduction
The goal of this project was to demonstrate digital image processing (DIP) algorithms on practice, we achieved this by using traffic video feed to detect, count and classify the vehicles passing by. To each frame, we apply a series of DIP methods in sequence, until we get the finished result. the complete algorithm can be found in the file [withclassifier.py](https://github.com/vicforpublic/vehicle_recognition_suite/blob/master/DIP/withclassifier.py). The video file we will be working with was ceded by Professor (TODO) at Universidade Federal do Rio Grande do Norte. Below an example of frame.
![imgs/orig.png](TODO)

## pipeline
We start by getting the ROI of the frame, in this case, we want the trapezoidal portion that covers the road section of the image.
![imgs/roi.png](TODO)

> The ROI is taken by creating another, black image, and filling a shape with the vertices we want for the ROI with non black pixels. we then perform an AND operation between each pixel in the resulting image and the original image.

We then blur the image using the median filter. This results in less noise in later steps of the processing.

> The median filter works by getting all pixel values in each step of the sliding window and ordering them. The result will be the median value.

Then, we use an algorithm that compares the actual frame with the frames before using the Improved adaptive gaussian mixture model for background subtraction (MOG2)TODO
