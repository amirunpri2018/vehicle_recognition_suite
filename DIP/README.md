# Vehicle detection, counting and classification using simple digital image processing techniques, python and openCV3

### Victor Cortez Trigueiro de Oliveira
### victor.cto@hotmail.com

## Introduction
The goal of this project was to demonstrate digital image processing (DIP) algorithms on practice, we achieved this by using traffic video feed to detect, count and classify the vehicles passing by. To each frame, we apply a series of DIP methods in sequence, until we get the finished result. the complete algorithm can be found in the file [withclassifier.py](https://github.com/vicforpublic/vehicle_recognition_suite/blob/master/DIP/withclassifier.py). The video file we will be working with was ceded by Professor (TODO) at Universidade Federal do Rio Grande do Norte. Below an example of frame.
![imgs/orig.png](TODO)

## pipeline
We start by getting the ROI of the frame, in this case, we want the trapezoidal portion that covers the road section of the image.
![imgs/roi.png](TODO)
