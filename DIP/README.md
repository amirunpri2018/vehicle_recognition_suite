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

Then, we use an algorithm that compares the actual frame with the frames before using the Improved adaptive gaussian mixture model for background subtraction (MOG2)

>The Improved Adaptative Gaussian mixture model is an algorithm based on bayesian probability that decides, based on previous values in the image, if a certain pixel belongs to background or foreground

After removing the background, we end up with white pixels representing the moving pixels, and also noise from the enviroment. To filter out the noise, we use the opening algorithm, and then dilation, to accentuate the remaining pixels.

> The opening algorithm is used to filter noise out of an image, usually it consists of a series of erosions and dilations. In this case, it was verified to be better than the erosion and dilation alone.

Hopefully having only white mass on passing-by vehicles now, we fill any black patches left in the white islands, trace their contours, get the convex hull of said contours, and also an enclosing rectangle.

> A convex hull is the smallest convex polygon that can be made from a set of points. We rather work with a convex shape because the center of mass might be affecte by a shape with too much variance.

After obtaining the center of mass of each polygon, we finally track each center by analysing which center was the closest in the last frame, and when a center crosses the line, we get the rectangular patch that corresponds to that contour, classify the vehicle using a simple neural net trained with keras using the CIFAR-100 dataset, and count that a car has passed.