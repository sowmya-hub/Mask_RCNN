### IMAGE SEGMENTATION: Mask R-CNN for Object Detection and Segmentation

**Abstract**

Image segmentation is the process of individually identifying and labeling every pixel in an image, where each pixel having the same label shares certain characteristics. It can detect an object at a granular level and it can identify the shape of that object also.

It is an advanced and more accurate way of detecting an object’s edge and shape detection. Image segmentation divides an image into different partitions known as segments. This collection of segments are represented by a mask or a labeled image. In this way, we can process only the important segments instead of the entire image.

Deep learning-based image segmentation: In this approach convolutional neural networks are used to segment each object instance in an image. MASK-RCNN is a popular algorithm for DNN based image segmentation.

In this project, we are going to build an image segmentation model using the Mask RCNN pre-trained model using OpenCV.

OpenCV is a free open-source computer vision library. OpenCV has an inbuilt solution to run DNN models. That’s why we don’t need any other deep learning framework to build this project.

Mask RCNN is a deep learning model for image segmentation problems. It can separate different images in an image or video by giving their bounding box, classes, and corresponding binary image mask.

Mask RCNN built with Faster RCNN. F-RCNN has two outputs for each candidate, a class label and a bounding box. In addition, a 3rd branch is added to the model that outputs the object mask. The third branch works parallel with the existing branch for bounding box recognition.
