# A-modular-sliding-window-object-detector-training-and-detection-system-in-Java
A modular sliding window object detector training and detection system in Java

One of the most successful types of object detection systems takes the form of the following algorithm:

1. Thousands, hundreds of thousands or millions of cropped images of the object category to be detected are gathered.
1. Negative images are collected.
1. With a given feature extraction method (on a given cropped window), features are extracted and a high dimensional feature vector is obtained. These vectors are collected from positive training images and sampled from negative images.
1. An initial classifier is trained.
1. This classifier is run on negative images to find the "hard negatives".
1. Features are extracted on these hard negatives and these vectors are added to the original dataset.
1. A new classifier is trained.
1. This round of hard negative mining and classifier training or updating may repeat for several rounds.
1. Then the object category detector is obtained.
1. At test or prediction time, given an unseen image, in order to detect objects of the trained category, the following algorithm is used.
1. A sliding window at all positions of the image at different scales of the image. Due to the space of all positions and scales being too large, a discretization approach is usually used.
1. At each sliding window position, feature extraction is performed and the trained classifier is applied to give a score for that window position and scale.
1. After repeating on the sampled sliding window position and scales, non-maximum suppression (including thresholding) is performed in order to get the final detection.

This projects implements all of the aforementioned steps to train any object category detector and use it to detect on any image. One of the most notable points about the system is that the system is highly modular using the principles of Object Oriented Programming (OOP), encapsulation, inheritance, composition, polymorphism, etc. Some of the highlights of the system are:

- Can train a detector for any object category of any given fixed aspect ratio of object.
- Can be given any set of cropped positive images and negative images.
- The feature extraction is completely modular and can easily be replaced with any algorithm by inheriting the base feature extraction class and overriding the relevant method.
- The classification component is also completely modular and can easily be replaced with any algorithm by inheriting the base classification class and overriding the relevant method.
- There are two levels of feature extraction for maximum efficiency: level-1 and level-2 feature extraction methods. Both of them are completely modular. Level-1 feature extraction works on the whole image basis and the outcome is an image-like spatial features with a possibly large number of channels. Level-2 feature extraction works on sliding window basis and outputs a feature vector which is the input to the classifier. These two levels of feature extraction subsume most of the cases of feature extraction methods in literature such as:
    - Histogram of Oriented Gradients (HOG)
    - Integral Channel Features
    - Local Binary Patterns (LBP)
    - Many types of Convolutional Neural Networks (CNN)
    - Dense SIFT
    - Raw pixel values
    - HOG followed by CNN
    - HOG + LBP
    - Informed haar-like features
    - Color histograms
    - Gradient histograms
    - Gradient features
    - Resized image + any of the above feature extraction schemes
- The non-maximum suppression method is completely modular and can easily be replaced with any algorithm by inheriting the base classification class and overriding the relevant method.
- Can train detectors, save them and load them in the future for object detection.
- Three different types of matrices can be used as the data structure to store images and other entities in the entire object detector training and detection pipeline: OpenCV matrix, [Matk](https://github.com/Kyaw-Kyaw-Htike/Comprehensive-pure-Java-matrix-classes-for-Computer-Vision-Machine-Learning-and-AI-applications) and [Matkc](https://github.com/Kyaw-Kyaw-Htike/Comprehensive-pure-Java-matrix-classes-for-Computer-Vision-Machine-Learning-and-AI-applications).
- Some feature extraction, classification and non-maximum suppression algorithms are given as part of the library. For level-1 feature extraction, raw pixel values is given, although any other can be easily used as mentioned before. For level-2 feature extraction, HOG is given as part of the library, although any can be used. For classification, Perceptron and two different implementations of AdaBoost is provided in the library and the user is free to come up with any classification algorithm. For non-maximum suppression algorithm, the famous greedy overlap suppression approach is given.
- The system is highly flexible and there are many hyper-parameters that can be set to tune the object detection system as desired.

https://kyaw.xyz/2017/12/18/modular-sliding-window-object-detector-training-detection-system-java

Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.



Dr. Kyaw Kyaw Htike @ Ali Abdul Ghafur



https://kyaw.xyz
