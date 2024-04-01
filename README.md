# Unofficial implementation of UnDeepVO method

<b> UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning </b> [arxiv](https://arxiv.org/pdf/1709.06841.pdf)

UnDeepVO is able to estimate the 6-DoF pose of a monocular camera and the depth of its view by using deep neural networks. 
There are two salient features of the proposed UnDeepVO: one is the unsupervised deep learning scheme, and the other is the
absolute scale recovery. Specifically, we train UnDeepVO by using stereo image pairs to recover the scale but test it by
using consecutive monocular images. Thus, UnDeepVO is a monocular system. The loss function defined for training the
networks is based on spatial and temporal dense information. The experiments on KITTI dataset show our UnDeepVO achieves 
good performance in terms of pose accuracy.

![scheme](pictures/scheme.png)
