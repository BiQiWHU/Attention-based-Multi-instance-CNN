# Attention-based-Multi-instance-CNN
Spatial/Channel-spatial attention based multi-instance CNN for image classification

# Brief Introduction
The combination of multiple instance learning and ConvNets in a trainable manner, also known as deep MIL, has drawn increasing attention in the past few years.

In the past few years, we have studied deep MIL and explored a variety of solutions to adapt MIL into existing ConvNets.

This Github project is a summarization of our former works, and we provided three types of MI-CNN on the backbone of:
 - Our designed light-weight DenseNet structure
 - AlexNet with pre-trained parameters on ImageNet
 - VGG-16 with pre-trained parameters on ImageNet

# Developing Environment

Python > 3.5

Tensorflow > 1.6

OpenCV > 3

Numpy > 1.16

# Implementation Details
The implementation details are provided as below.

- For MIL with DC-Net backbone, the code can be directly implemented by following steps.

Step 1, download the image classification benchmarks, and put them in the same file direction path.

Step 2, run tfdata.py to generate the tfrecord format file for training and testing. Remember to re-set the input image size of your own dataset.

Step 3, run DenseRS.py file to train your model. Generally, it needs to run three times, for second time and third time, the initial parameters from the last round needs to be reloaded, and the learning rate of the later round should be one tenth of the former one. Remember to adjust to other hyper-parameters according to your onw dataset. 

Step 4, run test.py for test accuracy report.

Generally speaking, the advantage of MIL with DC-Net backbone is its light-weight model, little prediction time and computation cost. However, its feature representation capability is usually weaker than models pre-trained from ImageNet. 

- For MIL with Pre-trained AlexNet or VGG, an additional step is needed at first so as to download the pre-trained model, which can be found here: https://github.com/tensorflow/models/tree/master/research/slim

Then, it is also a four-step process to run the code.

Step 1, download the image classification benchmarks, and put them in the same file direction path.

Step 2, run tfdata.py to generate the tfrecord format file for training and testing. Remember to re-set the input image size of your own dataset.

Step 3, run train_cam.py file to train your model. Remember to adjust to other hyper-parameters according to your onw dataset. 

Step 4, run test.py for test accuracy report.

The advantage to utilize pre-trained model as the initial parameters for deep MIL is it has stronger feature representation capability, and leads to higher performance. But, the model size and prediction time increases.


# References for Citation
If you find our project beneficial to your research, please remember to cite our below works from either deep MIL or dense connection CNN.

 - Deep MIL

[1] Qi Bi, Kun Qin, Zhili Li, Han Zhang, Kai Xu, Gui-Song Xia. A multiple-instance densely-connected ConvNet for aerial scene classification. TIP, 2020.

[2] Q Bi, K Qin, Z Li, H Zhang, K Xu. Multiple instance dense connected convolution neural network for aerial image scene classification. ICIP, 2019.


 - Light-weight dense connection structure

[1] Qi Bi, Kun Qin, Zhili Li, Han Zhang, Kai Xu, Gui-Song Xia. A multiple-instance densely-connected ConvNet for aerial scene classification. TIP, 2020.

[3] Q Bi, K Qin, H Zhang, Z Li, K Xu. A residual attention based convolution network for aerial scene classification, Neurocomputing, 2020.


# Other Resources

 - Self-implementation of dense connection structures:

[1] https://github.com/BiQiWHU/DenseNet121-VHRRSI

[2] https://github.com/BiQiWHU/DenseNet40-for-HRRSISC

 - Gated attention based deep MIL from ICML 2018:

[3] https://github.com/AMLab-Amsterdam/AttentionDeepMIL 

- Original design of DenseNet:

[4] https://github.com/liuzhuang13/DenseNet
