# Multimodal Deep Learning for Robust RGB-D Object Recognition

## Requirements

- Pillow (Pillow requires an external library that corresponds to the image format)

## Description

This is an implementation of 'Multimodal Deep Learning for Robust RGB-D Object Recognition'.
It requires the training and validation dataset of following format:

* Each line contains one training example.
* Each line consists of two elements separated by space(s).
* The first element is a path to 256x256 RGB image.
* The second element is its groundtruth label from 0 to arbitrary.

The text format is equivalent to what Caffe uses for ImageDataLayer.

This example requires "mean file" which is computed by `compute_mean.py`.

This example also requires CaffeNet model 'bvlc_reference_faffenet.caffemodel' sited at http://dl.caffe.berkeleyvision.org/

So, you must to download its model before implement training.

The process to train is follow:
1) command 'python train_rgb_d.py' with color datas.
2) command 'python train_rgb_d.py' with depth datas.
3) command 'python train_full.py' with color datas and depth datas.
