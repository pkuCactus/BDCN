## [Bi-Directional Cascade Network for Perceptual Edge Detection(BDCN)](https://arxiv.org/pdf/1902.10903.pdf)

This paper proposes a Bi-Directional Cascade Network for edge detection. By introducing a bi-directional cascade structure to enforce each layer to focus on a specific scale, BDCN trains each network layer with a layer-specific supervision. To enrich the multi-scale representations learned with a shallow network, we further introduce a Scale Enhancement
Module (SEM). Here are the code for this paper.


### Prerequisites

- pytorch >= 0.2.0
- numpy >= 1.11.0
- pillow >= 3.3.0

### Train and Evaluation

1. Clone this repository to local
```shell
git clone https://github.com/pytorch/pytorch.git
```

2. Download the imagenet pretrained vgg16 pytorch model [vgg16.pth]() or the caffemodel from the [model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) and then transfer to pytorch version. You also can download our pretrained model for only evaluation.

3. Download the dataset to the local folder

4. running the training code and test code

### Pretrained models

BDCN model for BSDS500 dataset is available[1]()

BDCN model for NYUDv2 datset is available[RGB]() and [depth]()


The pretrained model will be updated soon.


