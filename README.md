# GFPNet
## A Single-view 3D Object Reconstruction using Generic Fitted Primitives
3D volumetric reconstruction from incomplete point clouds still remains one of the fundamental challenges in
the process of interpreting the environment, by machines. One solution is to register an a-priori defined shape (i.e. a CAD model) onto the incomplete point cloud of the object. The subsequent challenge that arises is to identify the most similar shape of the object from a shapes database. In this paper, we propose a volumetric reconstruction apparatus that uses so-called Generic Primitive Primitives (GFP) to abstract the large variety of shapes that an object may
have. We use a kernel-based deformation technique to fit a GFP to real-world objects, where the kernel is encoded within
the layers of a Deep Neural Network (DNN). The objective is to transfer the particularities of the perceived object to
the raw volume represented by the generic primitive. We show that GPFNet outperforms competing algorithms on 3D volume reconstruction challenges, by being tested on the ModelNet and KITTI benchmarking datasets. GFPNet is compared with a
baseline approach, as well as with state-of-the-art data-driven approaches for volumetric reconstruction.

![GFPNet workflow](https://github.com/RovisLab/GFPNet/blob/master/images/block_diagram.png)

## Getting started
Implemented and tested on Windows 10 x64, Python 3.5 and Tensorflow 1.12.0rc0

1. Clone this repo
``` 
git clone git@github.com:RovisLab/GFPNet.git 
```

2. Install Python dependencies
``` 
pip3 install -r requirements.txt
```

## Training
### Dataset
To train on [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):
    - Download the data and place it in your dataset folder at ~/kitti/
1. The folder should look something like the following:
   * kitti
    * calib
    * label_2
    * velodyne
    * image_2
