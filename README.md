# Tranquil Clouds: Neural Networks for Learning Temporally Coherent Features in Point Clouds
Lukas Prantl, Nuttapong Chentanez, Stefan Jeschke, and Nils Thuerey

## Introduction
Point clouds, as a form of Lagrangian representation, allow for powerful and 
flexible applications in a large number of computational disciplines. We propose 
a novel deep-learning method to learn stable and temporally coherent feature 
spaces for points clouds that change over time. We identify a set of inherent 
problems with these approaches: without knowledge of the time dimension, the 
inferred solutions can exhibit strong flickering, and easy solutions to suppress 
this flickering can result in undesirable local minima that manifest themselves 
as halo structures. We propose a novel temporal loss function that takes into 
account higher time derivatives of the point positions, and encourages mingling, 
i.e., to prevent the aforementioned halos. We combine these techniques in a 
super-resolution method with a truncation approach to flexibly adapt the size of 
the generated positions. We show that our method works for large, deforming 
point sets from different sources to demonstrate the flexibility of our approach.
For more information, please refer to the following: 
https://ge.in.tum.de/publications/2020-iclr-prantl/

An example generated with our method can be seen here: http://lukas.prantl.it/portfolio/tranquil-clouds/

This repository contains the code for our ICLR paper 
'[Tranquil Clouds: Neural Networks for Learning Temporally Coherent Features in Point Clouds](https://openreview.net/forum?id=BJeKh3VYDH)'. 
The code is modified from [PointNet++](https://github.com/charlesq34/pointnet2) 
and [PUNet](https://github.com/yulequan/PU-Net/blob/master/README.md).
The training data was generated using [Mantaflow](http://mantaflow.com)

## Usage

### Installation
1.  Clone repository
2.  Install Tensorflow 1.9.0 GPU with CUDA 9.0 and Python 3.6 (newer versions are not supported!)  
    `conda create -n tc python=3.6`  
    `conda activate tc`  
    `pip install tensorflow-gpu==1.9.0`
  
3.  Build the required TF tools  
    `cd neuralparticles/tensorflow; make`
4.  Install requirements with pip  
    `pip install matplotlib keras==2.2.4`

### Run Training
**2D Data**
*  Download data: ... 
*  Run training: `python -m train config config/ours.txt data 2D_data/`

**3D Data**
*  Download data: ...
*  Run training: `python -m train config config_3d/ours.txt data 3D_data/`

### Run Inference
**2D Data**
*   Download data: ... 
*   Run training: `python -m run config config/ours.txt data 2D_data/`

**3D Data**
*   Download data: ...
*   Run training: `python -m run config config_3d/ours.txt data 3D_data/`

### Visualization
We are using a special binary *.uni* file format to write out the generated data.
You can use [Mantaflow](http://mantaflow.com) or an online available viewer to visualize the data:
http://lukas.prantl.it/portfolio/webgl-viewer/

## Citation

If our work is useful for your research, please consider citing:

    @inproceedings{
        Prantl2020Tranquil,
        title={Tranquil Clouds: Neural Networks for Learning Temporally Coherent Features in Point Clouds},
        author={Lukas Prantl and Nuttapong Chentanez and Stefan Jeschke and Nils Thuerey},
        booktitle={International Conference on Learning Representations},
        year={2020},
        url={https://openreview.net/forum?id=BJeKh3VYDH}
    }

### Questions

Please contact 'lukas.prantl@tum.de'
