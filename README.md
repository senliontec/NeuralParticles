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
Further Informations: https://ge.in.tum.de/publications/2020-iclr-prantl/

This repository contains the code for our ICLR paper 
'[Tranquil Clouds: Neural Networks for Learning Temporally Coherent Features in Point Clouds](https://openreview.net/forum?id=BJeKh3VYDH)'. 
The code is modified from [PointNet++](https://github.com/charlesq34/pointnet2) 
and [PUNet](https://github.com/yulequan/PU-Net/blob/master/README.md).

## Usage

### Installation
1.  ss
2.  ss

### Generate Data

### Run Training

### Run Evaluations

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
