# Hyper Volume Generative Adversarial Network - hGAN

Work in progress.

Replication of [Stabilizing GAN Training with Multiple Random Projections](https://arxiv.org/abs/1705.07831) and extension including training with multi-objective training via hyper volume maximization (based on [Multi-Objective Optimization for Self-Adjusting Weighted Gradient in Machine Learning Tasks](https://arxiv.org/abs/1506.01113))

## Requirements

- Python 3.6
- Pytorch 0.3.0
- [Cropped and aligned version of CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## To run

1. Create a directory named 'celebA' and unzip img_align_celeba.zip inside it
2. 

```
python train.py --ndiscrimiators 12
```


Collaborators: Joao Monteiro, Breandan Considine
