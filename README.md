# WWGAN (Worm Wasserstein GAN)

This is the official Pytprch implementation of WWGAN model presented in [https://ieeexplore.ieee.org/document/9760052](https://ieeexplore.ieee.org/document/9760052)

> Worm Wasserstein Generative Adversarial Network(WWGAN) is a Generative Adversarial Network (GAN) for time-series data augmentation. WWGAN builds upon two [WGAN-GP](https://github.com/caogang/wgan-gp) by constructing the WGAN-GP into a recurrent structure like RNN to improve its data augmentation ability on time-series data. 

> WWGAN can generate synthetic time-series data that carry realistic intrinsic patterns with the original data and expands a small sample without prior knowledge or hypotheses. Which means it can augment one-dimensional time-series data.

The following is the architecture of WWGAN model.

![WWGAN architecture](/WWGAN%20architecture.png)

## Publication

- Title: Small Sample Reliability Assessment With Online Time-Series Data Based on a Worm Wasserstein Generative Adversarial Network Learning Method

- Citation: Bo Sun, Zeyu Wu, Qiang Feng et al., "Small Sample Reliability Assessment With Online Time-Series Data Based on a Worm Wasserstein Generative Adversarial Network Learning Method," in IEEE Transactions on Industrial Informatics, vol. 19, no. 2, pp. 1207-1216, Feb. 2023, doi: 10.1109/TII.2022.3168667.

## Requirements

- Python 3.8.10
- Pytorch 1.9.0
- Numpy 1.21.2
- Matplotlib 3.5.1
- Pandas 1.2.4
- Scipy 1.6.2
- Seaborn 0.11.1

It is recommended to use a virtual environment for running this project and installing the required dependencies in it by using the [requirements.txt](/requirements.txt) file.

## 