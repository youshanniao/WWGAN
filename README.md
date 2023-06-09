# WWGAN (Worm Wasserstein GAN)

This is the official Pytorch implementation of WWGAN model presented in [https://ieeexplore.ieee.org/document/9760052](https://ieeexplore.ieee.org/document/9760052)

> Worm Wasserstein Generative Adversarial Network(WWGAN) is a Generative Adversarial Network (GAN) for time-series data augmentation. WWGAN builds upon two [WGAN-GP](https://github.com/caogang/wgan-gp) by constructing the WGAN-GP into a recurrent structure like RNN to improve its data augmentation ability on time-series data. 

> WWGAN can generate synthetic time-series data that carry realistic intrinsic patterns with the original data and expands a small sample without prior knowledge or hypotheses. Which means it can augment one-dimensional time-series data.

The following is the architecture of WWGAN model.


![WWGAN architecture](/WWGAN%20architecture.png)

## Requirements

- Python 3.8.10
- Pytorch 1.9.0
- Numpy 1.21.2
- Matplotlib 3.5.1
- Pandas 1.2.4
- Scipy 1.6.2
- Seaborn 0.11.1

A virtual environment is recommended for running this project. The required dependencies are listed in [requirements.txt](/requirements.txt).

## Quick start

1. Run the [WWGAN_toy.py](/WWGAN_toy.py) to learn the toy dataset.
2. Run the [WWGAN_verification.py](/WWGAN_verification.py) to drow some figs to verify the augmentation results.

> The real sample and fake(generated) data:
![Real & fake verification](/Real%20%26%20Fake.png)

> The fitting verification:
![Real & fake verification](/FinalOut.png)

## Customization 

- The WWGAN model can be customized by modifying the [model.py](/model.py).
- The input time-series data can be changed into other datasets. The **Hyperparameters** should be modified accordingly.

## Publication

If you found this code useful, please cite our paper:

- Title: Small Sample Reliability Assessment With Online Time-Series Data Based on a Worm Wasserstein Generative Adversarial Network Learning Method.

- Citation: Bo Sun, Zeyu Wu, Qiang Feng et al., "Small Sample Reliability Assessment With Online Time-Series Data Based on a Worm Wasserstein Generative Adversarial Network Learning Method," in IEEE Transactions on Industrial Informatics, vol. 19, no. 2, pp. 1207-1216, Feb. 2023, doi: 10.1109/TII.2022.3168667.

```tex
@ARTICLE{9760052,
  author={Sun, Bo and Wu, Zeyu and Feng, Qiang and Wang, Zili and Ren, Yi and Yang, Dezhen and Xia, Quan},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Small Sample Reliability Assessment With Online Time-Series Data Based on a Worm Wasserstein Generative Adversarial Network Learning Method}, 
  year={2023},
  volume={19},
  number={2},
  pages={1207-1216},
  doi={10.1109/TII.2022.3168667}}
```

## Acknowledgements

This repository is based on the code published in [WGAN-GP](https://github.com/caogang/wgan-gp).