# Reproducibility Study: Equal Improvability: A New Fairness Notion Considering the Long-term Impact (Published at TMLR 2024)

This work was published at the Transactions on Machine Learning Research (TMLR) 2024. The paper can be found [here](https://openreview.net/forum?id=Yj8fUQGXXL).

## Overview

This reproducibility study aims to evaluate the robustness of Equal Improvability (EI) - an effort-based framework for ensuring long-term fairness introduced in the paper "Equal Improvability: A New Fairness Notion Considering the Long-term Impact", which got a poster presentation at ICLR 2023 (see [original paper](https://openreview.net/forum?id=dhYUMMy0_Eg) and [video](https://recorder-v3.slideslive.com/?share=80966&s=eb8caaef-2818-4e2e-b687-e8d5eac09800)). Additionally, building upon the original codebase we adapt the proposed framework to include multiple sensitive attributes and evaluate its performance on a new dataset.

To this end, we seek to analyze the three proposed EI-ensuring regularization techniques, i.e. Covariance-based, KDE-based, and Loss-based EI. Our findings largely substantiate the initial assertions, demonstrating EI’s enhanced performance over Empirical Risk Minimization (ERM) techniques on various test datasets. Furthermore, while affirming the long-term effectiveness in fairness, the study also uncovers challenges in resilience to overfitting, particularly in highly complex models. The conducted experiments indicate that the EI approach remains effective in ensuring fairness across multiple sensitive attributes, further underscoring its adaptability and robustness and highlighting its potential for broader applications.

The reproducibility study and further code extensions were done by students of [University of Amsterdam](https://www.uva.nl/), namely:

- Berkay Chakar
- Amina Izbassar
- Mina Janićijević
- Jakub Tomaszewski

The original repository was authored by [Ozgur Guldogan](https://guldoganozgur.github.io)\*, [Yuchen Zeng](https://yzeng58.github.io/zyc_cv/)\*, [Jy-yong Sohn](https://itml.yonsei.ac.kr/professor), [Ramtin Pedarsani](https://web.ece.ucsb.edu/~ramtin/), and [Kangwook Lee](https://kangwooklee.com).

![Poster](poster.png)

## Installation instructions

First install the repo.

To create virtual environment:

```shell
cd ei_fairness_reproducibility
conda env create --file environment.yml
```

Note: Assuming you installed TeX. Since, the figures require TeX. To get TeX, you can follow the instructions at that [link](https://www.latex-project.org/get/).

## Reproducing Results

You can quickly replicate the outcomes described in the paper by referring to the notebooks located in the `notebooks/` directory.

## Citation

If you use this code, please consider citing our paper:

```
@article{
chakar2024reproducibility,
title={Reproducibility Study: Equal Improvability: A New Fairness Notion Considering the Long-Term Impact},
author={Berkay Chakar and Amina Izbassar and Mina Jani{\'c}ijevi{\'c} and Jakub Tomaszewski},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=Yj8fUQGXXL},
note={Reproducibility Certification}
}
```
