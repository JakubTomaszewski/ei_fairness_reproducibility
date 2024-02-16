# Reproducibility Study: Equal Improvability: A New Fairness Notion Considering the Long-term Impact (ICLR 2023 Poster)

## Reproducibility study
This reproducibility study aims to evaluate the robustness of Equal Improvability (EI) - an effort-based framework for ensuring long-term fairness introduced in the paper "Equal Improvability: A New Fairness Notion Considering the Long-term Impact". To this end, we seek to analyze the three proposed EI-ensuring regularization techniques, i.e. Covariance-based, KDE-based, and Loss-based EI. This paper got a poster presentation at ICLR 2023.
Links for the [original paper](https://openreview.net/forum?id=dhYUMMy0_Eg).

Our findings largely substantiate the initial assertions, demonstrating EI’s enhanced performance over Empirical Risk Minimization (ERM) techniques on various test datasets. Furthermore, while affirming the long-term effectiveness in fairness, the study also uncovers challenges in resilience to overfitting, particularly in highly complex models.
Building upon the original study, the experiments were extended to include a new dataset and multiple sensitive attributes. These additional tests further demonstrated the effectiveness of the EI approach, reinforcing its continued success. Our study highlights the importance of adaptable strategies in AI fairness, contributing to the ongoing discourse in this field of research.

The reproducibility study and further code extensions were done by students of University of Amsterdam:

- Berkay Chakar
- Amina Izbassar
- Mina Janicijevic
- Jakub Tomaszewski

The reproducibility notebooks can be found in the [notebooks/reproducibility](notebooks/reproducibility) directory, whereas the multivariate and additional dataset experiments can be found in [notebooks/multivariate_study](notebooks/multivariate_study) and [notebooks/additional_dataset](notebooks/additional_dataset) respectively.

## Original paper
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
