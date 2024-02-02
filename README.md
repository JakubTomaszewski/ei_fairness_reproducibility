# Reproducibility Study: Equal Improvability: A New Fairness Notion Considering the Long-term Impact (ICLR 2023 Poster)

This reproducibility study aims to evaluate the robustness of Equal Improvability (EI) - an effort-based framework for ensuring long-term fairness introduced in the paper "Equal Improvability: A New Fairness Notion Considering the Long-term Impact". To this end, we seek to analyze the three proposed EI-ensuring regularization techniques, i.e. Covariance-based, KDE-based, and Loss-based EI. This paper got a poster presentation at ICLR 2023.
Links for [paper](https://openreview.net/forum?id=dhYUMMy0_Eg) and [video](https://recorder-v3.slideslive.com/?share=80966&s=eb8caaef-2818-4e2e-b687-e8d5eac09800).

Our findings largely substantiate the initial assertions, demonstrating EI’s enhanced performance over Empirical Risk Minimization (ERM) techniques on various test datasets. Furthermore, while affirming the long-term effectiveness in fairness, the study also uncovers challenges in resilience to overfitting, particularly in highly complex models.
Building upon the original study, the experiments were extended to include a new dataset and multiple sensitive attributes. These additional tests further demonstrated the effec- tiveness of the EI approach, reinforcing its continued success. Our study highlights the importance of adaptable strategies in AI fairness, contributing to the ongoing discourse in this science field.

The reproducibility study and further code extensions were done by students of [University of Amsterdam:](https://www.uva.nl/)

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
cd ei_fairness
conda env create --file environment.yml
```

Note: Assuming you installed TeX. Since, the figures require TeX. To get TeX, you can follow the instructions at that [link](https://www.latex-project.org/get/).

## Reproducing Results

You can quickly replicate the outcomes described in the paper by referring to the notebooks located in the `notebooks/` directory.

## Citation

If you use this code, please consider citing our paper:

```
@inproceedings{
guldogan2023equal,
title={Equal Improvability: A New Fairness Notion Considering the Long-term Impact},
author={Ozgur Guldogan and Yuchen Zeng and Jy-yong Sohn and Ramtin Pedarsani and Kangwook Lee},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=dhYUMMy0_Eg}
}
```

## Contact

Please let us know if you have further questions or comments, you can reach out at `ozgurguldogan@ucsb.edu`, `yzeng58@wisc.edu`.
