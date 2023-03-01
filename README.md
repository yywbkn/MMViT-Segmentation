# MMViT-seg

# 0.Preface

This repo holds the implementation code of the paper.

*This repository provides code for "MMViT-Seg: A lightweight transformer and CNN fusion network for COVID-19 segmentation" .



# 1.Introduction

If you have any questions about our paper, feel free to contact us ([Contact](#5-Contact)). 

And if you are using COVID-SemiSeg Dataset for your research, please cite this paper ([BibTeX](#4-citation)).



## 2. Proposed Methods

- **Preview:**

  Our proposed methods consist of two key points: 

  - MMViTSeg(Supervised learning with segmentation).
  - Multi query attention

#### 2.1.Usage

1. Train
   - In the first step, you can directly run the  `MyTrain_LungInf_MMViT.py` and just run it! 
2. Test
   - When training is completed, the weights will be saved in `./Snapshots/save_weights/MMViT-Seg/`. 
   - Assign the path `--pth_path` of trained weights and `--save_path` of results save and in `MyTest_LungInf_MMViT_for.py`.
   - Just run it and results will be saved in `./Results/Lung infection segmentation/MMViT-Seg'

### 2.2.  MMViT-Seg+ Multi-class

#### 2.2.1. Usage

1. Train
   - Just run `MyTrain_MulCls_MMViTSeg.py`

2. Test

- When training is completed, the weights will be saved in `./Snapshots/save_weights/MulCls_MMViTSeg/`. 

- Assigning the path of weights in parameters `snapshot_dir` and run `MyTest_MulCls_MMViTSeg.py`.

  

# 3. Citation

Please cite our paper if you find the work useful: 

```
@article{Yang2023,
title={MMViT-Seg: A lightweight transformer and CNN fusion network for COVID-19 segmentation},
author={Yang, Yuan and Zhang, Lin and Ren, Lei and Wang, Xiaohan},
journal={Computer Methods and Programs in Biomedicine},
year={2023}}
```

## 4. Citation

Please cite this paper: 

```
@article{fan2020inf,
```

  	title={Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Images},
  	author={Fan, Deng-Ping and Zhou, Tao and Ji, Ge-Peng and Zhou, Yi and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
  	journal={IEEE TMI},
  	year={2020}
	}



### 5. Contact

If you have any questions about our paper,  Feel free to email me(yangyuan@buaa.edu.cn) . 