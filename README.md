# DI4SLF
PyTorch implementation of ICCV 2021 paper: "[Learning Dynamic Interpolation for Extremely Sparse Light Fields with Wide Baselines](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Learning_Dynamic_Interpolation_for_Extremely_Sparse_Light_Fields_With_Wide_ICCV_2021_paper.pdf)"

## Requrements
- Python 3.7.4
- Pytorch 1.6.0

## Train and Test
### Data
We provide MATLAB code for preparing the training and test data in the folder ./LFData. We estimate the optical flow of source views using a pre-trained [optical-flow model](https://drive.google.com/file/d/1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM/view?usp=share_link) provided by [RAFT](https://github.com/princeton-vl/RAFT). Before generating the training and testing datasets, please first put the estimated optical flow of source views in the folder ./LFData/flow_source in the .mat format with the shape [num_lf, an_h, an_w, h, w], where num_lf, [an_h, an_w], [h, w] are the number, angular resolutions, and spatial resolutions of sparse lfs, respectively.
Our training dataset can be downloaded from [here](https://drive.google.com/file/d/1QhQlOj9WgBiq-EptFPBM5I6Ds1iXdQNS/view?usp=sharing), and the testing dataset can be downloaded from [here](https://drive.google.com/file/d/1mmhP_1QbOliheNF3co4h7AtGNANTKlOL/view?usp=sharing).

### Test
The testing codes are in the folder ./test
- model: the trained model which is in this subfolder;
- Inference:
```
python lfr_test.py
```

### Train
The testing codes are in the folder ./train
- Training:
```
python lfr_train.py
```
