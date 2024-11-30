# Correlation-SFSwin
Cross-modal object detection based on PyTorch and OpenMMLab MMDetection.

# Installation
### Install PyTorch

(Please refer to the installation guide on the [official PyTorch website](https://pytorch.org/get-started/locally/))
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
### Install MMCV
(Please refer to the installation guide on the [official MMDetection website](https://mmdetection.readthedocs.io/en/latest/get_started.html))
```
pip install -U openmim
mim install mmcv
```
### Clone this GitHub repository
```
git clone https://github.com/Bazenr/Correlation-SFSwin.git
```
### Install this GitHub repository
```
cd [path to istalled folder]
pip install -v -e .
```
### Install third-party libraries
```
pip install tifffile
```
Optional:
```
pip install sahi
pip install openpyxl
pip install xlsxWriter
```

# Links
|**Content**|**Link**|
|:--------|:-------------|
|**Open source dataset**|https://www.kaggle.com/datasets/bazenr/rgb-hsi-rgb-nir-municipal-solid-waste|
|**Related paper**|[Plastic waste identification based on multimodal feature selection and cross-modal Swin Transformer](https://doi.org/10.1016/j.wasman.2024.11.027)|

# Citation
```
@article{JI202558,
title = {Plastic waste identification based on multimodal feature selection and cross-modal Swin Transformer},
author = {Tianchen Ji and Huaiying Fang and Rencheng Zhang and Jianhong Yang and Zhifeng Wang and Xin Wang},
journal = {Waste Management},
volume = {192},
pages = {58-68},
year = {2025},
issn = {0956-053X},
doi = {https://doi.org/10.1016/j.wasman.2024.11.027},
url = {https://www.sciencedirect.com/science/article/pii/S0956053X24005841},
}
```
