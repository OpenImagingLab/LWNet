# Learning-Based Lens Wavefront Aberration Recovery
### [Paper](https//)
Liqun Chen, Yuyao Hu, Jiewen Nie, Tianfan Xue and Jinwei Gu

## Environment requirements
The codes was tested on Windows 10, with Python and PyTorch. Packages required to reproduce the results can be found in `requirements.txt`. The following software / hardware is tested and recommended:
- numpy  
- tqdm
- Python >= 3.9
- matplotlib >= 3.5
- pytorch >= 2.0
- torchvision >= 0.15
- pandas >= 1.4

## File structure
This repository contains codes for LWNet.
```
LWNet
|   README.md
|   requirements.txt
|
|---Stage_I
|   |   sort_zernike_coefficient_save_PSF_mat.py
|   |   dataset_generator_mat.py
|   |   main_wavefront_PSF_CircleMSE_0.1TV.py
|   |   ...
|
|---Stage_II  
|   |---data  
|   |---model
|   |---results  
|   |   main_lwnet.py
|   |   main_lwnet_strict.py
|   |   statistic_zer_rule.py
|   |   ...
| 
```
`/Stage_I` and `/Stage_II` contain the codes for each stage. 

`/Stage_II/data` include input data (GT). 

`/Stage_II/models` include models for Stage_I and stage_II. 

`/Stage_II/results` store the optimization results.

## Test
To test LWNet and reproduce some results shown in the paper, following these steps:
- Run `/Stage_II/main_lwnet.py`. The outputs will be saved in `/Stage_II/results` folders
To test additional version:
- Run `/Stage_II/test.py`. The outputs will be saved in `/Stage_II/results` folders

## Train
To train Stage_I model from scratch, cuda is needed, follow these steps:

- Run `Sort_zernike_coefficient_save_PSF_mat.py` to generate .mat files, which are saved in  `/Stage_I/mat`
- Run `dataset_generator_mat.py` to generate dataset file from .mat files,  train and valid dataset are generated at the same time. Files are saved in  `/Stage_I/h5py`
- Run `main_wavefront_PSF_CircleMSE_0.1TV.py` to train supervised net by dataset, checkpoints will be saved in `model` folders
