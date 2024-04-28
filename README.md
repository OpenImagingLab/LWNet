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
|---data  
|---model
|---results  
|   main_lwnet.py
|   statistic_zer_rule.py
|   ...
| 
```

`/data` include input data (GT). 

`/models` include models for Stage_I and stage_II. 

`/results` store the optimization results.

## Test
To test LWNet and reproduce some results shown in the paper:
- Run `main_lwnet.py`. The outputs will be saved in `results` folders
- Modify parameters in `configs/lwnet.yaml`, run `main_lwnet.py`. The outputs will be saved in `results` folders
