## Introduction
This is the official code of [Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation](!https://arxiv.org/abs/2012.15175).

This repo is built on [Bottom-up-Higher-HRNet](!https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation.git).

## Main Results
### Results on COCO val2017 without multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------| 
| HigherHRNet        | HRNet-w32  | 512      |  28.6M  | 47.9   | 67.1  | 86.2  |  73.0  |  61.5  |  76.1  | 
| HigherHRNet + SWAHR| HRNet-w32  | 512      |  28.6M  | 48.0   | 68.9  | 87.8  |  74.9  |  63.0  |  77.4  | 
| HigherHRNet        | HRNet-w48  | 640      |  63.8M  | 154.3  | 69.9  | 87.2  |  76.1  |  65.4  |  76.4  |
| HigherHRNet + SWAHR| HRNet-w48  | 640      |  63.8M  | 154.6  | 70.8  | 88.5  |  76.8  |  66.3  |  77.4  |


### Results on COCO val2017 *with* multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------| 
| HigherHRNet        | HRNet-w32  | 512      |  28.6M  | 47.9   | 69.9  | 87.1  |  76.0  |  65.3  |  77.0  | 
| HigherHRNet + SWAHR| HRNet-w32  | 512      |  28.6M  | 74.8   | 71.4  | 88.9  |  77.8  |  66.3  |  78.9  | 
| HigherHRNet        | HRNet-w48  | 640      |  63.8M  | 154.3  | 72.1  | 88.4  |  78.2  |  67.8  |  78.3  |
| HigherHRNet + SWAHR| HRNet-w48  | 640      |  63.8M  | 154.6  | 73.2  | 89.8  |  79.1  |  69.1  |  79.3  |

### Results on COCO test-dev2017 without multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------|
| OpenPose\*         |    -     | -          |   -     |  -     | 61.8  | 84.9  |  67.5  |  57.1  |  68.2  | 
| Hourglass          | Hourglass  | 512      | 277.8M  | 206.9  | 56.6  | 81.8  |  61.8  |  49.8  |  67.0  | 
| PersonLab          | ResNet-152  | 1401    |  68.7M  | 405.5  | 66.5  | 88.0  |  72.6  |  62.4  |  72.3  |
| PifPaf             |    -     | -          |   -     |  -     | 66.7  | -     |  -     |  62.4  |  72.9  | 
| Bottom-up HRNet    | HRNet-w32  | 512      |  28.5M  | 38.9   | 64.1  | 86.3  |  70.4  |  57.4  |  73.9  | 
| HigherHRNet        | HRNet-w32  | 512      |  28.6M  | 47.9   | 66.4  | 87.5  |  72.8  |  61.2  |  74.2  |
| HigherHRNet + SWAHR| HRNet-w32  | 512      |  28.6M  | 48.0   | 67.9  | 88.9  |  74.5  |  62.4  |  75.5  |  
| HigherHRNet        | HRNet-w48  | 640      |  63.8M  | 154.3  | 68.4  | 88.2  |  75.1  |  64.4  |  74.2  |
| HigherHRNet + SWAHR| HRNet-w48  | 640      |  63.8M  | 154.6  | 70.2  | 89.9  |  76.9  |  65.2  |  77.0  |

### Results on COCO test-dev2017 *with* multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------|
| Hourglass          | Hourglass  | 512      | 277.8M  | 206.9  | 63.0  | 85.7  |  68.9  |  58.0  |  70.4  | 
| Hourglass\*        | Hourglass  | 512      | 277.8M  | 206.9  | 65.5  | 86.8  |  72.3  |  60.6  |  72.6  | 
| PersonLab          | ResNet-152  | 1401    |  68.7M  | 405.5  | 68.7  | 89.0  |  75.4  |  64.1  |  75.5  | 
| HigherHRNet        | HRNet-w48  | 640      |  63.8M  | 154.3  | 70.5  | 89.3  |  77.2  |  66.6  |  75.8  |
| HigherHRNet + SWAHR| HRNet-w48  | 640      |  63.8M  | 154.6  | 72.0  | 90.7  |  78.8  |  67.8  |  77.7  |

### Results on CrowdPose test
| Method             |    AP | Ap .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|--------------------|-------|-------|--------|--------|--------|--------|
| Mask-RCNN          | 57.2  | 83.5  | 60.3   | 69.4   | 57.9   | 45.8   |
| AlphaPose          | 61.0  | 81.3  | 66.0   | 71.2   | 61.4   | 51.1   |
| SPPE               | 66.0. | 84.2 | 71.5 | 75.5 | 66.3 | 57.4 |
| OpenPose           | - | - | - | 62.7 | 48.7 | 32.3 |
| HigherHRNet        | 65.9  | 86.4  | 70.6   | 73.3   | 66.5   | 57.9   |
| HigherHRNet + SWAHR| 71.6  | 88.5  | 77.6   | 78.9   | 72.4   | 63.0   |
| HigherHRNet*        | 67.6  | 87.4  | 72.6   | 75.8   | 68.1   | 58.9   |
| HigherHRNet + SWAHR*| 73.8  | 90.5  | 79.9   | 81.2   | 74.7   | 64.7   |

'*' indicates multi-scale test

## Installation

The details about preparing the environment and datasets can be referred to [README.md](!https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/README.md).

Downlaod our pretrained weights from [BaidunYun](!https://pan.baidu.com/s/1aifAVwUbfAvRN4ZxrItZxQ)(Password: 8weh) or [GoogleDrive](!https://drive.google.com/drive/folders/13FFvwK7bDZLD4H_toueopbLhJqFjimlu?usp=sharing) to [./models](!./models).

### Training and Testing

#### Testing on COCO val2017 dataset using pretrained weights

For single-scale testing:

```
python tools/dist_valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_coco/pose_higher_hrnet_w32_512.pth
```

By default, we use horizontal flip. To test without flip:

```
python tools/dist_valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_coco/pose_higher_hrnet_w32_512.pth \
    TEST.FLIP_TEST False
```

Multi-scale testing is also supported, although we do not report results in our paper:

```
python tools/dist_valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_coco/pose_higher_hrnet_w32_512.pth \
    TEST.SCALE_FACTOR '[0.5, 1.0, 2.0]'
```


#### Training on COCO train2017 dataset

```
python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml 
```

By default, it will use all available GPUs on the machine for training. To specify GPUs, use

```
CUDA_VISIBLE_DEVICES=0,1 python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml 
```

## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{LuoSWAHR,
  title={Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation},
  author={Zhengxiong Luo and Zhicheng Wang and Yan Huang and Liang Wang and Tieniu Tan and Erjin Zhou},
  booktitle={CVPR},
  year={2021}
}
````

