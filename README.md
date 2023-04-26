# Multiperson Pose Detection and Segementation
This project combines both the codes of [Pos2Seg](https://github.com/MiryalaNarayanaReddy/Pose2Seg) and [Multiperson pose estimation](https://github.com/MiryalaNarayanaReddy/multi-person-pose-estimation) and gets the colored segmentations of humans.

Results can be viewed in videoes 

1. [mutiperson pos2seg 1](https://www.youtube.com/watch?v=5FE6Fgl-f_c) 

2. [multiperson pos2seg 2](https://www.youtube.com/watch?v=IFD3JH0SqfM)

3. [key points detected](https://youtu.be/TO098-uBKtU)

4. [plain color detection](https://youtu.be/olo9B_kALnc)


# 1. Installing

1. clone this repo
```
git clone <repo url>
```
2. create new environment from environment.yml (this is for all codes in this repo) 

```
conda env create -f environment.yml
```

# 2. Download models
1. pos2seg_release.pkl from [here](https://drive.google.com/file/d/193i8b40MJFxawcJoNLq1sG0vhAeLoVJG/view?usp=sharing)
2. yolov7-w6.pt from [here](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt)
3. yolov7.pt from [here](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)


# 3. Running on your own images/videos

## 3a) for segmentation (using yolov7-medipipe-pos2seg pipeline)

```
python realtime_test_pos2seg.py --weights pose2seg_release.pkl --img_dir <path> --save_dir <path>
```
or
```
python realtime_test_pos2seg.py --weights pose2seg_release.pkl --source <video> --save_dir <path>
```
![segmented img 1](./figure/readme_images/results/dv1.png)
![segmented img 2](./figure/readme_images/results/dv2.png)

## 3b) For key points (using yolov7-mediapipe pipeline)

```
python detect_pose.py --source <video>
```

![pose image](./figure/readme_images/bus.jpg)


## 3c) if you have keypoints already
```
python realtime_run.py --weights pose2seg_release.pkl --img_dir <path> --kp <key_points_json> --save_dir <path>

```

## 3d) for yolov-pose keypoints 
This give better keypoints than mediapipe but takes more time (approx 2-3 seconds per frame where as mediapipe takes 0.3-0.5 seconds per frame)
```
python yolov-pose.py --source <video>
```


video can be 0 for webcam or path to video file or path to image folder

## 3e) creating dataset for your own images
Use [Multi-person-pose-estimation ](https://github.com/MiryalaNarayanaReddy/multi-person-pose-estimation) to generate pose keypoints for your images. This code creates dataset in compatible format for this repo.

![image](./figure/readme_images/multiperson_pose_estimation.png)


### 3f) Testing pos2seg (Getting segmented images)
1. run the following to test on your own images
```
python realtime_run.py --weights pose2seg_release.pkl --img_dir <path> --kp <key_points_json> --save_dir <path>

```

This will give you the cropped segmented images in the save_dir folder.

![image](./figure/readme_images/multiperson_pose_estimation_2.png)

__________________________

_______________________
______________________
______________________

# Pose2Seg (CVPR2019) offical code

*Official* code for the paper "Pose2Seg: Detection Free Human Instance Segmentation"[[ProjectPage]](http://www.liruilong.cn/projects/pose2seg/index.html)[[arXiv]](https://arxiv.org/abs/1803.10683) @ CVPR2019.

The *OCHuman dataset* proposed in our paper is released [here](https://github.com/liruilong940607/OCHumanApi)

<div align="center">
<img src="figures/pipeline.jpg" width="1000px"/>
<p> Pipeline of our pose-based instance segmentation framework.</p>
</div>

## Setup environment

``` bash
pip install cython matplotlib tqdm opencv-python scipy pyyaml numpy
pip install torchvision torch

cd ~/github-public/cocoapi/PythonAPI/
python setup.py build_ext install
cd -
```

## Download data

- COCO 2017
    - [COCO 2017 Train images [118K/18GB]](http://images.cocodataset.org/zips/train2017.zip)
    - [COCO 2017 Val images [5K/1GB]](http://images.cocodataset.org/zips/val2017.zip)
    - [COCOPersons Train Annotation (person_keypoints_train2017_pose2seg.json) [166MB]](https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_train2017_pose2seg.json)
    - [COCOPersons Val Annotation (person_keypoints_val2017_pose2seg.json) [7MB]](https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_val2017_pose2seg.json)
    
- OCHuman
    - [images [667MB] & annotations](https://cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ochuman)
    
**Note**: 
`person_keypoints_(train/val)2017_pose2seg.json` is a subset of `person_keypoints_(train/val)2017.json` (in [COCO2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)). We choose those instances with both keypoint and segmentation annotations for our experiments.

## Setup data

The `data` folder should be like this:

    data  
    ├── coco2017
    │   ├── annotations  
    │   │   ├── person_keypoints_train2017_pose2seg.json 
    │   │   ├── person_keypoints_val2017_pose2seg.json 
    │   ├── train2017  
    │   │   ├── ####.jpg  
    │   ├── val2017  
    │   │   ├── ####.jpg  
    ├── OCHuman 
    │   ├── annotations  
    │   │   ├── ochuman_coco_format_test_range_0.00_1.00.json   
    │   │   ├── ochuman_coco_format_val_range_0.00_1.00.json   
    │   ├── images  
    │   │   ├── ####.jpg 

## How to train

``` bash
python train.py
```

**Note**: Currently we only support for single-gpu training.

## How to test

This allows you to test the model on (1) COCOPersons val set and (2) OCHuman val & test set.

``` bash
python test.py --weights last.pkl --coco --OCHuman
```

We retrained our model using this repo, and got similar results with our paper. The final weights can be download [here](https://drive.google.com/file/d/193i8b40MJFxawcJoNLq1sG0vhAeLoVJG/view?usp=sharing).

## About Human Pose Templates in COCO

<div align="center">
<img src="figures/pose_templates.png" width="500px"/>
<p> Pose templates clustered using K-means on COCO.</p>
</div>

This repo already contains a template file `modeling/templates.json` which was used in our paper. But you are free to explore different cluster parameters as discussed in our paper. See [visualize_cluster.ipynb](visualize_cluster.ipynb) for an example.




