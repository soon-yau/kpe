# KPE: Keypoint Pose Encoding for Transformer-based Image Generation

This is the official code repo for the paper:
"KPE: Keypoint Pose Encoding for Transformer-based Image Generation" https://arxiv.org/abs/2203.04907

## Inference
- Download [DeepFashion Fashion Synthesis Benchmark images](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html), resize to 256x256 and place in datasets/syn/img. 
- Download pretrained checkpoint from [Google Drive](https://drive.google.com/drive/folders/1bpXN4z2qy2XrWx5DgBUPwyxo2v4qH3c0?usp=sharing) and unzip. To run inference on Deepfashion Synthesis Benchmark test set and samples will be stored in /results:
```
python train.py --config configs/kpe.yaml --gpus 0, --finetune_from checkpoints/deepshion/kpe.ckpt 
```

## To train
```
python train.py -t --config configs/kpe.yaml --gpus 0,
```

## Prepare Dataset
The DeepFashion dataset can be downloaded from DeepFashion http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/. You will need to sign agreement prior to obtain password for unzipping the files. 

### 1. Download the following files from DeepFashion http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/ and place them in ./data
The data from DeepFashion Fashion Synthesis Benchmark images are only 128x128, therefore, we will need to download the high resolution images images img_highres.zip and resize them to 256x256.
- Category and Attribute Prediction Benchmark/img/img_highres.zip

Unzip img_highres.zip to ./data/img_highres
Go to ./scripts and run resize_segment.ipynb

### 2. Data pre-preprocesing (not required)
I have preprocessed the data of DeepFashion Synthesis Benchmark and stored in ./data/deepfashion_1.pickle. This includes removing a few wrong images, correcting typos in captions, extracting keypoints and meta data such as gender. The steps are listed below for reference, but warning, this requires installation of OpenPose and can take hours to run. If you run the following steps, the resulting multiperson dataset will be different.

1. Download Fashion Synthesis Benchmark and place them in ./data/
 - Anno/language_original.mat
 - Img/subset_index.tar.gz

2. Go to ./scripts and run create_pickle.ipynb
3. run tokenize.ipynb to train BPE tokenizer
3. Install OpenPose Python API
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/03_python_api.md
4. run get_pose.py to obtain pose keypoints

### 3. Create multiperson dataset
Run create_multiperson.ipynb and deepfashion_123_train.pickle and deepfashion_123_test.pickle will be created in ./data
