# Instruction to create multiperson Deepfashion dataset

## 1. Download the following files from DeepFashion http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/ and place them in /data
The data from DeepFashion Fashion Synthesis Benchmark images are only 128x128, therefore, we will need to download the high resolution images images img_highres.zip and resize them to 256x256.
- Fashion Synthesis Benchmark
 - Anno/language_original.mat
 - Img/subset_index.tar.gz
- Category and Attribute Prediction Benchmark/img/img_highres.zip

## 2. Install OpenPose Python API
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/03_python_api.md

## 3. Resize image and create multiperson images
```
python create_deepfashion.py
```

## 4. Capture Keypoints
```
python get_pose.py
```

## 5. Create train and test split
```
python  split_train_test.py
```
Two pickle files deepfashion_123_train.pickle and deepfashion_123_test.pickle will be created in /data.