#!/usr/bin/env python
# coding: utf-8


import os, sys
sys.path.append('/usr/local/python')

import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from openpose import pyopenpose as op

df = pd.read_pickle('../data/deepfashion_123.pickle')
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU id')
parser.add_argument('--openpose_model', type=str, default="/home/soon/github/openpose/models", 
                    required=False, help='path to openpose/models')
parser.add_argument('--number_people_max', type=int, default=4)


args = parser.parse_args()

params = dict()
params["model_folder"] = args.openpose_model
params["disable_blending"] = True
params["keypoint_scale"] = 3 #normalise keypoints to [0,1]
params["num_gpu"] = 1
params["num_gpu_start"] = args.gpu
params["number_people_max"] = args.number_people_max

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

for i in tqdm(range(len(df))):
    image_path = df.iloc[i].image
    datum = op.Datum()
    datum.cvInputData = cv2.imread(image_path)
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    df.at[i, 'pose_score'] = datum.poseScores
    df.at[i, 'keypoints'] = datum.poseKeypoints

df.to_pickle('../data/deepfashion_123.pickle')

