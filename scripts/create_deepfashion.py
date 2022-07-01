#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from core.segment import Segmentor


# In[2]:


TARGET_DIM = 256 # target resolution


# In[3]:


language = loadmat('../data/language_original.mat')
subset = loadmat('../data/subset_index.mat')

captions = [x[0][0] for x in language['engJ']]
image_files = [x[0][0].split('img/')[-1] for x in subset['nameList']]


# In[21]:


src_dir = '../data/img_highres/'
dst_dir = '../data/img_256/'
mask_dir = '../data/mask_256/'
multi_dst_dir = '../data/img_multi_256/'
os.makedirs(multi_dst_dir)
os.makedirs(mask_dir)


# # Resize high resolution to 256x256

# In[39]:


df = pd.DataFrame(columns=['image', 'text', 'num_people', 'keypoints', 'pose_score'])

for image_fname, caption in tqdm(zip(image_files, captions)):
    src_file = os.path.join(src_dir,image_fname)
    dst_file = os.path.join(dst_dir,image_fname)
    img = cv2.imread(src_file)
    if img is None:
        continue
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    
    h, w, _ = img.shape
    if h>w:
        pad = (h-w)//2
        img = cv2.copyMakeBorder(img, top=0, bottom=0, left=pad, right=pad,
                                       borderType=cv2.BORDER_REPLICATE)
    elif h<w:
        pad = w-h
        img = cv2.copyMakeBorder(img, top=pad, bottom=0, left=0, right=0,
                                       borderType=cv2.BORDER_REPLICATE)
    h, w, _ = img.shape
    
    if h!=TARGET_DIM or w!=TARGET_DIM:
        img = cv2.resize(img, (TARGET_DIM, TARGET_DIM), cv2.INTER_AREA)
    
    cv2.imwrite(dst_file, img)    

    new_row = {'image':dst_file, 'text':[caption], 'num_people':1}
    df = df.append(new_row, ignore_index=True)
    
df.to_pickle("deepfashion.pickle")


# In[113]:


df = pd.read_pickle("deepfashion.pickle")

image_files = list(df.image)
texts = list(df.text)
N = len(images_files)
assert N==len(texts)


# In[217]:


import re
reg = re.compile('women|woman|lady|girl')
df['female'] = df.text.map(lambda x: 1 if reg.search(x[0]) else 0)
df['male'] = 1^ df['female']


# In[229]:


male_indices = list(df[df.male==1].index)


# ## Get Mask

# In[6]:


segmentor = Segmentor()


# In[236]:


for image_fname in tqdm(image_files):
    bmp_name = image_fname.replace('img_256','mask_256').replace('.jpg','.bmp')
    os.makedirs(os.path.dirname(bmp_name), exist_ok=True)
    img = cv2.imread(image_fname)
    _, mask = segmentor(img)
    cv2.imwrite(bmp_name, mask)


# In[237]:


def get_sample(sampled_idx, margin=5, mask_background=True):
    row = df.iloc[sampled_idx]
    image_fname = row.image #image_files[sampled_idx]
    image = cv2.imread(image_fname)
    bmp_name = image_fname.replace('img_256','mask_256').replace('.jpg','.bmp')
    mask = cv2.imread(bmp_name, 0)    
    if mask_background:
        image = cv2.bitwise_and(image, image, mask=mask)        
        image[mask==0] = (255, 255, 255)
        
    # crop 
    vertical = np.mean(mask, axis=0)
    height, width = mask.shape
    for w in range(width):
        if vertical[w] > 0.1:
            left = w
            break
            
    for w in range(width-1, -1, -1):
        if vertical[w] > 0.1:
            right = w
            break
            
    left = max(0, left-margin)
    right = min(width, right+margin)
    
    return image[:,left:right], row.text[0], row.female, row.male
    


# In[241]:


HEIGHT = 256
WIDTH = 256
multi_df = pd.DataFrame(columns=['image', 'text', 'num_people', 'keypoints', 'pose_score'])

empty_percent = 0.2
male_percent = 0.2
random.seed(888)
np.random.seed(888)

num_samples = {2: 50000, 3:50000} # samples per numbeer of people

for p, num_sample in num_samples.items():
    for idx in tqdm(range(num_sample)):
        num_slots = p
        empty = random.uniform(0,1) < empty_percent
        if empty: # have empty slot
            num_slots += 1

        slots_avail = [1 for _ in range(num_slots)]
        if empty:
            slots_avail[random.randrange(0,num_slots,1)] = 0
        '''
        print("num_slot", num_slots)
        print("empty_slot_id", empty_slot_id)
        print("slots_avail", slots_avail)
        '''
        merged = np.zeros((TARGET_DIM, TARGET_DIM, 3), dtype=np.uint8)
        slot_width = TARGET_DIM//num_slots

        slot_images = []
        captions = ""
        male_count = 0
        female_count = 0
        for not_empty in slots_avail:
            if not not_empty:
                empty_image = 255*np.ones((HEIGHT, int(WIDTH*0.4),3), dtype=np.uint8)
                slot_images.append(empty_image)
                continue
            # for certain percentage, sample from male only
            if random.uniform(0,1) < male_percent:
                sampled_idx = random.sample(male_indices[:100], 1)[0]
            else:
                sampled_idx = random.randrange(N)
            cropped_image, caption, female, male = get_sample(sampled_idx)
            captions += caption
            male_count += male
            female_count += female
            # random resize
            scale_factor = np.random.uniform(0.9, 1.1)
            cropped_image = cv2.resize(cropped_image, None, fx=scale_factor, fy=scale_factor,
                                      interpolation=cv2.INTER_CUBIC)

            slot_images.append(cropped_image)


        merged = 255*np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8)
        total_width = sum(image.shape[1] for image in slot_images)

        start_x = 0
        for i, slot_image in enumerate(slot_images):
            h, w, _ = slot_image.shape
            scale_factor = WIDTH/total_width
            slot_image = cv2.resize(slot_image, None, fx=scale_factor, fy=scale_factor)
            h, w, _ = slot_image.shape

            end_x = min(start_x + w, WIDTH-1)
            capped_w = end_x - start_x
            merged[-h:,start_x:end_x] = slot_image[:HEIGHT,:capped_w]
            start_x += w
        #plt.imshow(merged[:,:,::-1])
        #plt.show()
        dst_file = os.path.join(multi_dst_dir, f'{p}_{idx}.png')
        cv2.imwrite(dst_file, merged)

        new_row = {'image':dst_file, 'text':[caption], 'num_people':p, 'female':female_count, 'male':male_count}
        multi_df = multi_df.append(new_row, ignore_index=True)


# In[245]:


multi_df.to_pickle("../data/deepfashion_23.pickle")


# In[248]:


pd.concat([df, multi_df], ignore_index=True).to_pickle("../data/deepfashion_123.pickle")

