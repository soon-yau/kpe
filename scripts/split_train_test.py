#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[23]:


df = pd.read_pickle('../data/deepfashion_123.pickle')


# In[28]:


df['pose_count']=df.pose_score.map(lambda x:len(x) if x is not None else 0)
df.drop(df[df['pose_count']!=df['num_people']].index, inplace=True)


# In[29]:


train_df, test_df = train_test_split(df, test_size=1000)
test_df.reset_index(inplace=True)
train_df.reset_index(inplace=True)


# In[30]:


train_df.to_pickle('../data/deepfashion_123_train.pickle')
test_df.to_pickle('../data/deepfashion_123_test.pickle')

