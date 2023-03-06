from pathlib import Path
from random import randint, choice, uniform

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T

import pandas as pd
from einops import rearrange
import numpy as np

from core.pose_utils import keypoints_to_heatmap, ToTensor, ConcatSamples
from core.pose_utils import CenterCropResize, pad_keypoints, PoseVisualizer
from core.utils import instantiate_from_config, get_obj_from_str

class PoseDatasetPickle(Dataset):
    def __init__(self,
                 pickle_file,
                 folder,
                 pose_format, # 'image' or 'keypoint' or 'heatmap'
                 text_encoder_config,
                 pose_encoder_config,
                 shuffle=False,                 
                 image_size=256,
                 pose_image_shape=256,                
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        #self.pose_visualizer = PoseVisualizer('keypoint', (pose_image_shape, pose_image_shape))
        self.pose_format = pose_format
        self.shuffle = shuffle
        self.df = pd.read_pickle(pickle_file)
        self.root_dir = Path(folder)
        self.text_encoder = instantiate_from_config(text_encoder_config)
        self.pose_encoder = instantiate_from_config(pose_encoder_config)
        self.image_keypoint_transform = T.Compose([
            #CenterCropResize(),
            #RotateScale((-10,10),(1.0,1.1)),
            #Crop((0.0, 0.08)),
            ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        sample = self.df.iloc[ind]
        image_file = self.root_dir / sample.image
        descriptions = sample.text.copy()
        keypoints = sample.keypoints.copy()
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load captions.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        image = PIL.Image.open(str(image_file))
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = np.array(image)

        text_tokens = self.text_encoder(description).squeeze(0)

        augmented = self.image_keypoint_transform({'image':image, 'keypoints':keypoints})
        image_tensor, keypoints = augmented['image'], augmented['keypoints']

        pose_tokens = self.pose_encoder(keypoints)

        ret = {'text_tokens':text_tokens,
               'pose_tokens':pose_tokens,
               'image_tensor':image_tensor,
               'filename':sample.image}
        return ret
