from pathlib import Path
from random import randint, choice, uniform

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T

import pandas as pd
from einops import rearrange
import numpy as np

from core.pose_utils import keypoints_to_heatmap, RotateScale, Crop, ToTensor, ConcatSamples
from core.pose_utils import CenterCropResize, pad_keypoints, PoseVisualizer

class PoseDatasetPickle(Dataset):
    def __init__(self,
                 pickle_file,
                 folder='./data/',
                 text_len=256,
                 image_size=256,
                 truncate_captions=False,
                 tokenizer=None,
                 shuffle=False,
                 pose_format='image', # 'image' or 'keypoint' or 'heatmap'
                 pose_image_shape=(256, 256),
                 max_people=3,
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.pose_visualizer = PoseVisualizer('keypoint', pose_image_shape)
        self.pose_format = pose_format
        self.shuffle = shuffle
        self.df = pd.read_pickle(pickle_file)
        self.root_dir = Path(folder)
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.tokenizer = tokenizer
        self.max_people = max_people
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
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        image = PIL.Image.open(str(image_file))
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = np.array(image)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        
        try:
            # augmentation, to do, multiple keypoints
            padded_keypoints = pad_keypoints(keypoints, self.max_people)
            #padded_keypoints = keypoints
            augmented = self.image_keypoint_transform({'image':image, 'keypoints':padded_keypoints})
            image_tensor, keypoints = augmented['image'], augmented['keypoints']

            if self.pose_format == 'keypoint':
                pose_tensor = keypoints
            elif self.pose_format == 'image':
                pose_tensor = self.pose_visualizer.convert(keypoints)
            else:
                pose_tensor = keypoints
                #raise(ValueError, f'f pose format of {self.pose_format}is undefined')
                
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success

        return tokenized_text, image_tensor, pose_tensor
