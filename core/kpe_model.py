import os, sys, argparse, pdb
from types import TracebackType
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

from importlib import import_module
from omegaconf import OmegaConf
from typing import List
from PIL import Image
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

from core.utils import instantiate_from_config, get_obj_from_str
from core.pose_utils import Keypoints2Image

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]


class KPEModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        self.pose2image = Keypoints2Image()

        self.transformer = instantiate_from_config(config['transformer'])
        d_model = self.transformer.d_model

        # encoding to tokens
        self.text_encoder = instantiate_from_config(config['text_encoder'])
        self.pose_encoder = instantiate_from_config(config['pose_encoder'])
        self.image_encoder = instantiate_from_config(config['image_encoder'])
        set_requires_grad(self.image_encoder, False)

        # sequence length
        self.text_token_len = self.text_encoder.text_len
        self.pose_token_len = self.pose_encoder.num_keypoints

        # dimension
        # unique token for each padding position, hence + self.text_token_len
        self.text_token_size = self.text_encoder.vocab_size + self.text_token_len
        self.image_token_size = self.image_encoder.num_tokens
        
        # tokens to embedding
        self.text_embed = nn.Embedding(self.text_token_size, d_model)
        self.image_embed = nn.Embedding(self.image_token_size, d_model)        
        self.pose_embed = nn.Linear(3*self.pose_encoder.max_num_people, d_model)

        # positional encoding
        self.text_pos_emb = nn.Embedding(self.text_token_len + 1, d_model)
        image_token_dim = self.image_encoder.fmap_size
        self.image_token_len = image_token_dim**2
        self.image_pos_emb = AxialPositionalEmbedding(d_model, \
            axial_shape = (image_token_dim, image_token_dim))
        
        # embedding to logits
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.image_token_size+self.text_token_size))        

        # loss constants
        self.lambda_text = config['loss_constant']['text']
        self.lambda_pose = config['loss_constant']['pose']
        self.lambda_image = config['loss_constant']['image']

        total_tokens = self.text_token_size + self.image_token_size
        logits_range = torch.arange(total_tokens)
        logits_range = rearrange(logits_range, 'd -> () () d')
        self.image_mask = logits_range < self.text_token_size
        self.text_mask = ~self.image_mask

    def configure_optimizers(self):
        # Optimizer
        config_optimizer = self.config.optimizer
        optimizer = get_obj_from_str(config_optimizer.target)(\
            params=get_trainable_params(self), **config_optimizer.params)

        config_scheduler = self.config.get('scheduler', None)

        if config_scheduler == None:
            return optimizer

        scheduler = get_obj_from_str(config_scheduler.target)(\
            optimizer=optimizer, **config_scheduler.params)
    
        return {"optimizer":optimizer, "lr_scheduler":scheduler, 
                "monitor":"train/loss_image"}

    def forward(self, text_tokens, pose_tokens, image_tokens=None, return_loss=False):
        batch_size, text_len = text_tokens.shape
        assert text_len==self.text_token_len

        total_seq_len = self.text_token_len + self.image_token_len + self.pose_token_len

        # add unique padding
        text_range = torch.arange(text_len, device = self.device) \
            + (self.text_token_size - text_len)
        text_tokens = torch.where(text_tokens == 0, text_range, text_tokens)

        text_tokens = F.pad(text_tokens, (1, 0), value = 0)

        text_embedding = self.text_embed(text_tokens)
        text_embedding += self.text_pos_emb(torch.arange(text_tokens.shape[1], device=self.device))
        
        pose_len = pose_tokens.shape[1]
        pose_embedding = self.pose_embed(pose_tokens)

        if image_tokens == None:
            image_tokens = torch.zeros((batch_size,1), dtype=torch.int64).to(self.device)

        image_embedding = self.image_embed(image_tokens)
        image_embedding += self.image_pos_emb(image_embedding)
        image_len = pose_tokens.shape[1]
        tokens = torch.cat((text_embedding, pose_embedding, image_embedding), axis=1)            

        if tokens.shape[1] > total_seq_len:
            tokens = tokens[:, :-1]

        seq_offset = text_len
        outputs = self.transformer(tokens)
        logits = self.to_logits(outputs)
        max_neg_value = -torch.finfo(logits.dtype).max

        text_logits = logits[:,:seq_offset,:]
        pose_outputs = outputs[:,seq_offset:seq_offset+pose_len,:]
        image_logits = logits[:,seq_offset+pose_len:,:]

        text_logits.masked_fill_(self.text_mask.to(self.device), max_neg_value)
        image_logits.masked_fill_(self.image_mask.to(self.device), max_neg_value)

        if not return_loss:
            return text_logits, pose_outputs, image_logits
    
        text_logits = rearrange(text_logits, 'n d c -> n c d')
        image_logits = rearrange(image_logits, 'n d c -> n c d')
        
        loss_text = self.lambda_text * F.cross_entropy(text_logits, text_tokens[:,1:])
        loss_image = self.lambda_image * F.cross_entropy(image_logits, \
                                                        image_tokens+self.text_token_size)
        loss_pose = self.lambda_pose * F.mse_loss(pose_outputs, pose_embedding)
        
        total_loss = (loss_text + loss_pose + loss_image)/self.lambda_image

        return {'total':total_loss, 'text':loss_text, 'image':loss_image, 'pose':loss_pose}

    def training_step(self, batch, batch_idx):

        text_tokens, pose_tokens, image_tensor = batch
        image_tokens = self.image_encoder(image_tensor)
        losses = self.forward(text_tokens, pose_tokens, image_tokens, return_loss=True)

        self.log('train/loss_text', losses['text'])
        self.log('train/loss_pose', losses['pose'])
        self.log('train/loss_image', losses['image'])
        self.log('train/loss_total', losses['total'])

        return losses['total']

    def validation_step(self, batch, batch_idx):

        text_tokens, pose_tokens, image_tensor = batch
        image_tokens = self.image_encoder(image_tensor)
        losses = self.forward(text_tokens, pose_tokens, image_tokens, return_loss=True)

        self.log('val/loss_text', losses['text'])
        self.log('val/loss_pose', losses['pose'])
        self.log('val/loss_image', losses['image'])
        self.log('val/loss_total', losses['total'])

    def preprocess(self, texts:List[str], 
                      poses:np.array, 
                      images:np.array,
                      image_mask=None):

        text_tokens = torch.vstack([self.text_encoder(t) for t in texts])

        pose_tokens = torch.from_numpy(np.array([self.pose_encoder(pose) for pose in poses]))

        image_tensor = torch.vstack([T.ToTensor()(image).unsqueeze(0) for image in images])
        #image_tokens = self.image_encoder(image_tensor)

        return text_tokens, pose_tokens, image_tensor

    @torch.no_grad()
    @eval_decorator
    def generate_image(self, text_tokens, pose_tokens, image_tensor=None, filter_thres=0.9, temperature=1.):
        image_tokens = None
        start_idx = 0 #if image_tokens==None else image_tokens.shape[1]
        for i in range(start_idx, self.image_token_len):
            text_logits, pose_outputs, image_logits = self(text_tokens, pose_tokens, image_tokens)
            logits = image_logits[:,-1,:]
            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1) - self.text_token_size
            if image_tokens == None:
                image_tokens = sample
            else:
                image_tokens = torch.cat((image_tokens, sample), dim=-1)

        images = self.image_encoder.decode(image_tokens)

        return images

class ImageLogger(Callback):
    def __init__(self, frequency, max_num_images=None):
        self.frequency = frequency
        self.max_num_images = max_num_images

    def log_image(self, pl_module, batch):
        device = pl_module.device
        text_tokens, pose_tokens, image_tensor = batch
        text_tokens, pose_tokens, image_tensor = text_tokens.to(device), \
                                                 pose_tokens.to(device), image_tensor.to(device)
        if self.max_num_images:
            text_tokens = text_tokens[:self.max_num_images] 
            pose_tokens = pose_tokens[:self.max_num_images] 
            image_tensor = image_tensor[:self.max_num_images] 
        images = pl_module.generate_image(text_tokens, pose_tokens, None)
        texts = [pl_module.text_encoder.decode(t) for t in text_tokens.cpu().numpy()]

        #reconst_images = pl_module.image_encoder.decode(pl_module.image_encoder(image_tensor))

        poses = pl_module.pose_encoder.decode(pose_tokens)
        pose_images =torch.stack([T.ToTensor()(pl_module.pose2image(pose)) for pose in poses])
        pose_images = pose_images.to(pl_module.device)
        display_image = torch.cat((pose_images, images, image_tensor), dim=-1)
        pl_module.logger.experiment.log({"generated": [wandb.Image(image, caption=caption) \
            for image, caption in zip(display_image, texts)]})

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if (batch_idx + 1) % self.frequency == 0:
            self.log_image(pl_module, batch)