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


class KPEModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pose2image = Keypoints2Image()

        self.transformer = instantiate_from_config(config['transformer'])
        d_model = self.transformer.d_model

        # encoding to tokens
        self.text_encoder = instantiate_from_config(config['text_encoder'])
        self.pose_encoder = instantiate_from_config(config['pose_encoder'])
        self.image_encoder = instantiate_from_config(config['image_encoder'])
        set_requires_grad(self.image_encoder, False)
        
        
        self.text_token_size = self.text_encoder.vocab_size
        self.image_token_size = self.image_encoder.num_tokens
        
        # tokens to embedding
        self.text_embed = nn.Embedding(self.text_token_size, d_model)
        self.image_embed = nn.Embedding(self.image_token_size, d_model)        
        self.pose_embed = nn.Linear(3*self.pose_encoder.max_num_people, d_model)

        # positional encoding
        self.text_pos_emb = nn.Embedding(self.text_token_size, d_model)
        image_token_dim = self.image_encoder.fmap_size
        self.image_token_len = image_token_dim**2
        self.image_pos_emb = AxialPositionalEmbedding(d_model, \
            axial_shape = (image_token_dim, image_token_dim))
        
        # embedding to logits
        self.to_logits_text = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.text_token_size))    

        self.to_logits_image = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.image_token_size))        
        
        # loss constants
        self.lambda_text = config['loss_constant']['text']
        self.lambda_pose = config['loss_constant']['pose']
        self.lambda_image = config['loss_constant']['image']

        self.save_hyperparameters()

    def configure_optimizers(self):
        # Optimizer
        config_optimizer = self.config.optimizer
        optimizer = get_obj_from_str(config_optimizer.target)(\
            params=self.parameters(), **config_optimizer.params)

        config_scheduler = self.config.get('scheduler', None)

        if config_scheduler == None:
            return optimizer

        scheduler = get_obj_from_str(config_scheduler.target)(\
            optimizer=optimizer, **config_scheduler.params)
    
        return {"optimizer":optimizer, "lr_scheduler":scheduler, "monitor":"test_loss_total"}

    def forward(self, text_tokens, pose_tokens, image_tokens=None):
        batch_size, text_len = text_tokens.shape        
        text_embedding = self.text_embed(text_tokens)
        text_embedding += self.text_pos_emb(torch.arange(text_len).to(self.device))
        
        pose_len = pose_tokens.shape[1]
        pose_embedding = self.pose_embed(pose_tokens)

        if image_tokens == None:
            image_tokens = torch.zeros((batch_size,1), dtype=torch.int64).to(self.device)

        image_embedding = self.image_embed(image_tokens)
        image_embedding += self.image_pos_emb(image_embedding)
        image_len = pose_tokens.shape[1]
        tokens = torch.cat((text_embedding, pose_embedding, image_embedding), axis=1)            
            
        outputs = self.transformer(tokens)

        seq_offset = text_len
        text_outputs = outputs[:,:seq_offset,:]
        pose_outputs = outputs[:,seq_offset:seq_offset+pose_len,:]
        seq_offset += pose_len
        image_outputs = outputs[:,seq_offset:,:]
        
        text_logits = self.to_logits_text(text_outputs)
        image_logits = self.to_logits_image(image_outputs)
        
        return text_logits, pose_outputs, image_logits, pose_embedding
    
    def training_step(self, batch, batch_idx):
        text_tokens, pose_tokens, image_tensor = batch
        image_tokens = self.image_encoder(image_tensor)
        text_logits, pose_outputs, image_logits, pose_embedding = self.forward(text_tokens, pose_tokens, image_tokens)
        
        text_logits = rearrange(text_logits, 'n d c -> n c d')
        image_logits = rearrange(image_logits, 'n d c -> n c d')
        

        loss_text = self.lambda_text * F.cross_entropy(text_logits, text_tokens)
        loss_image = self.lambda_image * F.cross_entropy(image_logits, image_tokens)
        loss_pose = self.lambda_pose * F.mse_loss(pose_outputs, pose_embedding)
        
        total_loss = loss_text + loss_pose + loss_text

        self.log('train_loss_text', loss_text)
        self.log('train_loss_pose', loss_pose)
        self.log('train_loss_image', loss_image)
        self.log('train_loss_total', total_loss)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        text_tokens, pose_tokens, image_tensor = batch
        texts = [self.text_encoder.decode(t) for t in text_tokens.cpu().numpy()]
        images = self.generate_image(text_tokens, pose_tokens, None)

        poses = self.pose_encoder.decode(pose_tokens)
        pose_images =torch.stack([T.ToTensor()(self.pose2image(pose)) for pose in poses])
        pose_images = pose_images.to(self.device)
        display_image = torch.cat((pose_images, images, image_tensor), dim=-1)
        self.logger.experiment.log({"generated": [wandb.Image(image, caption=caption) \
            for image, caption in zip(display_image, texts)]})

    def preprocess(self, texts:List[str], 
                      poses:np.array, 
                      image:np.array,
                      image_mask=None):

        text_tokens = torch.vstack([self.text_encoder(t) for t in texts])

        pose_tokens = torch.from_numpy(np.array([self.pose_encoder(pose) for pose in poses]))

        image_tensor = torch.vstack([T.ToTensor()(image).unsqueeze(0) for image in images])
        #image_tokens = self.image_encoder(image_tensor)

        return text_tokens, pose_tokens, image_tensor

    @torch.no_grad()
    @eval_decorator
    def generate_image(self, text_tokens, pose_tokens, image_tensor=None, filter_thres=0.9, temperature=1.):
        #image_tokens = self.image_encoder(image_tensor)
        image_tokens = None
        start_idx = 0 #if image_tokens==None else image_tokens.shape[1]
        for i in range(start_idx, self.image_token_len):
            text_logits, pose_outputs, image_logits, _ = self.forward(text_tokens, pose_tokens, image_tokens)
            logits = image_logits[:,-1,:]
            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)
            if image_tokens == None:
                image_tokens = sample
            else:
                image_tokens = torch.cat((image_tokens, sample), dim=-1)
        images = self.image_encoder.decode(image_tokens)
        return images