from taming.models.vqgan import VQModel, GumbelVQ
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from core.utils import instantiate_from_config
from einops import rearrange
import math
import numpy as np

# Source: https://github.com/lucidrains/DALLE-pytorch
class VQGanVAE(nn.Module):
    def __init__(self, vqgan_model_path=None, vqgan_config_path=None, device='cpu'):
        super().__init__()

        model_path = vqgan_model_path
        config_path = vqgan_config_path

        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config["model"])

        #state = torch.load(model_path, map_location = device)['state_dict']
        state = torch.load(model_path)['state_dict']
        model.load_state_dict(state, strict = False)

        print(f"Loaded VQGAN from {model_path} and {config_path}")

        self.model = model

        self.image_size = config.model.params.ddconfig.resolution
        self.fmap_size = int(self.image_size//np.prod(config.model.params.ddconfig.ch_mult))
        self.num_tokens = config.model.params.n_embed
        self.is_gumbel = isinstance(self.model, GumbelVQ)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        if self.is_gumbel:
            return rearrange(indices, 'b h w -> b (h w)', b=b)
        return rearrange(indices, '(b n) -> b n', b = b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        z = one_hot_indices @ self.model.quantize.embed.weight if self.is_gumbel \
            else (one_hot_indices @ self.model.quantize.embedding.weight)

        z = rearrange(z, 'b (h w) c -> b c h w', h = int(math.sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, img):
        return self.get_codebook_indices(img)
