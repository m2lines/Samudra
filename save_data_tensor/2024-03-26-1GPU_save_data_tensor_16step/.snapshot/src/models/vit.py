import torch
from torch import nn
from torchvision.transforms import Resize
from einops.layers.torch import Rearrange
import copy
from omegaconf import OmegaConf
import collections


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=9, patch_size=16, emb_size=768, img_size=(128, 144)):
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (img_size[0] * img_size[1]) // (patch_size**2) + 1, emb_size)
        )
        self.patch_to_embedding = nn.Linear(
            patch_size * patch_size * in_channels, emb_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.rearrange = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
        )

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.rearrange(x)
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels=9,
        patch_size=16,
        emb_size=768,
        num_layers=6,
        num_heads=8,
        output_channels=3,
        img_size=[128, 144],
    ):
        super().__init__()

        if not isinstance(
            img_size, list
        ):  # Possibility of passing parameters through omegaconf. Need to convert to native list.
            img_size = OmegaConf.to_object(img_size)

        self.img_size = copy.copy(img_size)

        if img_size[0] % patch_size != 0:
            img_size[0] = (int(img_size[0] / patch_size) + 1) * patch_size

        if img_size[1] % patch_size != 0:
            img_size[1] = (int(img_size[1] / patch_size) + 1) * patch_size

        self.resize = Resize(img_size)
        self.patch_embedding = PatchEmbedding(
            in_channels, patch_size, emb_size, img_size
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.to_pixel_values = nn.Sequential(
            nn.Linear(emb_size, patch_size * patch_size * output_channels),
            Rearrange(
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                h=img_size[0] // patch_size,
                w=img_size[1] // patch_size,
                p1=patch_size,
                p2=patch_size,
                c=output_channels,
            ),
            Resize(img_size),
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.to_pixel_values(x[:, 1:, :])  # Skip the CLS token
        x = x[:, :, : self.img_size[0], : self.img_size[1]]
        return x
