import enum
from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


@dataclass
class MLPConfig:
    hidden_dim: int = 3072
    dropout: float = 0.0


@dataclass
class MultiHeadAttentionConfig:
    n_heads: int = 12
    dropout: float = 0.0


@dataclass
class TransformerEncoderConfig:
    depth: int = 12
    attention_config: MultiHeadAttentionConfig = MultiHeadAttentionConfig()
    mlp_config: MLPConfig = MLPConfig()


@dataclass
class PatchEmbedderConfig:
    patch_size: tuple[int, int] = (16, 16)
    dropout: float = 0.0


class ViTPoolType(enum.Enum):
    cls_token = enum.auto()
    mean = enum.auto()


@dataclass
class ViTConfig:
    dim: int = 768
    embedder_config: PatchEmbedderConfig = PatchEmbedderConfig()
    transformer_config: TransformerEncoderConfig = TransformerEncoderConfig()
    pool_type: ViTPoolType = ViTPoolType.cls_token


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim: int, config: MLPConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, config: MultiHeadAttentionConfig):
        super().__init__()
        self.n_heads = config.n_heads

        # assertion
        assert dim % config.n_heads == 0, "Dimensions must be divisible by n_heads"

        head_dim = dim // config.n_heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.scale = head_dim**-0.5
        self.to_attn = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(config.dropout))
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(config.dropout))

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            rearrange(t, "b n (h d) -> b h n d", h=self.n_heads) for t in [q, k, v]
        ]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.to_attn(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        config: TransformerEncoderConfig,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(config.depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, MultiHeadAttention(dim, config.attention_config)),
                        PreNorm(dim, MLP(dim, config.mlp_config)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PatchEmbedder(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        num_channels: int,
        dim: int,
        config: PatchEmbedderConfig,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels

        image_width, image_height = image_size
        patch_width, patch_height = config.patch_size

        # assertion
        assert (
            image_width % patch_width == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size"

        num_patches = (image_width // patch_width) * (image_height // patch_height)
        patch_dim = patch_height * patch_width * num_channels

        # modules
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(config.dropout)

    def _assert_input_shape(self, x):
        assert (
            (x.shape[1] == self.num_channels)
            and (x.shape[2] == self.image_size[1])
            and (x.shape[3] == self.image_size[0])
        ), "Input shape does not match the model."

    def forward(self, x):
        self._assert_input_shape(x)

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        num_channels: int,
        num_classes: int,
        config: ViTConfig,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.pool_type = config.pool_type

        dim = config.dim
        self.embedder = PatchEmbedder(
            image_size, num_channels, dim, config.embedder_config
        )
        self.transformer = TransformerEncoder(dim, config.transformer_config)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.embedder(x)
        x = self.transformer(x)

        if self.pool_type is ViTPoolType.cls_token:
            x = x[:, 0]
        elif self.pool_type is ViTPoolType.mean:
            x = x.mean(dim=1)
        else:
            raise NotImplementedError()

        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x
