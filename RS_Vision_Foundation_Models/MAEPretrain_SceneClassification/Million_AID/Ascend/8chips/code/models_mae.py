# Copyright 2021, 2022, 2023 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022, 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import luojianet_ms
from luojianet_ms import nn
from luojianet_ms import ops as P
from luojianet_ms import Tensor
from luojianet_ms.common.initializer import initializer, Normal, XavierUniform, Constant
from luojianet_ms import dtype as mstype

import numpy as np
from functools import partial


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=bias)
        self.norm = nn.LayerNorm((embed_dim,), epsilon=1e-6).to_float(mstype.float32) if norm_layer else P.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        """
        In static GRAPH_MODE, can not use 'assert' in forward() function.
        Reference:
            https://www.mindspore.cn/docs/zh-CN/r1.8/note/static_graph_syntax_support.html
        """
        # assert H == self.img_size[0], 'Input image height does not match model'
        # assert W == self.img_size[1], 'Input image width does not match model'
        if H != self.img_size[0]:
            raise ValueError("Input image height :{} does not match model".format(self.img_size[0]))
        if W != self.img_size[1]:
            raise ValueError("Input image width :{} does not match model".format(self.img_size[1]))

        x = self.proj(x)
        # if self.flatten:
        #     x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.flatten:
            x = P.Reshape()(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
            x = P.Transpose()(x, (0, 2, 1))

        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.qkv = nn.Dense(dim, dim * 3, weight_init='he_uniform', has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=(1. - attn_drop))
        self.proj = nn.Dense(dim, dim, weight_init='he_uniform')
        self.proj_drop = nn.Dropout(keep_prob=(1. - proj_drop))

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = P.Reshape()(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = P.Transpose()(qkv, (2, 0, 3, 1, 4))
        # q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q, k, v = P.Unstack(axis=0)(qkv)   # make torchscript happy (cannot use tensor as tuple)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        k = P.Transpose()(k, (0, 1, 3, 2))
        attn = P.matmul(q, k) * self.scale
        # attn = attn.softmax(dim=-1)
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = P.matmul(attn, v)
        x = P.Transpose()(x, (0, 2, 1, 3))
        x = P.Reshape()(x, (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        # self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.gamma = luojianet_ms.Parameter(init_values * P.Ones()(dim, mstype.float32))

    def forward(self, x):
        # return x.mul_(self.gamma) if self.inplace else x * self.gamma
        return P.Mul()(x, self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor = luojianet_ms.numpy.empty(shape).bernoulli_(P=keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        # random_tensor.div_(keep_prob)
        random_tensor = P.Div()(random_tensor, keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # bias = to_2tuple(bias)
        # drop_probs = to_2tuple(drop)
        bias = (bias, bias)
        drop_probs = (drop, drop)

        # self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1 = nn.Dense(in_features, hidden_features, weight_init='he_uniform', has_bias=bias[0])
        self.act = act_layer()
        # self.drop1 = nn.Dropout(drop_probs[0])
        self.drop1 = nn.Dropout(keep_prob=(1. - drop_probs[0]))
        # self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.fc2 = nn.Dense(hidden_features, out_features, weight_init='he_uniform', has_bias=bias[1])
        # self.drop2 = nn.Dropout(drop_probs[1])
        self.drop2 = nn.Dropout(keep_prob=(1. - drop_probs[1]))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def _init_weights(model):
    # initialize nn.Linear and nn.LayerNorm
    for name, cell in model.cells_and_names():
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(init=XavierUniform(), shape=cell.weight.shape, dtype=cell.weight.dtype))
            if isinstance(cell, nn.Dense) and cell.has_bias is not None:
                cell.bias.set_data(initializer(init=Constant(value=0), shape=cell.bias.shape, dtype=cell.bias.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer(init=Constant(value=1), shape=cell.gamma.shape, dtype=cell.gamma.dtype))
            cell.beta.set_data(initializer(init=Constant(value=0), shape=cell.beta.shape, dtype=cell.beta.dtype))
    return model


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.norm1 = norm_layer((dim,), epsilon=1e-05)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else P.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else P.Identity()

        # self.norm2 = norm_layer(dim)
        self.norm2 = norm_layer((dim,), epsilon=1e-05)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        # self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else P.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else P.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, mask_ratio=0.75, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.cls_token = luojianet_ms.Parameter(P.Zeros()((1, 1, embed_dim), mstype.float32))
        self.pos_embed = luojianet_ms.Parameter(P.Zeros()((1, num_patches + 1, embed_dim), mstype.float32), requires_grad=False)  # fixed sin-cos embedding

        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(depth)])
        self.blocks = nn.CellList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        # self.norm = norm_layer(embed_dim)
        self.norm = norm_layer((embed_dim,), epsilon=1e-05)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed = nn.Dense(embed_dim, decoder_embed_dim, weight_init='he_uniform', has_bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token = luojianet_ms.Parameter(P.Zeros()((1, 1, decoder_embed_dim), mstype.float32))

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = luojianet_ms.Parameter(P.Zeros()((1, num_patches + 1, decoder_embed_dim), mstype.float32), requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        self.decoder_blocks = nn.CellList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        # self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_norm = norm_layer((decoder_embed_dim,), epsilon=1e-05)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.decoder_pred = nn.Dense(decoder_embed_dim, patch_size**2 * in_chans, weight_init='he_uniform', has_bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed.set_data(P.ExpandDims()(Tensor(pos_embed, mstype.float32), 0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        self.decoder_pos_embed.set_data(P.ExpandDims()(Tensor(decoder_pos_embed, mstype.float32), 0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        weight = self.patch_embed.proj.weight.data
        channels, bs, h, w = weight.shape
        weight = P.Reshape()(weight, (channels, bs * h * w))
        weight = initializer(init=XavierUniform(), shape=weight.shape, dtype=mstype.float32)
        weight = P.Reshape()(weight, (channels, bs, h, w))
        self.patch_embed.proj.weight.set_data(initializer(init=XavierUniform(), shape=weight.shape, dtype=mstype.float32))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)
        self.cls_token = initializer(init=Normal(sigma=0.02, mean=0.0), shape=self.cls_token.shape, dtype=mstype.float32)
        self.mask_token = initializer(init=Normal(sigma=0.02, mean=0.0), shape=self.mask_token.shape, dtype=mstype.float32)

        # initialize nn.Linear and nn.LayerNorm
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     # we use xavier_uniform following official JAX ViT:
        #     torch.nn.init.xavier_uniform_(m.weight)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        if imgs.shape[2] != imgs.shape[3]:
            raise ValueError("image height :{} does not match image width".format(imgs.shape[2]))

        h = w = imgs.shape[2] // p
        # x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = P.Reshape()(imgs, (imgs.shape[0], 3, h, p, w, p))
        # x = torch.einsum('nchpwq->nhwpqc', x)
        x = P.Transpose()(x, (0, 2, 4, 3, 5, 1))
        # x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        x = P.Reshape()(x, (imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        # assert h * w == x.shape[1]
        if h * w != x.shape[1]:
            raise ValueError("h * w :{} does not match feature.shape[1]".format(x.shape[1]))

        # x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = P.Reshape()(x, (x.shape[0], h, w, p, p, 3))
        # x = torch.einsum('nhwpqc->nchpwq', x)
        x = P.Transpose()(x, (0, 5, 1, 3, 2, 4))
        # imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        imgs = P.Reshape()(x, (x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))  # the remained token number

        # noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        noise = P.UniformReal()((N, L))  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        # ids_shuffle = torch.argsort(noise, dim=1)
        # ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_shuffle = P.Sort(axis=1)(noise)
        ids_restore = P.Sort(axis=1)(ids_shuffle[1].astype(mstype.float32))

        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        ids_keep = ids_shuffle[1][:, :len_keep]
        x_masked = P.GatherD()(x, 1, luojianet_ms.numpy.tile(P.ExpandDims()(ids_keep, -1), (1, 1, D)))

        # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=x.device)
        mask = P.Ones()((N, L), mstype.float32)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask for computing loss
        # mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = P.GatherD()(mask, 1, ids_restore[1])

        return x_masked, mask, ids_restore[1]

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]  # cls_token doesn't have positional encod.

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        cls_tokens = P.BroadcastTo((x.shape[0], -1, -1))(cls_token)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = P.Concat(axis=1)((cls_tokens, x))

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # x: [N, L+1, D]
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        mask_tokens = luojianet_ms.numpy.tile(self.mask_token, (x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1))
        x_ = P.Concat(axis=1)(([x[:, 1:, :], mask_tokens]))  # no cls token

        # inversely random sequence
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        x_ = P.GatherD()(x_, 1, luojianet_ms.numpy.tile(P.ExpandDims()(ids_restore, -1), (1, 1, x.shape[2])))  # unshuffle
        x = P.Concat(axis=1)(([x[:, :1, :], x_]))

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(axis=-1, keep_dims=True)
            var = target.var(axis=-1, keepdims=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(axis=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        # return loss, pred, mask
        return loss


def mae_vit_base_patch16_dec512d8b(**kwargs):
    # model = MaskedAutoencoderViT(
    #     patch_size=16, embed_dim=768, depth=12, num_heads=12,
    #     decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    #     mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)

    model = _init_weights(model)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    # model = MaskedAutoencoderViT(
    #     patch_size=16, embed_dim=1024, depth=24, num_heads=16,
    #     decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    #     mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model = _init_weights(model)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    # model = MaskedAutoencoderViT(
    #     patch_size=14, embed_dim=1280, depth=32, num_heads=16,
    #     decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    #     mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model = _init_weights(model)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


if __name__ == '__main__':
    from luojianet_ms import context
    device_target = 'CPU'
    context.set_context(mode=context.PYNATIVE_MODE, device_target=device_target)
    input = luojianet_ms.Tensor(shape=(1, 3, 224, 224), dtype=mstype.float32, init=Normal())
    print('input:')
    print(input.shape)

    model = mae_vit_base_patch16(mask_ratio=0.75, norm_pix_loss=False)
    loss = model(input)
    print('loss:')
    print(loss)