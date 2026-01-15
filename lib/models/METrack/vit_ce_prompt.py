import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_prompt

_logger = logging.getLogger(__name__)


class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output


class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, me_weight=None):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C/2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        if me_weight is not None:
            rgbe_weight = me_weight[:, 0].view(me_weight.shape[0], 1, 1, 1)
            depthe_weight = me_weight[:, 1].view(me_weight.shape[0], 1, 1, 1)
            #
            x0 = self.fovea(x0 + 0.05 * rgbe_weight * x0) + x1 * (0.95 + 0.05 * depthe_weight)
            # x0 = self.fovea(x0 + 0.05 * 0.5 * x0) + x1 * (0.95 + 0.05 * 0.5)    #ablation study
        else:
            x0 = self.fovea(x0) + x1

        return self.conv1x1(x0)

class ModalityEffectiveModule(nn.Module):
    def __init__(self):
        super(ModalityEffectiveModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 64, 128)
        self.fc2 = nn.Linear(128, 2)  # 输出层神经元数量改为2
        # self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        # # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv2 = nn.Conv2d(3, 1, kernel_size=1, stride=2, padding=0)
        # self.fc1 = nn.Linear(1024, 128)
        # self.fc2 = nn.Linear(128, 2)  # 输出层神经元数量改为2
        # self.sigmoid = nn.Sigmoid()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = x.view(-1, 1024)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # x = self.sigmoid(x)  # 使用 Sigmoid 激活函数
        # epsilon = 1e-8
        # x_sum = x.sum(dim=1, keepdim=True)
        # x = x / (x_sum + epsilon)
        x = F.softmax(x, dim=1)
        return x

class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.prompt_MEM = ModalityEffectiveModule()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_prompt = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        prompt parameters
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.prompt_type = prompt_type
        # various architecture
        if self.prompt_type in ['vipt_shaw', 'vipt_deep']:
            prompt_blocks = []
            block_nums = depth if self.prompt_type == 'vipt_deep' else 1
            for i in range(block_nums):
                prompt_blocks.append(Prompt_block(inplanes=embed_dim, hide_channel=8, smooth=True))
            self.prompt_blocks = nn.Sequential(*prompt_blocks)
            prompt_norms = []
            for i in range(block_nums):
                prompt_norms.append(norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def modality_effectiveness(self, tensor_img):
        # 检查输入是否为PyTorch张量
        if not isinstance(tensor_img, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")

        # 找到张量的最小和最大值
        min_val = torch.min(tensor_img)
        max_val = torch.max(tensor_img)

        # 缩放张量的值到 [0, 1] 的范围
        tensor_img = (tensor_img - min_val) / (max_val - min_val)

        # 如果张量是CUDA上的，将其移到CPU上进行处理
        tensor_img = tensor_img.cpu()

        # 调整张量形状为[batchsize, channels, height, width]
        # if len(tensor_img.shape) != 4:
        #     raise ValueError("Input tensor must have 4 dimensions [batchsize, channels, height, width].")

        # 将张量转换为numpy数组
        img_array = tensor_img.numpy()

        # 逐通道将图像转换为灰度图
        gray_images = np.sum(img_array, axis=1) / img_array.shape[1]

        # 获取图像的高度和宽度
        batch = gray_images.shape[0]
        height = gray_images.shape[1]
        width = gray_images.shape[2]
        piexs_sum = height * width
        dark_sum = [0] * batch
        dark_prop = [0] * batch
        for i in range(batch):
            dark_sum[i] = np.sum(gray_images[i] < 0.25)  # 统计小于50的像素数
            dark_prop[i] = dark_sum[i] / piexs_sum  # 计算偏暗像素所占比例
        dark_prop = torch.Tensor(dark_prop).view([-1, 1]).to('cuda:0')
        label = torch.cat([1 - dark_prop, dark_prop], dim=1)

        return label

    # def get_MEloss(self, predict, label):
    #     loss = nn.MSELoss()
    #     return loss(predict, label)

    def normImg(self, img):
        # 找到张量的最小和最大值
        min_val = torch.min(img)
        max_val = torch.max(img)
        # 缩放张量的值到 [0, 1] 的范围
        img = (img - min_val) / (max_val - min_val)
        return img

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]

        x_copy = F.interpolate(x_rgb, size=[64, 64], mode='bilinear', align_corners=False)
        z_copy = F.interpolate(z_rgb, size=[64, 64], mode='bilinear', align_corners=False)

        x_copy = self.normImg(x_copy)
        z_copy = self.normImg(z_copy)

        x_label = self.modality_effectiveness(x_copy)
        z_label = self.modality_effectiveness(z_copy)

        x_me_weight = self.prompt_MEM(x_copy)
        z_me_weight = self.prompt_MEM(z_copy)

        # MEloss = self.get_MEloss(x_label, x_me_weight) + self.get_MEloss(z_label, z_me_weight)

        # depth thermal event images
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]
        # overwrite x & z
        x, z = x_rgb, z_rgb

        z = self.patch_embed(z)
        x = self.patch_embed(x)

        z_dte = self.patch_embed_prompt(z_dte)
        x_dte = self.patch_embed_prompt(x_dte)

        #ablation study
        z_rgbe_weight = z_me_weight[:, 0].view(z_me_weight.shape[0], 1, 1)
        z_depthe_weight = z_me_weight[:, 1].view(z_me_weight.shape[0], 1, 1)
        x_rgbe_weight = z_me_weight[:, 0].view(x_me_weight.shape[0], 1, 1)
        x_depthe_weight = z_me_weight[:, 1].view(x_me_weight.shape[0], 1, 1)
        z = z + 0.05 * z_rgbe_weight * z + z_dte * 0.05 * z_depthe_weight + z_dte * 0.95
        x = x + 0.05 * x_rgbe_weight * x + x_dte * 0.05 * x_depthe_weight + x_dte * 0.95

        '''input prompt: by adding to rgb tokens'''
        if self.prompt_type in ['vipt_shaw', 'vipt_deep']:
            z_feat = token2feature(self.prompt_norms[0](z))
            x_feat = token2feature(self.prompt_norms[0](x))
            z_dte_feat = token2feature(self.prompt_norms[0](z_dte))
            x_dte_feat = token2feature(self.prompt_norms[0](x_dte))
            z_feat = torch.cat([z_feat, z_dte_feat], dim=1)
            x_feat = torch.cat([x_feat, x_dte_feat], dim=1)
            z_feat = self.prompt_blocks[0](z_feat, z_me_weight)
            x_feat = self.prompt_blocks[0](x_feat, x_me_weight)
            z_dte = feature2token(z_feat)
            x_dte = feature2token(x_feat)
            z_prompted, x_prompted = z_dte, x_dte

            # ablation -MEP -DP
            # z = z + z_dte
            # x = x + x_dte
        # else:
        #     z = z + z_dte
        #     x = x + x_dte

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []
        removed_flag = False
        for i, blk in enumerate(self.blocks):
            '''
            add parameters prompt from 1th layer
            '''
            if i >= 1:
                if self.prompt_type in ['vipt_deep']:
                    x_ori = x
                    # recover x to go through prompt blocks
                    lens_z_new = global_index_t.shape[1]
                    lens_x_new = global_index_s.shape[1]
                    z = x[:, :lens_z_new]
                    x = x[:, lens_z_new:]
                    if removed_indexes_s and removed_indexes_s[0] is not None:
                        removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
                        pruned_lens_x = lens_x - lens_x_new
                        pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
                        x = torch.cat([x, pad_x], dim=1)
                        index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
                        C = x.shape[-1]
                        x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
                    x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
                    x = torch.cat([z, x], dim=1)

                    # prompt
                    x = self.prompt_norms[i - 1](x)  # todo
                    z_tokens = x[:, :lens_z, :]
                    x_tokens = x[:, lens_z:, :]
                    z_feat = token2feature(z_tokens)
                    x_feat = token2feature(x_tokens)

                    z_prompted = self.prompt_norms[i](z_prompted)
                    x_prompted = self.prompt_norms[i](x_prompted)
                    z_prompt_feat = token2feature(z_prompted)
                    x_prompt_feat = token2feature(x_prompted)

                    z_feat = torch.cat([z_feat, z_prompt_feat], dim=1)
                    x_feat = torch.cat([x_feat, x_prompt_feat], dim=1)
                    z_feat = self.prompt_blocks[i](z_feat, None)
                    x_feat = self.prompt_blocks[i](x_feat, None)

                    z = feature2token(z_feat)
                    x = feature2token(x_feat)
                    z_prompted, x_prompted = z, x

                    x = combine_tokens(z, x, mode=self.cat_mode)
                    # re-conduct CE
                    x = x_ori + candidate_elimination_prompt(x, global_index_t.shape[1], global_index_s)

            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "x_MEpre": x_me_weight,
            "x_MEgt": x_label,
            "z_MEpre": z_me_weight,
            "z_MEgt": z_label
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
