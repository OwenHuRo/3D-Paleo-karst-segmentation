"""
Adapted from the UCTransNet Channel Transformer implementation.

This variant combines center-relative positional encoding with reconstructed
feature spatial attention.
Original repository: https://github.com/McGregorWwww/UCTransNet
"""
import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv3d, LayerNorm
logger = logging.getLogger(__name__)
from torch.nn.modules.utils import _ntuple
from Utils import Config

import torch
import torch.nn as nn
import math


class Channel_Embeddings(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels, use_spherical=True, use_high_freq=False):
        super().__init__()
        _Tr = _ntuple(3)
        img_size = _Tr(img_size)
        patch_size = _Tr(patchsize)
        n_patches = ((img_size[0] // patch_size[0]) *
                     (img_size[1] // patch_size[1]) *
                     (img_size[2] // patch_size[2]))

        # Patch Embedding Layer
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        # 多模态位置编码参数
        self.use_spherical = use_spherical  # 是否使用球坐标系
        self.use_high_freq = use_high_freq  # 是否加入高频编码
        self.L = 0
        # 动态位置编码网络
        self.position_network = nn.Sequential(
            nn.Linear(3*(2*self.L+1), 128),  # 输入维度根据编码方式调整
            nn.ReLU(),
            nn.Linear(128, in_channels)
        )
        self.embed_dim = in_channels
        # 初始化位置编码
        self.position_embeddings = self.create_enhanced_position_embeddings(
            in_channels, img_size, patch_size
        )
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def create_enhanced_position_embeddings(self, embed_dim, img_size, patch_size):
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        position_encodings = torch.zeros(num_patches, embed_dim)

        # 计算每个 patch 的中心坐标
        patch_centers = []
        for z in range(0, img_size[2], patch_size[2]):
            for y in range(0, img_size[1], patch_size[1]):
                for x in range(0, img_size[0], patch_size[0]):
                    cx = x + patch_size[0] // 2
                    cy = y + patch_size[1] // 2
                    cz = z + patch_size[2] // 2
                    patch_centers.append((cx, cy, cz))

        # 计算全局中心
        global_center = [img_size[i] // 2 for i in range(3)]

        # 生成多模态位置编码
        for i in range(num_patches):
            dx = (patch_centers[i][0] - global_center[0]) / img_size[0]
            dy = (patch_centers[i][1] - global_center[1]) / img_size[1]
            dz = (patch_centers[i][2] - global_center[2]) / img_size[2]

            # 球坐标转换
            if self.use_spherical:
                r = torch.sqrt(torch.tensor(dx ** 2 + dy ** 2 + dz ** 2))
                theta = torch.atan2(torch.tensor(dy), torch.tensor(dx))  # 方位角
                phi = torch.atan2(torch.tensor(dz), torch.sqrt(torch.tensor(dx ** 2 + dy ** 2)))  # 仰角
                pos_input = torch.tensor([r, theta, phi], dtype=torch.float32)
            else:
                pos_input = torch.tensor([dx, dy, dz], dtype=torch.float32)

            # 高频编码
            if self.use_high_freq:
                self.L = 5
                pos_input = self.positional_encoding(pos_input, L=self.L)  # L为频率分量数
                self.position_network = nn.Sequential(
                    nn.Linear(3 * (2 * self.L + 1), 128),  # 输入维度根据编码方式调整
                    nn.ReLU(),
                    nn.Linear(128, self.embed_dim)
                )

            position_encodings[i] = self.position_network(pos_input)

        return nn.Parameter(position_encodings.unsqueeze(0))

    @staticmethod
    def positional_encoding(x, L=10):
        encodings = [x]
        for i in range(L):
            encodings.append(torch.sin(2 ** i * x))
            encodings.append(torch.cos(2 ** i * x))
        return torch.cat(encodings, dim=-1)

    def forward(self, x):
        if x is None:
            return None
        # 获取3D图像patch的embedding
        x = self.patch_embeddings(x)  # (B, C, H/P_h, W/P_w, D/P_d)
        x = x.flatten(2)  # 展平为（B,C n_patches）

        # 转置为(B, n_patches, C)
        x = x.transpose(-1, -2)

        # 获取相对位置编码
        n_patches = x.size(1)
        relative_position_encoding = self.position_embeddings[:, :n_patches, :]

        # 加入位置编码
        embeddings = x + relative_position_encoding
        embeddings = self.dropout(embeddings)
        return embeddings


class Reconstruct(nn.Module):# 相当于patch_embedding的逆操作
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = Conv3d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(num_groups=int(out_channels/4),num_channels=out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        d, h, w = int(np.cbrt(n_patch)), int(np.cbrt(n_patch)), int(np.cbrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, d, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor, mode="trilinear", align_corners=True)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class Attention_org(nn.Module):
    def __init__(self, config, vis,channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
            query2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
            query3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
            query4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
            key = nn.Linear( self.KV_size,  self.KV_size, bias=False)
            value = nn.Linear(self.KV_size,  self.KV_size, bias=False)
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.query3.append(copy.deepcopy(query3))
            self.query4.append(copy.deepcopy(query4))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])



    def forward(self, emb1,emb2,emb3,emb4, emb_all):
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_Q3_list = []
        multi_head_Q4_list = []
        multi_head_K_list = []
        multi_head_V_list = []
        if emb1 is not None:
            for query1 in self.query1:
                Q1 = query1(emb1)
                multi_head_Q1_list.append(Q1)
        if emb2 is not None:
            for query2 in self.query2:
                Q2 = query2(emb2)
                multi_head_Q2_list.append(Q2)
        if emb3 is not None:
            for query3 in self.query3:
                Q3 = query3(emb3)
                multi_head_Q3_list.append(Q3)
        if emb4 is not None:
            for query4 in self.query4:
                Q4 = query4(emb4)
                multi_head_Q4_list.append(Q4)
        for key in self.key:
            K = key(emb_all)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb_all)
            multi_head_V_list.append(V)
        # print(len(multi_head_Q4_list))

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1) if emb1 is not None else None
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1) if emb2 is not None else None
        multi_head_Q3 = torch.stack(multi_head_Q3_list, dim=1) if emb3 is not None else None
        multi_head_Q4 = torch.stack(multi_head_Q4_list, dim=1) if emb4 is not None else None
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None
        multi_head_Q3 = multi_head_Q3.transpose(-1, -2) if emb3 is not None else None
        multi_head_Q4 = multi_head_Q4.transpose(-1, -2) if emb4 is not None else None

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K) if emb1 is not None else None
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K) if emb2 is not None else None
        attention_scores3 = torch.matmul(multi_head_Q3, multi_head_K) if emb3 is not None else None
        attention_scores4 = torch.matmul(multi_head_Q4, multi_head_K) if emb4 is not None else None

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size) if emb1 is not None else None
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size) if emb2 is not None else None
        attention_scores3 = attention_scores3 / math.sqrt(self.KV_size) if emb3 is not None else None
        attention_scores4 = attention_scores4 / math.sqrt(self.KV_size) if emb4 is not None else None

        attention_probs1 = self.softmax(self.psi(attention_scores1)) if emb1 is not None else None
        attention_probs2 = self.softmax(self.psi(attention_scores2)) if emb2 is not None else None
        attention_probs3 = self.softmax(self.psi(attention_scores3)) if emb3 is not None else None
        attention_probs4 = self.softmax(self.psi(attention_scores4)) if emb4 is not None else None
        # print(attention_probs4.size())

        if self.vis:
            weights =  []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
            weights.append(attention_probs3.mean(1))
            weights.append(attention_probs4.mean(1))
        else: weights=None

        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None
        attention_probs3 = self.attn_dropout(attention_probs3) if emb3 is not None else None
        attention_probs4 = self.attn_dropout(attention_probs4) if emb4 is not None else None

        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V) if emb1 is not None else None
        context_layer2 = torch.matmul(attention_probs2, multi_head_V) if emb2 is not None else None
        context_layer3 = torch.matmul(attention_probs3, multi_head_V) if emb3 is not None else None
        context_layer4 = torch.matmul(attention_probs4, multi_head_V) if emb4 is not None else None

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() if emb1 is not None else None
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous() if emb2 is not None else None
        context_layer3 = context_layer3.permute(0, 3, 2, 1).contiguous() if emb3 is not None else None
        context_layer4 = context_layer4.permute(0, 3, 2, 1).contiguous() if emb4 is not None else None
        context_layer1 = context_layer1.mean(dim=3) if emb1 is not None else None
        context_layer2 = context_layer2.mean(dim=3) if emb2 is not None else None
        context_layer3 = context_layer3.mean(dim=3) if emb3 is not None else None
        context_layer4 = context_layer4.mean(dim=3) if emb4 is not None else None

        O1 = self.out1(context_layer1) if emb1 is not None else None
        O2 = self.out2(context_layer2) if emb2 is not None else None
        O3 = self.out3(context_layer3) if emb3 is not None else None
        O4 = self.out4(context_layer4) if emb4 is not None else None
        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        O3 = self.proj_dropout(O3) if emb3 is not None else None
        O4 = self.proj_dropout(O4) if emb4 is not None else None
        return O1,O2,O3,O4, weights


class Mlp(nn.Module):
    def __init__(self,config, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        self.attn_norm =  LayerNorm(config.KV_size,eps=1e-6)
        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        self.ffn1 = Mlp(config,channel_num[0],channel_num[0]*expand_ratio)
        self.ffn2 = Mlp(config,channel_num[1],channel_num[1]*expand_ratio)
        self.ffn3 = Mlp(config,channel_num[2],channel_num[2]*expand_ratio)
        self.ffn4 = Mlp(config,channel_num[3],channel_num[3]*expand_ratio)


    def forward(self, emb1,emb2,emb3,emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb"+str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat,dim=2)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)
        cx1,cx2,cx3,cx4, weights = self.channel_attn(cx1,cx2,cx3,cx4,emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, x4, weights





class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1,emb2,emb3,emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1,emb2,emb3,emb4, weights = layer_block(emb1,emb2,emb3,emb4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1,emb2,emb3,emb4, attn_weights


class SpatialAttention3D(nn.Module):
    """3D空间注意力模块（作用于通道注意力后的特征）"""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2  # 自动计算3D填充

        # 使用3D卷积处理体积数据
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 初始化参数
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):

        # 通道压缩（空间特征聚合）
        B, C, D, H, W = x.shape

        # 为每个通道独立计算avg和max
        avg_out = x.view(B * C, 1, D, H, W).mean(dim=1, keepdim=True)  # (B*C, 1, D, H, W)
        max_out = x.view(B * C, 1, D, H, W).max(dim=1, keepdim=True)[0]

        # 拼接通道级特征
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B*C, 2, D, H, W)

        # 逐通道卷积（每个通道独立处理）
        attn = self.conv(x_cat)  # (B*C, 1, D, H, W)
        attn = self.sigmoid(attn)

        # 恢复原始形状并加权
        attn = attn.view(B, C, D, H, W)
        return x * attn


class ChannelTransformerPlusSpatial(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config,self.patchSize_1, img_size=img_size,    in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config,self.patchSize_2, img_size=img_size//2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config,self.patchSize_3, img_size=img_size//4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config,self.patchSize_4, img_size=img_size//8, in_channels=channel_num[3])
        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,scale_factor=(self.patchSize_1,self.patchSize_1,self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,scale_factor=(self.patchSize_2,self.patchSize_2,self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,scale_factor=(self.patchSize_3,self.patchSize_3,self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,scale_factor=(self.patchSize_4,self.patchSize_4,self.patchSize_4))


        self.spatial_attention1 = self._build_spatial_attn(channel_num[0], img_size//1)
        self.spatial_attention2 = self._build_spatial_attn(channel_num[1], img_size//2)
        self.spatial_attention3 = self._build_spatial_attn(channel_num[2], img_size//4)
        self.spatial_attention4 = self._build_spatial_attn(channel_num[3], img_size//8)

    def _build_spatial_attn(self, channels, scale):
        if channels is None:
            return None
        # 根据特征图尺寸选择kernel_size
        kernel_size = 7 if scale > 64 else 3
        return SpatialAttention3D(kernel_size=kernel_size)

    def forward(self,en1,en2,en3,en4):

        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1,emb2,emb3,emb4)  # (B, n_patch, hidden)

        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None

        x1 = self.spatial_attention1(x1)
        x2 = self.spatial_attention2(x2)
        x3 = self.spatial_attention3(x3)
        x4 = self.spatial_attention4(x4)

        return x1, x2, x3, x4, attn_weights




if __name__ == "__main__":
    # 测试代码
    config = Config.get_CTranS_config()
    input_image_size = 128
    model = ChannelTransformerPlusSpatial(config=config, vis=False, img_size=128)
    en1 = torch.rand((1, 64, 128, 128, 128))  # 对应 img_size_1
    en2 = torch.rand((1, 128, 64, 64, 64))  # 对应 img_size_2
    en3 = torch.rand((1, 256, 32, 32, 32))  # 对应 img_size_3
    en4 = torch.rand((1, 512, 16, 16, 16))  # 对应 img_size_4
    x1, x2, x3, x4, attn_weights = model(en1, en2, en3, en4)
    print(x1.shape, x2.shape, x3.shape, x4.shape)

