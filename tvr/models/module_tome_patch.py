# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple, Union, List

import torch
import torch.nn.functional as F
from .module_clip import Attention, ResidualAttentionBlock, CLIP
from .module_tome_merge import bipartite_soft_matching, merge_source, merge_wavg
import logging

logger = logging.getLogger(__name__)

### modified from ToMe
class ToMeBlock(ResidualAttentionBlock):

    def forward(self, x: torch.Tensor, M_frame_num: int = 1, M_token_num: List[int] = [2], frame_pos: int = 0) -> torch.Tensor:
        r_f = M_frame_num # 就是进行帧间合并的帧数
        if r_f > 1:
            r = M_token_num[0]  # r 代表 在当前跨帧合并步骤中，需要合并（或移除）的Token数量  第9层是 20  第10层是 26
            metric = x.detach()  # mertric和x的值相同，但是metric被移出了梯度反向传播
            bsz, token_num, embed_dim = x.shape
            cls_num = self._tome_info["cls_num"]  # 第9层是1 第10层是2 第11层是4
            
            if self._tome_info["size"] is None:  # 第9层 size不是None，这里size记录了当前进行帧内token合并的相关信息
                self._tome_info["size"] = torch.ones_like(x[..., 0, None])
            
            x = x.reshape(bsz // r_f, r_f, token_num, embed_dim)  # [384, 34, 768]->[192, 2, 34, 768]
            metric = metric.reshape(bsz // r_f, r_f, token_num, embed_dim) # [384, 34, 768]->[192, 2, 34, 768]
            info_size = self._tome_info["size"].reshape(bsz // r_f, r_f, token_num, 1) # [384, 34, 1] ->  [192, 2, 34, 1]
            
            x_cls = x[:, :, :cls_num, :].reshape(bsz // r_f, -1, embed_dim) # [192, 2, 768]
            x_patch = x[:, :, cls_num:, :].reshape(bsz // r_f, -1, embed_dim) # [192, 66, 768]
            x = torch.cat([x_cls, x_patch], dim=1) # [192, 68, 768]
            
            metric_cls = metric[:, :, :cls_num, :].reshape(bsz // r_f, -1, embed_dim)  # [192,2,768]
            metric_patch = metric[:, :, cls_num:, :].reshape(bsz // r_f, -1, embed_dim)  # [192,66,768]
            metric = torch.cat([metric_cls, metric_patch], dim=1)  # 第九层,这里得到的metric和x是相等的
            
            info_size_cls = info_size[:, :, :cls_num, :].reshape(bsz // r_f, -1, 1)  # [192, 2, 1]  值全为1，应该cls就是没有参与token合并
            info_size_patch = info_size[:, :, cls_num:, :].reshape(bsz // r_f, -1, 1)  # [192,66,1]
            self._tome_info["size"] = torch.cat([info_size_cls, info_size_patch], dim=1) # [192,68,1]

            if frame_pos == 1:  # 第九层默认走这条分支 第10层也走这条路
                Position_Embed = self.TVPt_Video_Positional_embedding.reshape(1, self._tome_info["frame_num"] // r_f, r_f, 1, embed_dim)  # [1,6,2,1,768]  第9层张量的值目前是全0  第10层也是全0，可能需要反向传播更新
                Position_Embed = Position_Embed.expand(bsz // self._tome_info["frame_num"], -1, -1, token_num, -1).reshape(bsz // r_f, r_f, token_num, embed_dim) # [192,2,34,768]

                Position_Embed_cls = Position_Embed[:, :, :cls_num, :].reshape(bsz // r_f, -1, embed_dim)  # [192,2,768]
                Position_Embed_patch = Position_Embed[:, :, cls_num:, :].reshape(bsz // r_f, -1, embed_dim) # [192,66,768]
                Position_Embed = torch.cat([Position_Embed_cls, Position_Embed_patch], dim=1)  # [192,68,768]

            self._tome_info["cls_num"] = cls_num * r_f  # 2   cls数量
            self._tome_info["token_num"] = token_num * r_f  # 68  当前每帧token数
            self._tome_info["frame_num"] = self._tome_info["frame_num"] // r_f  # 6  当前帧数
            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_matching(
                    metric,  # [192,68,768]  [整个批次总帧数,每帧token数,dim]
                    r,  # 20 进行合并的token数
                    cls_num=self._tome_info["cls_num"],  # 2 cls总数
                    r_f=r_f  # 2
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = merge_source(
                        merge, x, self._tome_info["source"]
                    )
                ### add position embedd
                if frame_pos == 1:
                    x = x + Position_Embed  # 第九层 Position_Embed 全为0
                
                x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
                self._tome_info["token_num"] = self._tome_info["token_num"] - r
        
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None  # 是否开启“传播注意力（Propagate Attention）” 默认为False(第一层)  第二次就是Ture了
        x_attn, metric = self.attn(self.ln_1(x), attn_size) # x_attn：应用了自注意力之后的特征输出 metric：用于决定如何合并Token的“合并度量”  x.shape [768,50,768]  第一个 768=12*64 (帧数*batch) 50为token数 第二个768是特征维度
        # 在ToMeBlock的后续代码中，x_attn将用于标准的残差连接和MLP计算，而metric将被送入Token合并函数，来实际地减少序列的长度。
        x = x + x_attn

        r = M_token_num[-1]  # 进行帧内合并的token数  第9层是4 第10层是6
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,  # 合并度量  [768,50,64]  768 = 64*12 是所有图片(帧)数量  50是token数  64 是每个注意力头(默认12头，每个头特征维度都是64)平均后的特征值
                r,  # 代表在当前这一层，总共要合并（移除）多少个Token
                cls_num=self._tome_info["cls_num"],  # 告诉函数在输入的Token序列中，开头有多少个是特殊的[CLS] Token。在TempMe的ImgMe阶段，每一帧都有一个[CLS] Token，所以这个值通常是batch_size * 1。
                r_f=1
            )
            if self._tome_info["trace_source"]:  # 默认False 或者 第一层为False 第二层也是False
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
            self._tome_info["token_num"] = self._tome_info["token_num"] - r  # 得到当前每帧图片的token数量(合并token后的数量)
        
        x = x + self.mlp(self.ln_2(x))
        
        return x  # [p_num,token_num,dim]   p_num(当前的总帧数)  token_num(当前每帧的token数)

class ToMeAttention(Attention):

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, tgt_len, embed_dim = x.size()  # bsz：64*12  
        
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        qkv_delta = F.linear(x, self.TVPt_LoRA_a)  # 用线性层代替矩阵乘法  [768,50,8]
        qkv_delta = F.linear(qkv_delta, self.TVPt_LoRA_b).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)
        q,k,v = q+q_delta,k+k_delta,v+v_delta  # 组合低秩矩阵结果和冻结参数计算结果
        
        q = q * self.scaling  # 缩放因子
        attn = (q @ k.transpose(-2, -1))  # QK
        if size is not None:  # 偏置（Bias）”注意力得分
            attn = attn + size.log()[:, None, None, :, 0]   # 更多原始Token合并而来的“大”Token，人为地增加其注意力权重  
            #注意这里的size.log()得到的都是正数，而attn有正也有负，再softmax之前，attn的值越大，就表示QK相似度越高。
            # 因此这里要更关注某个token，就增加size.log()是合理的
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)
        
        return x, k.mean(1)  # k.mean(1)对每个注意力头(heads)进行平均 为序列中的每一个Token(patch)生成一个统一的、综合了所有语义子空间的“平均Key向量”

def apply_patch(
    model: CLIP, trace_source: bool = False, prop_attn: bool = True
):
    model._tome_info = {
        "frame_num": 0,
        "token_num": 0,
        "cls_num": 0,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn
    }

    for module in model.visual.transformer.modules():  # 深度遍历CLIP模型视觉部分的所有子模块
        if isinstance(module, ResidualAttentionBlock): # 判断当前遍历到的模块是否是一个ResidualAttentionBlock的实例
            module.__class__ = ToMeBlock   # 原本这个module是ResidualAttentionBlock类的实例，执行这行代码后，它就变成了ToMeBlock类的实例
            module._tome_info = model._tome_info   # 将我们之前创建的全局状态字典的引用，赋给了每一个被修改后的模块
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention   # 将标准的Attention类替换为带有ToMe逻辑的ToMeAttention类
            module._tome_info = model._tome_info
    print("end_for")