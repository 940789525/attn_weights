from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .module_tome_patch import apply_patch as tome_patch
from .module_tome_utils import parse_r
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL
import numpy as np
import copy
allgather = AllGather.apply
allgather2 = AllGather2.apply

logger = logging.getLogger(__name__)

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


class VTRModel(nn.Module):
    def __init__(self, config):
        super(VTRModel, self).__init__()
        
        self.config = config
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        self.lora_dim = config.lora_dim
        logger.info("v_LoRA: {} dim".format(self.lora_dim))
        
        assert backbone in _PT_NAME
        model_path = os.path.join(config.pretrained_path, _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.merge_layer = [int(_l) for _l in config.merge_layer.split('-')]  # 确定了 在哪里 合并  默认[8,9,10]  即在第 9，10，11层进行合并
        self.merge_frame_num = [int(_l) for _l in config.merge_frame_num.split('-')]  # 确定了 如何 合并（合并比例） 默认 [2,2,3]   12->6->3->1   12/2=6 6/2=3 3/3=1
        frame_num_list=[]  # 记录了合并过程中片段数量的变化，用于生成相应阶段的位置编码，帮助模型更好地学习时间关系
        frame_num = config.max_frames  # 在循环后代表了最终生成的视频表示的数量
        for _l in range(len(self.merge_layer)):
            frame_num_list.append(frame_num)
            frame_num = frame_num // self.merge_frame_num[_l]
        logger.info('Position_embedding: {}'.format(frame_num_list)) 
        
        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, self.lora_dim, 
                        self.merge_layer, config.frame_pos, frame_num_list)
            
        self.loss_fct = CrossEn(config)

        self.clip.load_state_dict(state_dict, strict=False)  # strict=False 参数可以不需要严格匹配，即self.clip可以有新的模块

        # 图像内空间Token合并（Image Merging）
        self.tome_r = config.tome_r  # 每层合并的Token数量
        self.tome_tracesource = config.tome_tracesource  # 追述来源 没什么用 False
        self.tome_propattn = config.tome_propattn   # 用来控制如何进行token融合  True  可能是直接平均融合  或者是按照某种权重进行融合
        logger.info("tome: {} r | {} tracesource | {} propattn".format(self.tome_r, self.tome_tracesource, self.tome_propattn))
        
        logger.info("merge_layer: {}".format(config.merge_layer))
        logger.info("merge_frame_num: {}".format(config.merge_frame_num))
        logger.info("merge_token_proportion: {}".format(config.merge_token_proportion))  # 配置了在片段合并时，Token级别的合并强度？？？？
        logger.info("frame_pos: {}".format(config.frame_pos))   # 配置了模型最底层的空间位置理解方式   就是是否要进行帧间的token合并的 config.frame_pos=1 表示进行帧间token合并

        self.merge_token_proportion = [int(_l) / 100 for _l in config.merge_token_proportion.split('-')]   #  从0~100 化成 0~1 并用list（列表）进行保存
        self.frame_pos = config.frame_pos
        
        self.merge_layer = [int(_l) for _l in config.merge_layer.split('-')]   # 用列表进行保存  进行token帧间合并的layer
        self.merge_frame_num = [int(_l) for _l in config.merge_frame_num.split('-')]   # 用列表进行保存  进行token帧间合并的强度
        self.TVPt_Video_Positional_embedding = []
        if config.base_encoder == "ViT-B/32":
            patch_num = 50
        else:
            patch_num = 197
        cls_num = 1
        frame_num = config.max_frames
        self.patch_list = [patch_num]
        self.frame_list = [frame_num]
        for _l in range(12):
            if _l not in self.merge_layer: # 如果_l(当前遍历到的层，不是进行token帧间合并的层)
                if _l < self.merge_layer[0]:  # 如果当前的_l层是在所有帧间合并层前
                    patch_num = patch_num - self.tome_r   # 当前层的token(patch)总数  即减去当前帧进行帧内合并的token
                    self.patch_list.append(patch_num)   # 记录每层patch(token)数的列表
                    self.frame_list.append(frame_num)  # 记录当前层(_l)时的总帧数
                else:
                    patch_num = patch_num - int(patch_num * self.merge_token_proportion[1])
                    self.patch_list.append(patch_num)
                    self.frame_list.append(frame_num)
            else:
                M_frame_num = self.merge_frame_num.pop(0)  # 进行合并的帧数(一次使用多少帧进行合并)
                M_token_num = int(patch_num * M_frame_num * self.merge_token_proportion[0])   # 跨片段合并的移除比例，对应论文中的 1 - R_c
                # M_token_num: 最终要移除的Token总数。
                assert frame_num % M_frame_num == 0  # 可以正好进行合并
                patch_num = patch_num * M_frame_num - M_token_num   # 计算跨片段合并之后，新形成的那个更长的片段包含了多少个Token
                cls_num = cls_num * M_frame_num  # 计算了合并后，新的大片段中有多少个[CLS] Token
                frame_num = frame_num // M_frame_num  # 计算合并后，还剩下多少个片段
                self.patch_list.append(patch_num)  # 将跨片段合并完成后的状态记录下来
                self.frame_list.append(frame_num)   # 记录当前层的帧数
                # 在完成跨片段合并后，还会立即进行一次片段内合并，进一步压缩Token
                patch_num = patch_num - int(patch_num * self.merge_token_proportion[1])
                self.patch_list.append(patch_num)
                self.frame_list.append(frame_num)
        
        self.merge_layer = [int(_l) for _l in config.merge_layer.split('-')]    # 字符串转列表
        self.merge_frame_num = [int(_l) for _l in config.merge_frame_num.split('-')]   # 字符串转列表
            
        tome_patch(self.clip, trace_source=self.tome_tracesource, prop_attn=self.tome_propattn)
        
    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:  # 五维
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)  # [384,3,224,224]
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        cls = self.get_text_feat(text_ids, text_mask)  # cls 是文本编码器的最终输出 即文本特征   [32,512]
        video_feat = self.get_video_feat(video, video_mask)  #  video_feat 是视频编码器最终输出 即视频特征  [32,512]
        
        cls = allgather(cls, self.config)  # 多gpu训练，拼接每个gpu的批次，形成一个大的张量
        video_feat = allgather(video_feat, self.config)
        torch.distributed.barrier()  # 等待所有gpu处理完成
        
        logit_scale = self.clip.logit_scale.exp()
        loss = 0.
        
        t_feat = cls / cls.norm(dim=-1, keepdim=True)  # 归一化，将每个文本对应的向量都化为单位向量
        v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True) # 归一化 将每个视频对应的向量都化为单位向量

        t2v_logits = torch.einsum('td,vd->tv', [t_feat, v_feat])  # 矩阵乘法 得到相似度矩阵

        loss_t2v = self.loss_fct(t2v_logits * logit_scale)
        loss_v2t = self.loss_fct(t2v_logits.T * logit_scale)
        loss = (loss_t2v + loss_v2t) / 2  # 最终的损失值
        
        return loss

    def stage1_eval(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        cls = self.get_text_feat(text_ids, text_mask)
        video = self.get_video_feat(video, video_mask)

        return cls, video

    def stage2_eval(self, cls, text_mask, video_feat, video_mask):
        logit_scale = self.clip.logit_scale.exp()
        
        t_feat = cls / cls.norm(dim=-1, keepdim=True) 
        v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True) 

        t2v_logits = torch.einsum('td,vd->tv', [t_feat, v_feat])
        
        return t2v_logits * logit_scale

    def get_text_feat(self, text_ids, orig_mask):
        b = text_ids.size(0)
        x = self.clip.token_embedding(text_ids) 
        max_t_len = x.size(1)
        pos_emd = self.clip.positional_embedding[:max_t_len, :]
        x = x + pos_emd

        mask = orig_mask  # [batch,len]   len 表示最长的句子token数量(一般为32)
        text_length = max_t_len  #  最大句子token数
        attn_mask = self.clip.build_attention_mask(text_length).repeat(x.size(0), 1, 1).to(mask.device)  # [batch,len,len]  因果掩码（三角矩阵）
        inf = torch.zeros((text_length, text_length)).fill_(float("-inf")).repeat(x.size(0), 1, 1).to(mask.device)  # inf: 创建一个用于填充的负无穷张量 所有元素都填充为负无穷的张量
        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)  # [batch,len,len]
        attn_mask = torch.where(mask>0, attn_mask, inf)  # 结合两种掩码，得到最终attn_mask
    
        x = self.clip.transformer(x, attn_mask)  # 将已经准备好的初始文本特征 x，送入完整的文本编码器（Text Transformer）进行深度处理，并得到最终的文本特征输出

        hidden = self.clip.ln_final(x) @ self.clip.text_projection  # 将文本编码器输出的高维特征，经过最终的归一化和线性投影，映射到与视觉特征对齐的多模态共享空间中
        cls = hidden[torch.arange(hidden.shape[0]), text_ids.argmax(dim=-1)]  # 精确抽取[EOS] Token作为最终句子表征

        cls = cls.float() # 将cls张量的数据类型显式地转换为32位浮点数
        cls = cls.view(b, -1, cls.size(-1)).squeeze(1)  # 确保cls张量具有确定的、正确的二维形状 (batch_size, embed_dim)
        return cls

    def get_video_feat(self, video, video_mask):
        self.clip._tome_info["size"] = None
        self.clip._tome_info["source"] = None
        self.clip._tome_info["cls_num"] = 1
        self.clip._tome_info["frame_num"] = self.frame_list[0]
        self.clip._tome_info["token_num"] = self.patch_list[0]

        self.merge_frame_num = [int(_l) for _l in self.config.merge_frame_num.split('-')]
        
        b, n_f = video_mask.size()
        org_n_f = n_f
        x = video  #  [768,3,224,224]   768 = 12*64  帧数*batch
            
        x = self.clip.visual.conv1(x)  

        x = x.reshape(x.shape[0], x.shape[1], -1) # [768,768,49]
        x = x.permute(0, 2, 1)  # [768,49,768]
        x = torch.cat(
            [self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)    # 拼接 cls
        
        x = x + self.clip.visual.positional_embedding.to(x.dtype)  # 给x添加位置编码
        x = self.clip.visual.ln_pre(x)  # 层归一化
        
        _, token_len, d_v = x.size()

        pos_count = 0
        for res_i, res_block in enumerate(self.clip.visual.transformer.resblocks):
            if res_i not in self.merge_layer:  # 非合并层  不进行跨帧合并的层
                if res_i < self.merge_layer[0]: # 第一个指定的合并层之前的所有层（默认即第0层到第7层）
                    x = res_block(x, M_frame_num=1, M_token_num=[self.tome_r])
                else:
                    M_token_num = int(self.clip._tome_info["token_num"] * self.merge_token_proportion[1])
                    M_token_num = min((self.clip._tome_info["token_num"] - self.clip._tome_info["cls_num"]) // 2, M_token_num)
                    x = res_block(x, M_frame_num=1, M_token_num=[M_token_num])
            else:
                M_frame_num = self.merge_frame_num.pop(0)   # 第一层经过该分支 x.shape [???,34,768]  M_frame_num 取出当前要进行合并的帧间 token数
                M_token_num_0 = int(self.clip._tome_info["token_num"] * M_frame_num * self.merge_token_proportion[0])  # self.clip._tome_info["token_num"]--当前每张图片token数   M_frame_num--要进行帧间token合并的帧数  self.merge_token_proportion[0]---？？？（合并尺度？）
                M_token_num_0 = min((self.clip._tome_info["token_num"] - self.clip._tome_info["cls_num"]) * M_frame_num // 2, M_token_num_0)  # 定义了将要进行的（跨片段\跨帧）合并的“数量”
                M_token_num_1 = int((self.clip._tome_info["token_num"] * M_frame_num - M_token_num_0) * self.merge_token_proportion[1])  # ？？？
                M_token_num_1 = min( ( (self.clip._tome_info["token_num"] - self.clip._tome_info["cls_num"]) * M_frame_num - M_token_num_0) // 2, M_token_num_1)  #  定义了第二次（片段内）合并的“数量”
                    
                x = res_block(x, M_frame_num=M_frame_num, M_token_num=[M_token_num_0, M_token_num_1], frame_pos=self.frame_pos)   # 返回进行帧间(不是严格的帧间)和帧内合并token的结果
        
        n_f = self.clip._tome_info["frame_num"]
        token_len = self.clip._tome_info["token_num"]
        cls_num = self.clip._tome_info["cls_num"]
        x = x.view(b, n_f, token_len, d_v)[:,:,:cls_num,:].reshape(b,org_n_f,d_v)  # [32,97,768]->[32,12,768]
        hidden = self.clip.visual.ln_post(x) @ self.clip.visual.proj
        video_feat = hidden.float()
        
        video_feat = video_feat.contiguous()
        
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        video_feat = self.get_video_avg_feat(video_feat, video_mask)  # 得到的video_feat就是代表整个视频的特征
        
        return video_feat

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un  # 先对“video_mask_un”广播，扩展成形状一样的张量，然后进行对应位置乘法
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.   # 防止出现除0错误
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum  # 先合并每帧的特征，然后在除帧数进行合并
        return video_feat

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
