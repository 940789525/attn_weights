# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch
import logging

logger = logging.getLogger(__name__)

def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    cls_num: int,
    r_f: int
) -> Tuple[Callable, Callable]:
    # 第一次执行 metric  合并度量  [768,50,64]  768 = 64*12 是所有图片(帧)数量  50是token数  64 是每个注意力头(默认12头，每个头特征维度都是64)平均后的特征值
    metric_cls = metric[:,:cls_num,:]  # 取出该批次中所有图片帧的cls  第一次执行[768,1,64]   第9层 [192,2,768]  192 = 384/2
    metric = metric[:,cls_num:,:]  # 去除cls  [768, 49, 64]  第9层[192,66,768]  66 = 34*2-2

    if r <= 0:  # 不进行token合并
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)  #  L2归一化 (L2 Normalization) 或 单位化 将每一个Token的“平均Key向量”（即metric中的每一个向量）都转换成一个长度为1的单位向量，从而为后续高效地计算余弦相似度做准备
        if cls_num % 2 == 0:
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
        else:  # 以间隔的方式取出Token得到集合a和b，然后为计算a和b中每个Token的相似度做准备
            a, b = metric[..., 1::2, :], metric[..., ::2, :]
        scores = a @ b.transpose(-1, -2)  # 得到间隔token的相似度得分  第9层 [192,33,33]

        node_max, node_idx = scores.max(dim=-1)  # node_max 每行(最后一个维度)的最大值  node_idx是这个最大值在最后一个维度对应的索引(index)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # 得到倒数第二维度相似度的排序  [...,None] 会增加一个维度 edge_idx.shape [768,24,1]  

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens  不进行合并的token得分索引   第9层 [192,13,1]
        src_idx = edge_idx[..., :r, :]  # Merged Tokens   准备进行合并的token得分索引  第9层 [192,20,1]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # 在node_idx中找 src_idx的值所对应的真实值  第9层[192,20,1]

        unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:  # 第一层 x.shape[768,50,768]  可以认为有两种模式  一种传入特征x,一种传入权重size
        x_cls = x[..., :cls_num, :]  # 取到 cls 应该没有进行帧间token合并时，都是cls.shape  [???,1,768]
        x_patch = x[..., cls_num:, :]  # 取到所有token(patch)特征
        if cls_num % 2 == 0:  # cls_num 为偶数
            src, dst = x_patch[..., ::2, :], x_patch[..., 1::2, :]
        else:  # cls_num为奇数
            dst, src = x_patch[..., ::2, :], x_patch[..., 1::2, :]  # 间隔取token，并分成两个集合
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))  # 选取 保留的 src 即 scores中的行(dim=-2)的维度 也就是 当前将保留的奇数token(patch)的索引  执行之后unm包含了所有在src集合中未被合并的Token
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c)) # 根据“牺牲者索引”src_idx，从src集合中抓取出即将被合并的Token  也是奇数token 要进行合并的索引  执行之后src只包含了要被合并掉的源头Token
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)   # dst集合中的被选择要进行融合的Token现在融合了来自src集合的某个对应Token的信息
        # 以上的mode参数，在第一层时是mode="sum",因此是直接将src加到了dst_idx记录的dst对应index的token上的
        return torch.cat([x_cls, unm, dst], dim=1)   # 返回了融合之后的token，并除去一半进行融合的token  注意这里返回时会将token顺序打乱

    return merge, None  # 函数返回一个元组，其中第一个元素merge是一个“专家函数”，它已经封装了所有关于如何合并当前序列的具体指令（即各种索引）；第二个元素None是一个占位符。


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:  # 第一层 size 一开始是全1 代表了权重，用于token合并后进行平均   第二层size已经初始化过了，其中size值不为1的元素表示进行了token合并
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")   # 得到token进行合并后(是直接相加)的特征  x*size 会将合并了n个token的对应token特征值乘n   第9层 [192,48,768]   48 = 68-20
    size = merge(size, mode="sum")    # 得到合并每个token的权重，即此时每个token的累计数量，例如进行一次合并的token对应位置的值为2

    x = x / size    # 进行token的平均，得到最终的帧内token合并结果  这里的x是所有帧(图片的)特征 第一层执行后形状为[384,48,768]
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source