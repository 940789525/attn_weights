from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import torch


def compute_metrics(x):

    # 1. 对每一行进行降序排序
    # -x 将所有得分取负，这样从小到大排序就等价于对原得分从大到小排序。
    # axis=1 表示沿着行的方向操作。
    # sx 的每一行现在都是一个从高到低排列的相似度得分列表。
    sx = np.sort(-x, axis=1)
    # 2. 提取所有正确匹配项的得分
    # np.diag(-x) 会提取出 sim_matrix 对角线上的元素（同样取负）。
    # d 现在是一个一维数组，包含了 score(t0,v0), score(t1,v1), ...
    d = np.diag(-x)
    
    # 3. 将 d 转换为列向量
    # d.shape 从 (1000,) 变为 (1000, 1)。
    # 这是为了利用NumPy的广播机制进行下一步的计算。
    d = d[:, np.newaxis]
    # 4. 找到每个正确匹配项在其所在行的排名（rank）
    # 这是一个非常巧妙的步骤。
    # sx (1000, 1000) - d (1000, 1)
    # NumPy会将列向量 d 广播，使得 sx 的每一行都减去该行对应的正确匹配得分。
    # 结果：对于每一行，只有原先等于正确匹配得分的那个位置，现在会变成 0。
    ind = sx - d

    # np.where(ind == 0) 会返回一个元组，其中包含了所有值为0的元素的坐标。
    # ind[0] 是行坐标（[0, 1, 2, ...]），ind[1] 是列坐标。
    # 我们需要的是列坐标，因为它代表了正确匹配项在排序后的列表中的位置（即排名，从0开始）。

    ind = np.where(ind == 0)
    ind = ind[1]
    #  ind 现在是一个一维数组，形状为 (1000,)。
    # ind[i] 的值，就是第 i 个文本的正确视频被排在了第几位（例如，值为0表示排名第一）。
    metrics = {}
    # R@1 (Recall at 1): 排名为0（即第一名）的样本数占总样本数的百分比。
    # np.sum(ind == 0) 计算了排名为0的样本总数。
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    # R@5 (Recall at 5): 排名前5（即排名 < 5）的样本数占总样本数的百分比。
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    # R@10 (Recall at 10): 排名前10的样本数占总样本数的百分比。
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    # R@50 (Recall at 50): 排名前50的样本数占总样本数的百分比。
    metrics['R50'] = float(np.sum(ind < 50)) * 100 / len(ind)
    # MR (Median Rank): 所有正确匹配项排名的中位数。
    # +1 是因为排名是从0开始的，而我们通常习惯从1开始计数。
    # 这个值越小越好。
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    # MeanR (Mean Rank): 所有正确匹配项排名的平均值。
    # 这个值也是越小越好。
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    r50 = metrics['R50']
    mr = metrics['MR']
    meanr = metrics["MeanR"]
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - R@50: {:.4f} - Median R: {} - MeanR: {}'.format(r1, r5, r10, r50,
                                                                                          mr, meanr))


# below two functions directly come from: https://github.com/Deferf/Experiments
def tensor_text_to_video_metrics(sim_tensor, top_k=[1, 5, 10, 50]):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim=-1, descending=True)
    second_argsort = torch.argsort(first_argsort, dim=-1, descending=False)

    # Extracts ranks i.e diagonals
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1=1, dim2=2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1=0, dim2=2))
    mask = ~ torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    # A quick dimension check validates our results, there may be other correctness tests pending
    # Such as dot product localization, but that is for other time.
    # assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])
    if not torch.is_tensor(valid_ranks):
        valid_ranks = torch.tensor(valid_ranks)
    results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results['MR'] = results["MedianR"]
    return results


def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)
    # Code to avoid nans
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    # Forms a similarity matrix for use with rank at k
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T


if __name__ == '__main__':
    test_sim = np.random.rand(1000, 1000)
    metrics = compute_metrics(test_sim)
    print_computed_metrics(metrics)
