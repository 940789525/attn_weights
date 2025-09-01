from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists

import torch
import torch.nn.functional as F
from tvr.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from tvr.dataloaders.data_dataloaders import DATALOADER_DICT
from tvr.dataloaders.dataloader_msrvtt_retrieval import MSRVTTDataset
from tvr.models.modeling import VTRModel, AllGather
from tvr.models.optimization_adamw import AdamW, get_cosine_schedule_with_warmup
from tvr.utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

from tvr.utils.comm import is_main_process, synchronize
from tvr.utils.logger import setup_logger
from tvr.utils.metric_logger import MetricLogger

from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

allgather = AllGather.apply

global logger

# --- 解码函数 ---
def decode_text_tensor(text_tensor, tokenizer):
    # 这个函数能处理 [B, L] 和 [B, 1, L] 两种形状
    if text_tensor.ndim == 3 and text_tensor.shape[1] == 1:
        text_tensor = text_tensor.squeeze(1)
    if text_tensor.is_cuda:
        text_tensor = text_tensor.cpu()

    decoded_sentences = []
    token_id_list = text_tensor.long().tolist()
    for single_sentence_ids in token_id_list:
        decoded_tokens = tokenizer.decode(single_sentence_ids)
        cleaned_sentence = decoded_tokens.replace('<|startoftext|>', '').split('<|endoftext|>')[0].strip()
        decoded_sentences.append(cleaned_sentence)
    return decoded_sentences

# --- 核心的 Hooks 分析工具 ---
class AttentionVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.attention_maps = []
        self.hooks = []
        # 目标模块的路径
        self.target_modules = self.model.clip.transformer.resblocks

    def _hook_fn(self, module, input, output):
        _, attn_weights = output
        if attn_weights is not None:
            self.attention_maps.append(attn_weights.detach().cpu())

    def register_hooks(self):
        for module in self.target_modules:
            handle = module.register_forward_hook(self._hook_fn)
            self.hooks.append(handle)
        print(f"成功在 {len(self.hooks)} 个文本编码器层上注册了 Hooks。")

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def clear_maps(self):
        self.attention_maps = []

    def analyze_and_save(self, text_tensor, save_path="tempme_text_attention_hooks", sample_index=0):
        if not self.attention_maps:
            print("警告：未能通过 Hooks 收集到注意力图。")
            return

        decoded_sentences = decode_text_tensor(text_tensor, self.tokenizer)
        sentence = decoded_sentences[sample_index]

        token_ids_tensor = text_tensor
        if token_ids_tensor.ndim == 3 and token_ids_tensor.shape[1] == 1:
            token_ids_tensor = token_ids_tensor.squeeze(1)

        token_ids = token_ids_tensor[sample_index]
        tokens = [self.tokenizer.decoder.get(tid.item(), "UNK") for tid in token_ids]

        try:
            eot_idx = tokens.index('<|endoftext|>')
            tokens = tokens[:eot_idx+1]
        except ValueError:
            pass

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f"\n--- 使用 Hooks 分析样本 {sample_index}: '{sentence}' ---")
        for layer_idx, attn_map in enumerate(self.attention_maps):
            attention_matrix = attn_map[sample_index]
            attention_matrix = attention_matrix[:len(tokens), :len(tokens)]

            plt.figure(figsize=(10, 8))
            sns.heatmap(attention_matrix.numpy(), xticklabels=tokens, yticklabels=tokens, cmap="viridis")
            plt.title(f"Layer {layer_idx} Attention (from Hook)")
            plt.xlabel("Key (Token being attended to)")
            plt.ylabel("Query (Token doing the attending)")

            output_filename = os.path.join(save_path, f"hook_sample_{sample_index}_layer_{layer_idx}.png")
            plt.savefig(output_filename, bbox_inches='tight')
            plt.close()

        print(f"所有层的注意力热力图已通过 Hooks 保存至 '{save_path}' 文件夹。")

    def calculate_sink_scores(self, text_tensor):
        """
        计算批次中每个句子的“沉降分数”，仅使用最后一层编码器的注意力。
        分数定义为：在最后一层，所有有效token对<startoftext>的平均注意力权重。

        Args:
            text_tensor (torch.Tensor): 形状为 [B, L] 或 [B, 1, L] 的 token ID 张量。

        Returns:
            list[float]: 一个包含了批次中每个句子“沉降分数”的列表。
        """
        if not self.attention_maps:
            print("警告：未能通过 Hooks 收集到注意力图，无法计算沉降分数。")
            return [0.0] * text_tensor.shape[0]

        # --- 核心修改：只取最后一层（-1）的注意力图 ---
        # 形状: [B, L, L]
        last_layer_attn = self.attention_maps[-1]
        
        batch_size = text_tensor.shape[0]
        sink_scores = []

        # 预处理 text_tensor 以获取句子长度
        if text_tensor.ndim == 3:
            text_tensor = text_tensor.squeeze(1)
            
        for i in range(batch_size):
            # 提取所有token对<startoftext> (第一列) 的注意力
            # 形状: [L]
            attn_to_start_token = last_layer_attn[i, :, 0]

            # 确定句子的实际长度（不包括padding）
            actual_length = (text_tensor[i] != 0).sum().item()
            
            # 只考虑实际token的注意力（忽略padding部分）
            if actual_length > 0:
                valid_attns = attn_to_start_token[:actual_length]
                # 计算平均值，得到该句子的最终“沉降分数”
                sink_score = valid_attns[1:].mean().item()
            else:
                sink_score = 0.0 # 处理空句子的情况
                
            sink_scores.append(sink_score)
            
        return sink_scores

def get_args(description='Temporal Token Merging for Efficient Text-Video Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0, help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0, help="Whether to run evaluation.")

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--anno_path', type=str, default='data/MSR-VTT/anns', help='annotation path')
    parser.add_argument('--video_path', type=str, default='data/MSR-VTT/videos', help='video path')
    parser.add_argument('--pretrained_path', type=str, default="your_path", help='pretrained model path')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--clip_lr', type=float, default=6e-4, help='learning rate')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=32, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')

    parser.add_argument("--device", default='cuda', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distribted training")
    parser.add_argument("--local-rank", default=0, type=int, help="distribted training")
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")

    parser.add_argument('--n_display', type=int, default=50, help='Information display frequence')
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--base_encoder", default="ViT-B/32", type=str, help="Choose a CLIP version")

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    
    parser.add_argument('--lora_dim', type=int, default=8)

    parser.add_argument('--tome_r', type=int, default=2)
    parser.add_argument('--tome_tracesource', type=bool, default=False)
    parser.add_argument('--tome_propattn', type=bool, default=True)

    ### 12--9-->6--10-->3--11-->1
    parser.add_argument('--merge_layer', type=str, default='8-9-10') # start from 0
    parser.add_argument('--merge_frame_num', type=str, default='2-2-3')

    ### R_c = 100% - 30% = 70%; R_I = 100% - 10% = 90%
    parser.add_argument('--merge_token_proportion', type=str, default='30-10')
    parser.add_argument('--frame_pos', type=int, default=1)
    
    args = parser.parse_args()

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if torch.cuda.is_available():
        torch.distributed.barrier()
    logger.info("local_rank: {} world_size: {}".format(args.local_rank, args.world_size))

    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and world_size parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.world_size, args.batch_size_val, args.world_size))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def build_model(args):
    model = VTRModel(args)
    if args.init_model:   # 这个参数也可以用于继续训练模型
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device) # 将模型部署到高性能的GPU
    return model


def build_dataloader(args):
    ## ####################################
    # dataloader loading
    ## ####################################
    tokenizer = ClipTokenizer()
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if isinstance(test_length, int):
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    elif len(test_length) == 2:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %dv %dt", test_length[0], test_length[1])
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d %d", len(test_dataloader[0]), len(test_dataloader[1]))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %dv %dt", val_length[0], val_length[1])

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", len(train_dataloader) * args.epochs)
    else:
        train_dataloader, train_sampler = None, None

    return test_dataloader, val_dataloader, train_dataloader, train_sampler


def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    if hasattr(model, 'module'):
        model = model.module
    clip_lr = args.clip_lr  # 1e-7
    weight_decay = args.weight_decay  # 0.2
    warmup_proportion = args.warmup_proportion
    param_optimizer = list(model.named_parameters())


    for name, param in param_optimizer:
        if "TVPt" in name:  # 检查参数的完整名称中是否包含子字符串 "TVPt"
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    
    optimizer_parameters_prompt = []
    enabled_prompt = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled_prompt.append(name)
            optimizer_parameters_prompt.append(param)
    logger.info(f"Tuned Parameters: {sorted(enabled_prompt)}")

    optimizer_grouped_params = [
        {'params': optimizer_parameters_prompt, 'lr': args.clip_lr}
    ]

    optimizer = AdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    num_warmup_steps = int(warmup_proportion * num_train_optimization_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)

    return optimizer, scheduler, model


def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(args.output_dir, "{}.pth".format(type_name))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def prompt_save_model(epoch, args, model, type_name=""):
    assert "Not Implement" == 0
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(args.output_dir, "{}.pth".format(type_name))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def reduce_loss(loss, args):
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss


# 将原来的 train_epoch 函数替换为这个新版本
def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                scheduler, global_step, max_steps, val_dataloader, visualizer):
    global logger
    global best_score
    global best_score_list
    global meters
    global sim_matrix_num
    global sim_name_list

    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0

    end = time.time()
    logit_scale = 0
    for step, batch in enumerate(train_dataloader, start=1):
        global_step += 1
        data_time = time.time() - end

        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        text_ids, text_mask, video, video_mask, inds, idx = batch
        # 在主进程中，每次前向传播前清空上一批次的注意力图
        visualizer.clear_maps()
        # 1. 计算原始的检索损失
        retrieval_loss = model(text_ids, text_mask, video, video_mask, idx, global_step)
        if n_gpu > 1:
            retrieval_loss = retrieval_loss.mean()

        # ====================================================================
        # ============== 新增：长度自适应沉降分数正则化 ==============
        # ====================================================================
        sink_score_loss = torch.tensor(0.0).to(device) # 初始化为0

        lambda_reg = 0.1  # <-- 这是一个需要您进行实验调整的超参数
        # a. 获取当前批次的沉降分数
        sink_scores_tensor = torch.tensor(visualizer.calculate_sink_scores(text_ids), device=device)

        # b. 获取当前批次每个句子的长度
        sentence_lengths = text_mask.sum(dim=1).float()

        # c. 为批次中的每个句子计算动态目标分数上限
        min_len, max_len = 5.0, 20.0
        target_short, target_long = 0.85, 0.75
        clamped_lengths = torch.clamp(sentence_lengths, min_len, max_len)
        target_scores = target_short + ((clamped_lengths - min_len) / (max_len - min_len)) * (target_long - target_short)
        target_scores = target_scores.to(device)
            
        # d. 计算修正后的正则化损失
        diff = sink_scores_tensor - target_scores
        sink_score_loss = F.relu(diff).mean()

        # e. 将正则化损失加入总损失
        total_loss = retrieval_loss + lambda_reg * sink_score_loss

        optimizer.zero_grad()

        with torch.autograd.detect_anomaly():  # 一个上下文管理器（Context Manager），它开启了PyTorch的“异常检测”模式  用于检测反向传播出现的错误或者异常值
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪 梯度的总体大小都不会超过一个设定的上限（1.0）
        
        optimizer.step()

        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule

        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.module.clip.logit_scale.exp().item()
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.clip.logit_scale.exp().item()

        batch_time = time.time() - end
        end = time.time()

        # 同步并记录损失
        reduced_retrieval_l = reduce_loss(retrieval_loss.detach(), args)
        reduced_sink_l = reduce_loss(sink_score_loss.detach(), args)
        meters.update(time=batch_time, data=data_time, retrieval_loss=float(reduced_retrieval_l), sink_loss=float(reduced_sink_l))


        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if (global_step % log_step == 0 or global_step == 1) and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "epoch: {epoch}/{max_epoch}",
                        "iteration: {iteration}/{max_iteration}",
                        "{meters}",
                        "lr: {lr}",
                        "logit_scale: {logit_scale:.2f}"
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    max_epoch=args.epochs,
                    iteration=global_step,
                    max_iteration=max_steps,
                    meters=str(meters),
                    lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(scheduler.get_last_lr())))]),
                    logit_scale=logit_scale,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if (global_step % (log_step * 3) == 0)  or global_step == 1:
            max_R1 = eval_epoch(args, model, val_dataloader, args.device, ClipTokenizer())
            if args.local_rank == 0:
                for list_idx in range(sim_matrix_num):
                    if best_score_list[list_idx] < max_R1[list_idx]:
                        best_score_list[list_idx] = max_R1[list_idx]
                    logger.info("The R1 is: {:.4f}\t| {:.4f}\tin {}".format(max_R1[list_idx], best_score_list[list_idx],sim_name_list[list_idx]))

                if best_score < max(max_R1):
                    best_score = max(max_R1)
                    output_model_file = save_model(epoch, args, model, type_name="best")
                logger.info("The best R1 is: {:.4f} at all".format(best_score))

            synchronize()
            model.train()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def eval_epoch(args, model, test_dataloader, device, tokenizer):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()

    visualizer = None
    if not args.distributed or args.local_rank == 0:
        visualizer = AttentionVisualizer(model, tokenizer)
        visualizer.register_hooks()

    # ----------------------------
    # 1. cache the features
    # ----------------------------
    batch_cls, batch_mask_t = [], []
    batch_video_feat, batch_mask_v = [], []
    batch_ids = []

    all_sink_scores = []
    all_inds = []
    all_sentence_lengths = [] 

    with torch.no_grad():
        tic = time.time()

        sim_matrix = []

        logger.info('[start] extract')
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):

            if visualizer:
                visualizer.clear_maps()

            batch = tuple(t.to(device) for t in batch)
            text_ids, text_mask, video, video_mask, inds, _ = batch  # inds 是句子的索引，得到索引可以通过查看 MSRVTT_JSFUSION_test.csv 这个文件找到该索引对应的句子
            # --- 1. 获取并存储句子长度 ---
            sentence_lengths = text_mask.sum(dim=1).cpu().numpy()
            all_sentence_lengths.extend(sentence_lengths)

            cls, video_feat = model.stage1_eval(text_ids, text_mask, video, video_mask)
            if visualizer:
                scores = visualizer.calculate_sink_scores(text_ids)
                # --- 存储当前批次的分数和索引 ---
                all_sink_scores.extend(scores)
                all_inds.extend(inds.cpu().numpy())


            batch_cls.append(cls)
            batch_mask_t.append(text_mask)
            batch_video_feat.append(video_feat)
            batch_mask_v.append(video_mask)
            batch_ids.append(inds)

        torch.distributed.barrier()
        
        batch_ids = allgather(torch.cat(batch_ids, dim=0), args).squeeze()  # shape->torch.Size([1000])  1000个句子(验证集所有句子)的索引拼接在一起
        
        batch_cls = allgather(torch.cat(batch_cls, dim=0), args) # shape -> torch.Size([1000, 512])  1000个句子的特征拼接在一起
        batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args) # torch.Size([1000, 32]) 1000个句子的长度标识拼接在一起
        batch_video_feat = allgather(torch.cat(batch_video_feat, dim=0), args)  # 1000个视频特征 也是可以通过MSRVTT_JSFUSION_test.csv 这个文件找到具体是哪个视频  注意可能在这1000个测试集视频中没有重复的视频
        batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)  # 1000个视频有效帧(长度) 标识
        
        batch_cls[batch_ids] = batch_cls.clone()  # 排序  让句子和视频按照 索引进行排序 之后就是 [句子0特征，句子1特征，句子2特征,...]这样的顺序了，以此类推下面的代码
        batch_mask_t[batch_ids] = batch_mask_t.clone()
        batch_video_feat[batch_ids] = batch_video_feat.clone()   # 按照 句子索引进行调整位置 这样一起调整还是配对的  例如 [0]->(text44,video7701) 调整后为 [44]->(text44,video7701)
        batch_mask_v[batch_ids] = batch_mask_v.clone()
        
        batch_cls = batch_cls[:batch_ids.max() + 1, ...]  # 清理可能出现的多余数据(因分布式评估出现的多余数据)，确保用于评估的张量形状为torch.Size([1000, 512])
        batch_mask_t = batch_mask_t[:batch_ids.max() + 1, ...]
        batch_video_feat = batch_video_feat[:batch_ids.max() + 1, ...]
        batch_mask_v = batch_mask_v[:batch_ids.max() + 1, ...]
        logger.info('[finish] extract')
        
        logger.info('[start] calculate the similarity')


        with torch.no_grad():
            mini_batch = args.batch_size_val
            sim_matrix = []
            
            batch_cls_split = torch.split(batch_cls, mini_batch)  # 进行分割  例如mini_batch=16时 就是 1000/16 62.xxx  最后一组剩下8个句子 组成一组 最后分成63组 tuple(元组类型)
            batch_mask_t_split = torch.split(batch_mask_t, mini_batch)
            batch_video_feat_split = torch.split(batch_video_feat, mini_batch)
            batch_mask_v_split = torch.split(batch_mask_v, mini_batch)
            
            for cls, text_mask in tqdm(zip(batch_cls_split, batch_mask_t_split)):  # 遍历所有句子 每次取 batch_size个句子
                each_row = []
                for video_feat, video_mask in zip(batch_video_feat_split, batch_mask_v_split):  # 遍历所有视频  每次取 batch_size个视频
                    logits = model.stage2_eval(cls, text_mask, video_feat, video_mask)
                    logits = logits.cpu().detach().numpy()
                    each_row.append(logits)
                each_row = np.concatenate(tuple(each_row), axis=-1)  # 62 * 16 + 8 = 992 + 8 = 1000   得到的 each_eow 其 shape = (16,1000)   # 它代表了当前外层循环处理的那16个文本，与整个数据集中全部1000个视频的相似度得分
                sim_matrix.append(each_row)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)  # 得到的sim_matrix形状为 [1000,1000]  sim_matrix[i, j] 这个值，代表了整个数据集中，第 i 个文本与第 j 个视频的最终相似度得分
        logger.info('[finish] calculate the similarity')
        
        
    logger.info('[start] compute_metrics')
    logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1])) 
    global sim_name_list
    
    max_R1=[]
    list_idx = 0
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    logger.info("Eval {} ...".format(sim_name_list[list_idx]))
    logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['R50'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['R50'], vt_metrics['MR'], vt_metrics['MeanR']))
    max_R1.append(tv_metrics['R1'])
    per_sample_ranks = tv_metrics["cols"]

    # if not args.distributed or args.local_rank == 0:
    #     # 将收集到的 sink_scores 按原始索引排序
    #     # 创建一个 (索引, 分数) 的元组列表
    #     indexed_scores = sorted(zip(all_inds, all_sink_scores))
    #     indexed_lengths = sorted(zip(all_inds, all_sentence_lengths)) # <--- 新增对齐句子长度
    #     # 排序后只保留分数，现在它的顺序和 per_sample_ranks 一致了
    #     sorted_sink_scores = [score for ind, score in indexed_scores]
    #     sorted_lengths = [length for ind, length in indexed_lengths] # <--- 获取排序后的长度

    #     # 使用 pandas 进行数据分析
    #     import pandas as pd
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns

    #     # 创建 DataFrame
    #     df = pd.DataFrame({
    #         'sentence_index': list(range(len(per_sample_ranks))),
    #         'sink_score': sorted_sink_scores,
    #         'retrieval_rank': per_sample_ranks,
    #         'sentence_length': sorted_lengths  # <--- 新增列
    #     })

    #     # --- 第4步：进行关联性分析 ---

    #     # 1. 计算相关系数
    #     correlation = df['sink_score'].corr(df['retrieval_rank'])
    #     logger.info(f"沉降分数与检索排名的皮尔逊相关系数为: {correlation:.4f}")
    #     # (解读：接近-1表示强负相关，即分数越高、排名越靠前(越好)；接近1表示强正相关；接近0表示无线性关系)

    #     # 2. 绘制散点图进行可视化
    #     plt.figure(figsize=(10, 6))
    #     sns.scatterplot(data=df, x='sink_score', y='retrieval_rank', alpha=0.5)
    #     plt.title('Sink Score vs. Retrieval Rank')
    #     plt.xlabel('Sink Score (Higher is more attention to <start>)')
    #     plt.ylabel('Retrieval Rank (Lower is better)')
    #     plt.grid(True)
    #     # 保存图像
    #     scatter_plot_path = os.path.join(args.output_dir, "sink_score_vs_rank_scatter.png")
    #     plt.savefig(scatter_plot_path)
    #     plt.close()
    #     logger.info(f"散点图已保存至: {scatter_plot_path}")

    #     # 3. 分箱分析 (Binning Analysis) - 更稳健的分析方法
    #     # 将句子按沉降分数分为几组，然后比较每组的平均检索性能
    #     try:
    #         df['score_bin'] = pd.qcut(df['sink_score'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
    #         bin_metrics = df.groupby('score_bin')['retrieval_rank'].agg(['mean', 'median', lambda x: (x < 1).sum() / len(x) * 100])
    #         bin_metrics.rename(columns={'<lambda_0>': 'R@1 (%)'}, inplace=True)
    #         logger.info("按沉降分数分箱后的性能分析:\n" + str(bin_metrics))
    #     except Exception as e:
    #         logger.warning(f"分箱分析失败: {e}")

    #     # --- 4. 新增：探究句子长度与沉降分数的关系 ---
    #     logger.info("\n" + "="*20 + " 句子长度 vs. 沉降分数分析 " + "="*20)

    #     # a. 计算句子长度和沉降分数的相关系数
    #     corr_length_score = df['sentence_length'].corr(df['sink_score'])
    #     logger.info(f"句子长度与沉降分数的皮尔逊相关系数为: {corr_length_score:.4f}")
        
    #     # b. 绘制句子长度 vs. 沉降分数的散点图
    #     plt.figure(figsize=(10, 6))
    #     sns.scatterplot(data=df, x='sentence_length', y='sink_score', alpha=0.5)
    #     plt.title('Sentence Length vs. Sink Score')
    #     plt.xlabel('Sentence Length (Number of Tokens)')
    #     plt.ylabel('Sink Score')
    #     plt.grid(True)
    #     length_vs_score_path = os.path.join(args.output_dir, "length_vs_score_scatter.png")
    #     plt.savefig(length_vs_score_path)
    #     plt.close()
    #     logger.info(f"句子长度与沉降分数的散点图已保存至: {length_vs_score_path}")

    #     # c. 按句子长度分箱，查看每一组的平均沉降分数
    #     try:
    #         df['length_bin'] = pd.qcut(df['sentence_length'], q=4, labels=['Shortest', 'Short', 'Long', 'Longest'], duplicates='drop')
    #         length_bin_metrics = df.groupby('length_bin')['sink_score'].agg(['mean', 'median', 'std'])
    #         logger.info("按句子长度分箱后的沉降分数分析:\n" + str(length_bin_metrics))
    #     except Exception as e:
    #         logger.warning(f"句子长度分箱(vs. 沉降分数)分析失败: {e}")

    #     # d. (可选但推荐) 直接分析句子长度和检索性能的关系，作为参照
    #     logger.info("\n" + "="*20 + " 句子长度 vs. 检索性能分析 " + "="*20)
    #     corr_length_rank = df['sentence_length'].corr(df['retrieval_rank'])
    #     logger.info(f"句子长度与检索排名的皮尔逊相关系数为: {corr_length_rank:.4f}")
    #     try:
    #         # 使用上面已创建的 length_bin
    #         length_perf_metrics = df.groupby('length_bin')['retrieval_rank'].agg(['mean', 'median', lambda x: (x < 1).sum() / len(x) * 100])
    #         length_perf_metrics.rename(columns={'<lambda_0>': 'R@1 (%)'}, inplace=True)
    #         logger.info("按句子长度分箱后的性能分析:\n" + str(length_perf_metrics))
    #     except Exception as e:
    #         logger.warning(f"句子长度分箱(vs. 性能)分析失败: {e}")
        
    #     # --- 新增：绘制句子长度 vs. 检索性能的散点图 ---
    #     plt.figure(figsize=(10, 6))
    #     sns.scatterplot(data=df, x='sentence_length', y='retrieval_rank', alpha=0.5)
    #     # 添加趋势线以便更清晰地观察关系
    #     sns.regplot(data=df, x='sentence_length', y='retrieval_rank', scatter=False, color='red', line_kws={'linewidth':2})
    #     plt.title('Sentence Length vs. Retrieval Rank')
    #     plt.xlabel('Sentence Length (Number of Tokens)')
    #     plt.ylabel('Retrieval Rank (Lower is better)')
    #     plt.grid(True)
    #     length_vs_rank_path = os.path.join(args.output_dir, "length_vs_rank_scatter.png")
    #     plt.savefig(length_vs_rank_path)
    #     plt.close()
    #     logger.info(f"句子长度与检索排名的散点图已保存至: {length_vs_rank_path}")

    #             # --- 新增: 识别并可视化“反常”高性能短句 ---
    #     target_indices = df[(df['retrieval_rank']==0)&(df['sentence_length']<7)&(df['sentence_index']<350)].index.tolist()
    #     if visualizer and target_indices:
    #         logger.info(f"\n--- 开始第二轮遍历以生成 {len(target_indices)} 个高性能短句的注意力热力图 ---")
    #         logger.info(f"目标句子索引: {target_indices}")
    #         visualized_count = 0
    #         for batch in tqdm(test_dataloader, desc="Visualization Pass"):
    #             visualizer.clear_maps()
    #             batch = tuple(t.to(device) for t in batch)
    #             text_ids, text_mask, video, video_mask, inds, _ = batch
                
    #             # 必须执行前向传播以触发hooks
    #             _ = model.stage1_eval(text_ids, text_mask, video, video_mask)
                
    #             # 检查当前批次是否包含我们的目标句子
    #             for target_idx in target_indices:
    #                 if target_idx in inds.cpu().tolist():
    #                     batch_pos = (inds.cpu() == target_idx).nonzero(as_tuple=True)[0].item()
    #                     save_path = os.path.join(args.output_dir, "attention_maps_anomalies_r1")
    #                     visualizer.analyze_and_save(
    #                         text_tensor=text_ids,
    #                         save_path=save_path,
    #                         sample_index=batch_pos
    #                     )
    #                     visualized_count += 1
                
    #             if visualized_count == len(target_indices):
    #                 logger.info("所有目标句子的注意力图已生成完毕。")
    #                 break


    if visualizer:
        visualizer.remove_hooks()
        print("所有 Hooks 已被成功移除。")

    return max_R1

def main():
    global logger
    global best_score
    global best_score_list
    global meters
    global sim_matrix_num
    global sim_name_list

    sim_name_list = ['base'] 
    sim_matrix_num = len(sim_name_list)

    meters = MetricLogger(delimiter="  ")
    args = get_args()
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('tvr', args.output_dir, args.local_rank)

    args = set_seed_logger(args)

    model = build_model(args)   # 这里得到的model是加载了预训练权重 或者 是加载过 之前训练权重的模型

    test_dataloader, val_dataloader, train_dataloader, train_sampler = build_dataloader(args)
    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        # --- 新增: 在训练开始前初始化 AttentionVisualizer ---
        # 注意：hooks会增加开销，只在主进程中为正则化开启

        # 这里的model是未被DDP包装的原始模型
        visualizer = AttentionVisualizer(model, ClipTokenizer())
        visualizer.register_hooks()

        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        _max_steps = len(train_dataloader) * args.epochs
        optimizer, scheduler, model = prep_optimizer(args, model, _max_steps, args.local_rank)

        best_score = 0.00001
        best_score_list = [0.00001 for _ in range(sim_matrix_num)]
        best_output_model_file = "None"
        global_step = 0
        for epoch in range(args.epochs):
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            synchronize()
            torch.cuda.empty_cache()
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader,
                                                        args.device, args.world_size, optimizer,
                                                        scheduler, global_step, max_steps, val_dataloader,
                                                        visualizer) # <-- 传入 visualizer
            torch.cuda.empty_cache()

            max_R1 = eval_epoch(args, model, val_dataloader, args.device, ClipTokenizer())
            torch.cuda.empty_cache()
            synchronize()

            if args.local_rank == 0:
                for list_idx in range(sim_matrix_num):
                    if best_score_list[list_idx] < max_R1[list_idx]:
                        best_score_list[list_idx] = max_R1[list_idx]
                    logger.info("The R1 is: {:.4f}\t| {:.4f}\tin {}".format(max_R1[list_idx], best_score_list[list_idx],sim_name_list[list_idx]))

                if best_score < max(max_R1):
                    best_score = max(max_R1)
                    output_model_file = save_model(epoch, args, model, type_name="best")
                logger.info("The best R1 is: {:.4f} at all".format(best_score))

            synchronize()

        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n' + f'training finished with {training_time}' + "*" * 20 + '\n')

        if args.local_rank == 0:
            with open("{}_{}.txt".format(args.output_dir, best_score),'w') as f:
                f.write(' ')

    elif args.do_eval:
        eval_epoch(args, model, test_dataloader, args.device, ClipTokenizer())


if __name__ == "__main__":
    main()
