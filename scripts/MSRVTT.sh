#!/bin/bash

# 指定您要使用的数据、输出和预训练模型的路径
DATA_PATH=/home/wa24301158/dataset/MSRVTT
OUTPUT_PATH=log/MSRVTT_train_b64_2_0901_source
PRETRAINED_PATH=/home/wa24301158/mywork/TempMe-master/tvr/models

# 指定要使用的GPU（例如，使用0号和1号GPU）
export CUDA_VISIBLE_DEVICES=2,3

# 启动训练
# --nproc_per_node=2 表示使用两张显卡
# --batch_size 和 --batch_size_val 是您指定的批次大小
python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --do_train 1 \
  --workers 8 \
  --batch_size 64 \
  --n_display 5 \
  --batch_size_val 16 \
  --anno_path ${DATA_PATH} \
  --epochs 2 \
  --video_path ${DATA_PATH}/all_compressed \
  --datatype msrvtt \
  --output_dir ${OUTPUT_PATH} \
  --pretrained_path ${PRETRAINED_PATH}