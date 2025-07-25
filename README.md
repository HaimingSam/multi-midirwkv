# multi-midirwkv
多轨音乐生成模型

## Project Description
> 略

## Method Overview
> 略

## Instructions

### Environement
```

```
### Datasets
```
cd 
python download_galidata.py
python process_data.py 
```
### Train

```
cd train
python train_midimodel.py \
--model_config /home/rwkv/cai-RWKV/Multi-midirwkv/train/hparams/midimodel.jsonl \
--midi_tokenizer_config_path ./tokenizer/tokenizer_params.json \
--midi_tokenizer_vocab_path ./tokenizer/tokenizer_vocab.json \
--jsonl_data_path /home/rwkv/cai-RWKV/Multi-midirwkv/gigamidi-test3.jsonl \
--output_dir ./result \
--num_epochs 3 \
--per_device_train_batch_size 1 \
--learning_rate 5e-5 \
--learning_rate_final 1e-6 \
--warmup_steps 200 \
--weight_decay 0.01 \
--gradient_checkpointing True \
--logging_steps 20 \
--save_steps 500 \
--wandb_project rwkv7-midi-lm-training \
--wandb_run_name midirwkv-test \
--seed 42
```
多个GPU分布式训练：
```

```

> 注：目前的通道处理方案 \
> planA 简单处理：在每个通道中仅保留一条轨道 \
> planB 合并逻辑：按通道分组 → 收集所有轨道 → 按时间排序事件 → 合并为单轨道
