# multi-midirwkv
多轨音乐生成模型

## Project Description


## Method Overview


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



## 附录：有关midi处理逻辑

