# multi-midirwkv
多轨音乐生成模型

## Project Description


## Method Overview
> 使用miditok分词库里的合成单一流模式把多轨midi按时间步对齐成一个长序列，然后用标准rwkv7模型训练音乐生成模型

## Instructions

### Environement
```

```
### Datasets
```
python download_galidata.py
cd train-rwkv/data
python pre_data.py
python fi_data.py
python split_data.py
```
### Train
```
cd ..
python train.py \
--model_config /home/rwkv/cai-RWKV/Multi-midirwkv/train/hparams/midimodel.jsonl \
--midi_tokenizer_config_path ./tokenizer/tokenizer_params.json \
--midi_tokenizer_vocab_path ./tokenizer/tokenizer_vocab.json \
--jsonl_data_path /home/rwkv/cai-RWKV/Multi-midirwkv/gigamidi.jsonl \
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
### Inference
```
python inference.py \
    --model_path output/final_model.pt \
    --tokenizer_config tokenizer_config.json \
    --output_path generated_music.mid

```
或者
```
python inference.py \
    --model_path output/final_model.pt \
    --tokenizer_config tokenizer_config.json \
    --output_path generated_music.mid \
    --prompt_tokens "1,2,3,4,5" \
    --max_length 2000 \
    --temperature 0.8

```

## 附录：有关midi处理逻辑
https://miditok.readthedocs.io/en/latest/tokenizations.html
