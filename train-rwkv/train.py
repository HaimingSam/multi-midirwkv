import multiprocessing as mp
import platform
import sys
import gc
import signal
import atexit

# 根据操作系统设置多进程启动方法
if platform.system() == 'Windows':
    mp.set_start_method('spawn', force=True)
else:
    mp.set_start_method('fork', force=True)

# 完全禁用PyTorch Dynamo编译以避免Triton编译器错误
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_JIT"] = "0"  # 禁用 JIT 编译
os.environ["TORCHINDUCTOR_DISABLE"] = "1"  # 禁用 Inductor
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 同步CUDA操作
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 限制CUDA内存分配

import torch
torch.backends.cudnn.enabled = False  # 禁用 cuDNN 加速（临时测试）
torch.set_num_threads(1)  # 限制 CPU 线程数
torch.backends.cuda.matmul.allow_tf32 = False  # 禁用 TF32
torch.backends.cudnn.allow_tf32 = False  # 禁用 cuDNN TF32

# 设置信号处理器
def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, cleaning up...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 清理函数
def cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print("Cleanup completed successfully")
    except Exception as e:
        print(f"Cleanup failed: {e}")

atexit.register(cleanup)

from torch.utils.data import DataLoader, DistributedSampler
from transformers import HfArgumentParser
from dataclasses import dataclass, field
import logging
import json
from typing import Optional, List
import time
import wandb
import numpy as np
import random
import math

#导入分词器库
from miditok import REMI  
# 导入标准RWKV模型
from model import RWKV

logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    """Command line arguments for training the RWKV model"""
    model_config: str = field(
        metadata={"help": "Path to the RWKV model config."}
    )
    midi_tokenizer_config_path: str = field(
        metadata={"help": "Path to the MIDI Tokenizer config YAML file."}
    )
    midi_tokenizer_vocab_path: str = field(
        metadata={"help": "Path to the MIDI Tokenizer vocab file."}
    )
    jsonl_data_path: str = field(
        metadata={"help": "Path to the JSONL file containing training data."}
    )
    output_dir: str = field(
        metadata={"help": "Directory to save the trained model and checkpoints."}
    )
    max_tokens_k: Optional[int] = field(
        default=None, 
        metadata={"help": "Maximum number of tokens in K units (1024s). Used for batch size control."}
    )
    # --- Training Hyperparameters ---
    num_epochs: int = field(default=3, metadata={"help": "Number of training epochs."})
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Training batch size per device."}
    )
    learning_rate: float = field(default=1e-5, metadata={"help": "Initial learning rate."})
    learning_rate_final: float = field(default=1e-6, metadata={"help": "Final learning rate."})
    warmup_steps: int = field(default=200, metadata={"help": "Number of warmup steps."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay."})
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Enable gradient checkpointing."}
    )
    # --- Logging and Saving ---
    logging_steps: int = field(default=20, metadata={"help": "Log every N steps."})
    save_steps: int = field(default=500, metadata={"help": "Save a checkpoint every N steps."})
    wandb_project: str = field(
        default="rwkv7-midi-lm-training", metadata={"help": "Name of W&B project."}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "Name of W&B run."}
    )
    # --- System and Environment ---
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training."})
    seed: int = field(default=42, metadata={"help": "Random seed."})

def padded_collate_fn(features):
    """
    填充函数，同批次填充到最大长度
    """
    if not features:
        return []
    
    # 找到批次中的最大长度
    max_length = max(len(f['midi_tokens']) for f in features)
    max_length = min(max_length, 14336)
    
    padded_features = []
    padding_stats = {
        'total_sequences': len(features),
        'max_length': max_length,
        'total_tokens': 0,
        'padding_tokens': 0
    }
    
    for feature in features:
        tokens = feature['midi_tokens']
        original_length = len(tokens)
        
        # 如果序列太长，先截断
        if original_length > max_length:
            tokens = tokens[:max_length]
            original_length = max_length
            logger.debug(f"Truncated sequence from {len(feature['midi_tokens'])} to {max_length}")
        
        # 如果序列太短，用0填充
        if original_length < max_length:
            padding_length = max_length - original_length
            tokens = tokens + [0] * padding_length
            padding_stats['padding_tokens'] += padding_length
        
        padding_stats['total_tokens'] += max_length
        padded_features.append({'midi_tokens': tokens})
    
    # 记录填充统计信息
    padding_ratio = padding_stats['padding_tokens'] / padding_stats['total_tokens']
    if padding_ratio > 0.3:
        logger.warning(f"High padding ratio: {padding_ratio:.2%} ({padding_stats['padding_tokens']}/{padding_stats['total_tokens']} tokens)")
    else:
        logger.debug(f"Padding ratio: {padding_ratio:.2%} ({padding_stats['padding_tokens']}/{padding_stats['total_tokens']} tokens)")
    
    return padded_features

def process_batch(features, vocab_size, device, args):
    """
    Processes a batch of raw data for standard RWKV model.
    注意：features已经通过padded_collate_fn填充，所有序列长度一致
    """
    processed_features = []

    # Process each feature
    for feature in features:
        midi_tokens = feature.get('midi_tokens')
        
        # 检查是否缺少 MIDI 数据
        if midi_tokens is None:
            logger.warning("Skipping sample due to missing MIDI tokens.")
            continue

        # 验证输入数据的有效性
        if not isinstance(midi_tokens, list) or len(midi_tokens) == 0:
            logger.warning("Skipping sample due to invalid MIDI tokens format.")
            continue

        # 将MIDI tokens转换为张量
        midi_tokens_tensor = torch.tensor(midi_tokens, dtype=torch.long)
        midi_tokens_tensor = midi_tokens_tensor.to(device)

        processed_features.append({
            'midi': midi_tokens_tensor
        })

    if not processed_features:
        return {}

    # Constants
    ignore_id = -100

    batch_input_ids = []
    batch_labels = []

    for p_feature in processed_features:
        midi_tokens = p_feature['midi']  # Shape: [T]

        # 直接使用处理后的 midi_tokens 作为 input_ids
        input_ids = midi_tokens

        # 通过左移 input_ids 生成 labels
        labels = torch.full_like(input_ids, ignore_id)
        labels[:-1] = input_ids[1:].clone()

        # 关键：将填充位置的labels设为-100，确保这些位置不参与损失计算
        padding_mask = (input_ids == 0)  # 0是填充token
        labels[padding_mask] = ignore_id  # 将填充位置的标签设为-100

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)

    # 在生成 final_labels 前，对每个样本的 labels 进行检查
    for i in range(len(batch_labels)):
        labels = batch_labels[i]
        # 检查当前样本的 labels 是否全为 -100
        if (labels == -100).all():
            logger.warning(f"Skipping all-ignore sample (index {i})")
            # 从批次中移除该样本
            batch_input_ids.pop(i)
            batch_labels.pop(i)
            # 修正索引（因列表长度变化）
            i -= 1

    # 若批次为空，直接返回空字典
    if not batch_input_ids:
        return {}

    # Stack into batch tensors
    final_input_ids = torch.stack(batch_input_ids, dim=0)
    final_labels = torch.stack(batch_labels, dim=0)

    # 检查形状
    if final_input_ids.dim() != 2:
        raise ValueError(f"final_input_ids must have shape (B, T), but got {final_input_ids.shape}")
    if final_labels.dim() != 2:
        raise ValueError(f"final_labels must have shape (B, T), but got {final_labels.shape}")

    # 基于最大token数动态调整批次大小
    if args.max_tokens_k is not None:
        maximum_tokens = args.max_tokens_k * 1024
        current_batch_size, current_batch_seq_len = final_input_ids.shape
        max_batch_size = maximum_tokens // current_batch_seq_len
        
        if max_batch_size < current_batch_size:
            logger.info(f'max_batch_size < current_batch_size, max_batch_size: {max_batch_size}, current_batch_size: {current_batch_size} shrink the batch size')
            final_input_ids = final_input_ids[:max_batch_size]
            final_labels = final_labels[:max_batch_size]
            logger.info(f'Adjusted batch size to {max_batch_size}, new shape: {final_input_ids.shape}')

    return {
        "input_ids": final_input_ids,
        "labels": final_labels
    }

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def configure_optimizer(model, args):
    """Configure optimizer with parameter grouping"""
    lr_1x = set()
    lr_2x = set()
    lr_decay = set()    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'attn.w_lora.lora.2.bias' in n:
            lr_2x.add(n)
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n) and ("lora" not in n):
            lr_decay.add(n)
        else:
            lr_1x.add(n)

    lr_1x = sorted(list(lr_1x))
    lr_2x = sorted(list(lr_2x))
    lr_decay = sorted(list(lr_decay))
    param_dict = {n: p for n, p in model.named_parameters()}
    
    optim_groups = [
        {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0, "name": "lr_1x"},
        {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0, "name": "lr_2x"}
    ]
    if args.weight_decay > 0:
        optim_groups.append({
            "params": [param_dict[n] for n in lr_decay],
            "weight_decay": args.weight_decay,
            "my_lr_scale": 1.0,
            "name": "lr_decay"
        })
    
    # 使用标准的PyTorch AdamW优化器，而不是DeepSpeed优化器
    from torch.optim import AdamW
    optimizer = AdamW(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-18, weight_decay=args.weight_decay)
  
    return optimizer

def update_learning_rate(optimizer, current_step, total_steps, warmup_steps, learning_rate, learning_rate_final, args, is_main_process):
    """更新优化器中每个参数组的学习率"""
    # 计算基础学习率
    if current_step < warmup_steps:
        base_lr = learning_rate * (0.01 + 0.99 * current_step / warmup_steps)
    else:
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = max(0, min(1, progress))
        lr_final_factor = learning_rate_final / learning_rate
        base_lr = learning_rate * ((0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress))
    
    # 更新每个参数组的学习率
    for param_group in optimizer.param_groups:
        if param_group.get('weight_decay', 0) > 0:
            param_group['weight_decay'] = args.weight_decay
        lr_scale = param_group.get('my_lr_scale', 1.0)
        param_group['lr'] = base_lr * lr_scale
        if is_main_process and current_step % 100 == 0:
            logger.info(f'param_group: {param_group["name"]} lr: {param_group["lr"]} weight_decay: {param_group["weight_decay"]}')

def log_metrics(optimizer, loss, avg_loss, epoch, total_steps, kts, all_tokens, current_lr):
    """记录训练指标到 wandb"""
    # 记录基本训练指标
    wandb.log({
        "loss": loss.item(),
        "avg_loss": avg_loss,
        "epoch": epoch,
        "step": total_steps,
        "KT/s": kts,
        "Gtokens": all_tokens/1e9,
        "learning_rate": current_lr
    })
    
    # 记录每个参数组的学习率和权重衰减
    for param_group in optimizer.param_groups:
        # 计算参数组的统计信息
        params = param_group['params']
        total_params = sum(p.numel() for p in params)
        
        # 记录到 wandb
        wandb.log({
            f"lr_group_{param_group.get('name', 'default')}": param_group['lr'],
            f"wd_group_{param_group.get('name', 'default')}": param_group.get('weight_decay', 0),
            f"params_count_{param_group.get('name', 'default')}": total_params
        })

def save_checkpoint(model, output_dir, epoch, step, logger):
    """Save model checkpoint using standard PyTorch"""
    if os.path.exists(output_dir):
        checkpoints = os.listdir(output_dir)
        # only list the directories   
        checkpoints = [f for f in checkpoints if os.path.isdir(os.path.join(output_dir, f))]
        # sort by creation time  
        checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
        if len(checkpoints) > 2:
            print(f'deleting older checkpoints {checkpoints[0]}')
            import shutil
            shutil.rmtree(os.path.join(output_dir, checkpoints[0]))    
    output_dir = f"{output_dir}/epoch_{epoch}_step_{step}"
    print(f'saving checkpoint to {output_dir}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 使用标准PyTorch保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'step': step,
    }, f"{output_dir}/model.pt")

class JSONLDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"The file {jsonl_path} does not exist.")
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding JSON line: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_model_config(config_path):
    """加载模型配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_model_args(config):
    """从配置文件创建模型参数"""
    class ModelArgs:
        def __init__(self, config):
            # 设置默认值
            defaults = {
                'vocab_size': 8193,
                'n_layer': 29,
                'n_embd': 512,
                'ctx_len': 4096,
                'head_size_a': 64,
                'head_size_divisor': 8,
                'dim_att': 512,
                'dim_ffn': 1792,
                'dropout': 0.0,
                'weight_decay': 0.01,
                'grad_cp': 1,
                'my_testing': 'x070',
                'my_pile_stage': 0,
                'my_pile_edecay': 0,
                'my_exit_tokens': 0,
                'magic_prime': 0,
                'head_qk': 0,
                'pre_ffn': 0,
                'tiny_att_dim': 0,
                'tiny_att_layer': -1,
                'my_pos_emb': 0,
                'my_qa_mask': 0,
                'my_random_steps': 0,
                'my_exit': 99999999,
                'lr_init': 1e-5,
                'lr_final': 1e-6,
                'warmup_steps': 200,
                'beta1': 0.9,
                'beta2': 0.95,
                'adam_eps': 1e-18
            }
            
            # 从配置文件加载参数，如果没有则使用默认值
            for key, default_value in defaults.items():
                setattr(self, key, config.get(key, default_value))
            
            # 确保必要的参数存在
            if not hasattr(self, 'vocab_size'):
                raise ValueError("vocab_size is required in config")
            if not hasattr(self, 'n_layer'):
                raise ValueError("n_layer is required in config")
            if not hasattr(self, 'n_embd'):
                raise ValueError("n_embd is required in config")
    
    return ModelArgs(config)

def validate_config(config):
    """验证配置文件的合理性"""
    required_keys = ['vocab_size', 'n_layer', 'n_embd', 'ctx_len', 'my_testing']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # 验证数值合理性
    if config['n_embd'] % 32 != 0:
        raise ValueError("n_embd must be divisible by 32")
    if config['dim_att'] % 32 != 0:
        raise ValueError("dim_att must be divisible by 32")
    if config['dim_ffn'] % 32 != 0:
        raise ValueError("dim_ffn must be divisible by 32")
    
    # 验证模型架构
    if config.get('my_testing') != 'x070':
        logger.warning(f"Expected my_testing='x070', got '{config.get('my_testing')}'")
    
    logger.info("Configuration validation passed!")

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    is_main_process = local_rank == 0
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # 移除DeepSpeed初始化，使用普通PyTorch分布式
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    # 初始化分布式训练（不使用DeepSpeed）
    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
        logger.info(f"Initialized distributed training: rank={local_rank}, world_size={world_size}")

    if is_main_process:
        logging.basicConfig(level=logging.INFO)
        # wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    logger.info(f"Starting training with arguments: {args}")

    # 加载MIDI分词器
    try:
        logger.info(f"Loading MIDI tokenizer...")
        midi_tokenizer = REMI(params=args.midi_tokenizer_config_path) 
        midi_tokenizer_vocab = args.midi_tokenizer_vocab_path
        vocab_size = len(midi_tokenizer.vocab)
        logger.info(f"Tokenizer vocab size: {vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    # 加载模型配置并创建模型
    try:
        logger.info(f"Loading model configuration from: {args.model_config}")
        config = load_model_config(args.model_config)
        
        # 验证配置
        validate_config(config)
        
        # 更新词汇表大小（使用分词器的实际大小）
        config['vocab_size'] = vocab_size
        logger.info(f"Updated vocab_size to match tokenizer: {vocab_size}")
        
        # 创建模型参数
        model_args = create_model_args(config)
        
        # 打印模型配置信息
        logger.info(f"Model configuration:")
        logger.info(f"  - vocab_size: {model_args.vocab_size}")
        logger.info(f"  - n_layer: {model_args.n_layer}")
        logger.info(f"  - n_embd: {model_args.n_embd}")
        logger.info(f"  - ctx_len: {model_args.ctx_len}")
        logger.info(f"  - dim_att: {model_args.dim_att}")
        logger.info(f"  - dim_ffn: {model_args.dim_ffn}")
        logger.info(f"  - my_testing: {model_args.my_testing}")
        
        # 创建模型
        model = RWKV(model_args)
        model = model.to(device)
        
        logger.info(f"Model created successfully!")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")

    # Prepare dataset
    try:
        logger.info(f"Loading dataset from JSONL file: {args.jsonl_data_path}")
        dataset = JSONLDataset(args.jsonl_data_path)
        
        # 限制数据集大小进行测试
        if len(dataset) > 20:  # 大幅减少数据集大小
            logger.info(f"Limiting dataset to 20 samples for testing")
            dataset.data = dataset.data[:20]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # 使用普通PyTorch的DistributedSampler
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    else:
        sampler = None
    
    # Create data collator
    data_collator = padded_collate_fn
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.per_device_train_batch_size, 
        sampler=sampler,
        num_workers=0,
        pin_memory=True, 
        collate_fn=data_collator
    )

    # Configure optimizer
    logger.info("Configuring optimizer...")
    optimizer = configure_optimizer(model, args)
    
    # 移除所有DeepSpeed相关代码
    logger.info("Using standard PyTorch training ")

    # Calculate total training steps
    total_steps = len(dataloader) * args.num_epochs
    
    # Create output directory if it doesn't exist
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info("*** Starting Training ***")
    
    # Test a single batch to ensure everything works
    logger.info("Testing data pipeline with a single batch...")
    try:
        raw_test_batch = next(iter(dataloader))
        if raw_test_batch:
            test_batch = process_batch(
                raw_test_batch,
                vocab_size,
                device,
                args
            )
            if test_batch:
                logger.info(f"Test batch shapes - input_ids: {test_batch['input_ids'].shape}, labels: {test_batch['labels'].shape}")
                logger.info("Data pipeline test successful!")
            else:
                logger.warning("Test batch is empty after processing, please check your data!")
        else:
            logger.warning("Test batch is empty, please check your data!")
    except Exception as e:
        logger.error(f"Data pipeline test failed: {e}")
        return
    
    # Training loop
    global_step = 0
    total_loss = 0.0
    all_tokens = 0
    
    try:
        for epoch in range(args.num_epochs):
            # 每个epoch开始时清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            model.train()
            if sampler:
                sampler.set_epoch(epoch)
            
            if is_main_process:
                update_time = time.time()
                tokens_since_last_log = 0
                logger.info(f"Epoch {epoch} starts training")
                try:
                    from tqdm import tqdm
                    pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}", disable=False)
                except Exception as tqdm_error:
                    logger.warning(f"Failed to initialize tqdm: {tqdm_error}")
                    pbar = None
            
            for step, raw_batch in enumerate(dataloader):
                if not raw_batch: 
                    logger.warning("Empty batch received, skipping...")
                    continue

                # Process the batch using original complex function
                try:
                    batch = process_batch(
                        raw_batch,
                        vocab_size,
                        device,
                        args
                    )
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                    continue

                if not batch: 
                    logger.warning("Empty batch after processing, skipping...")
                    continue

                # Update learning rate
                update_learning_rate(
                    optimizer,
                    global_step,
                    total_steps,
                    args.warmup_steps,
                    args.learning_rate,
                    args.learning_rate_final,
                    args,
                    is_main_process
                )

                # 初始化变量，避免作用域问题
                input_ids = None
                labels = None
                loss = None
                
                try:
                    # 确保张量在正确的设备上
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    labels = batch['labels'].to(device, non_blocking=True)

                    # 检查输入数据的有效性
                    if torch.any(torch.isnan(input_ids)) or torch.any(torch.isnan(labels)):
                        logger.warning("NaN detected in input data, skipping batch")
                        continue
                    
                    if torch.any(input_ids < 0) or torch.any(input_ids >= vocab_size):
                        logger.warning(f"Invalid token indices detected: min={input_ids.min()}, max={input_ids.max()}")
                        continue

                    # 在训练循环中
                    with torch.amp.autocast('cuda', enabled=False):
                        # 直接调用模型，不传递注意力掩码
                        outputs = model(input_ids)
                        
                        logits = outputs.view(-1, outputs.size(-1))
                        labels_flat = labels.view(-1)
                        
                        # 只计算非填充位置的损失
                        valid_labels = labels_flat != -100
                        if valid_labels.any():
                            loss = torch.nn.functional.cross_entropy(
                                logits[valid_labels], 
                                labels_flat[valid_labels]
                            )
                        else:
                            logger.warning("No valid labels in batch, skipping...")
                            continue

                    # 检查损失的有效性
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid loss detected: {loss}, skipping batch")
                        continue

                    # 使用普通PyTorch反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 更安全的梯度裁剪方式，避免列表推导式导致的引用计数问题
                    try:
                        # 使用更安全的方式获取有梯度的参数
                        parameters_with_grad = []
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                # 确保梯度内存连续
                                if not param.grad.is_contiguous():
                                    param.grad = param.grad.contiguous()
                                parameters_with_grad.append(param)
                        
                        # 执行梯度裁剪
                        if parameters_with_grad:
                            torch.nn.utils.clip_grad_norm_(parameters_with_grad, max_norm=1.0)
                    except Exception as grad_error:
                        logger.warning(f"Gradient clipping failed: {grad_error}, skipping...")
                    
                    optimizer.step()

                    # 安全清理内存
                    if 'outputs' in locals():
                        del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # 每20步强制垃圾回收（减少频率）
                    if global_step % 20 == 0:
                        gc.collect()

                    global_step += 1

                    # Logging and metrics
                    if is_main_process and input_ids is not None and loss is not None:
                        total_loss += loss.item()
                        
                        # Accumulate tokens for KT/s calculation
                        batch_tokens = input_ids.numel()
                        tokens_since_last_log += batch_tokens * world_size
                        all_tokens += batch_tokens * world_size

                        if global_step % args.logging_steps == 0:
                            elapsed_time = time.time() - update_time
                            kts = (tokens_since_last_log / elapsed_time / 1e3) if elapsed_time > 0 else 0.0
                            avg_loss = total_loss / args.logging_steps
                            current_lr = optimizer.param_groups[0]['lr']
                            
                            logger.info(
                                f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item():.4f}, "
                                f"Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}, KT/s: {kts:.2f}"
                            )
                            
                            # Reset for next logging interval
                            total_loss = 0.0
                            tokens_since_last_log = 0
                            update_time = time.time()
                        
                        if pbar is not None:
                            try:
                                pbar.update(1)
                                pbar.set_postfix({
                                    'loss': loss.item(),
                                    'avg_loss': total_loss / (step % args.logging_steps + 1) if args.logging_steps > 0 else total_loss,
                                    'lr': optimizer.param_groups[0]['lr']
                                })
                            except Exception as pbar_error:
                                logger.warning(f"tqdm update failed: {pbar_error}")

                    # Save checkpoint
                    if global_step % args.save_steps == 0 and is_main_process:
                        checkpoint_dir = f"{args.output_dir}/checkpoint_step_{global_step}"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'step': global_step,
                            'loss': loss.item() if loss is not None else 0.0,
                        }, f"{checkpoint_dir}/model.pt")
                        logger.info(f"Saved checkpoint to {checkpoint_dir}")
                        
                except Exception as e:
                    logger.error(f"Error in training step {global_step}: {e}")
                    # 安全清理变量
                    if input_ids is not None:
                        del input_ids
                    if labels is not None:
                        del labels
                    if loss is not None:
                        del loss
                    continue

            if is_main_process and pbar is not None:
                try:
                    pbar.close()
                except Exception as close_error:
                    logger.warning(f"Failed to close tqdm: {close_error}")
                
            # Save checkpoint at the end of each epoch
            if is_main_process:
                epoch_checkpoint_dir = f"{args.output_dir}/epoch_{epoch}"
                os.makedirs(epoch_checkpoint_dir, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': global_step,
                    'loss': loss.item(),
                }, f"{epoch_checkpoint_dir}/model.pt")
                logger.info(f"Saved epoch checkpoint to {epoch_checkpoint_dir}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

    logger.info("--- Training Finished ---")
    # 保存最终模型
    if is_main_process:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': args.num_epochs,
            'step': global_step,
        }, f"{args.output_dir}/final_model.pt")
        logger.info(f"Saved final model to {args.output_dir}/final_model.pt")
    
    # 安全清理
    try:
        if is_main_process and wandb.run is not None:
            wandb.finish()
    except Exception as e:
        logger.warning(f"Failed to finish wandb: {e}")
    
    # 安全清理模型和优化器
    try:
        del model
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.info("Model and optimizer safely cleaned up")
    except Exception as e:
        logger.warning(f"Failed to cleanup model: {e}")
    
    logger.info("Training script completed successfully")

if __name__ == "__main__":
    main()
