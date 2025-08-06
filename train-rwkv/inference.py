import torch
import json
import os
import argparse
from typing import List, Optional
import numpy as np

# 导入模型
from src.model import RWKV
from miditok import REMI

class ModelArgs:
    def __init__(self, vocab_size=16000):
        self.vocab_size = vocab_size
        self.n_layer = 29
        self.n_embd = 512
        self.ctx_len = 4096
        self.head_size_a = 64
        self.head_size_divisor = 8
        self.dim_att = 512
        self.dim_ffn = 1792
        self.dropout = 0.0
        self.weight_decay = 0.01
        self.grad_cp = 0
        self.my_testing = "x070"
        self.my_pile_stage = 0
        self.my_pile_edecay = 0
        self.my_exit_tokens = 0
        self.magic_prime = 0
        self.head_qk = 0
        self.pre_ffn = 0
        self.tiny_att_dim = 0
        self.tiny_att_layer = -1
        self.my_pos_emb = 0
        self.my_qa_mask = 0
        self.my_random_steps = 0
        self.my_exit = 99999999
        self.lr_init = 1e-5
        self.lr_final = 1e-6
        self.warmup_steps = 200
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.adam_eps = 1e-18

class MusicGenerator:
    def __init__(self, model_path: str, tokenizer_config_path: str, device: str = "cuda"):
        """
        初始化音乐生成器
        
        Args:
            model_path: 训练好的模型文件路径
            tokenizer_config_path: 分词器配置文件路径
            device: 设备类型 ("cuda" 或 "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载分词器
        self.tokenizer = REMI(params=tokenizer_config_path)
        self.vocab_size = len(self.tokenizer.vocab)
        print(f"Tokenizer vocab size: {self.vocab_size}")
        
        # 创建模型
        model_args = ModelArgs(vocab_size=self.vocab_size)
        self.model = RWKV(model_args)
        
        # 加载训练好的权重
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        
        # 获取特殊token的ID
        self.bos_token_id = self.tokenizer.vocab.get("BOS_None", 1)
        self.eos_token_id = self.tokenizer.vocab.get("EOS_None", 2)
        self.pad_token_id = self.tokenizer.vocab.get("PAD_None", 0)
        
    def generate_music(self, 
                      prompt_tokens: Optional[List[int]] = None,
                      max_length: int = 1000,
                      temperature: float = 1.0,
                      top_k: int = 50,
                      top_p: float = 0.9,
                      do_sample: bool = True) -> List[int]:
        """
        生成音乐序列
        
        Args:
            prompt_tokens: 起始tokens，如果为None则使用BOS token
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: nucleus采样参数
            do_sample: 是否使用采样
            
        Returns:
            生成的token序列
        """
        self.model.eval()
        
        with torch.no_grad():
            # 准备输入
            if prompt_tokens is None:
                input_ids = torch.tensor([[self.bos_token_id]], dtype=torch.long, device=self.device)
            else:
                input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            
            generated_tokens = input_ids[0].tolist()
            
            print(f"Starting generation with {len(generated_tokens)} prompt tokens...")
            
            for step in range(max_length):
                # 确保输入长度不超过上下文长度
                if input_ids.shape[1] > self.model.args.ctx_len:
                    input_ids = input_ids[:, -self.model.args.ctx_len:]
                
                # 获取模型输出
                outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :]
                
                # 应用温度
                next_token_logits = next_token_logits / temperature
                
                if do_sample:
                    # Top-k采样
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    # Nucleus采样
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # 采样下一个token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪婪解码
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 添加到序列中
                input_ids = torch.cat([input_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
                # 检查是否生成了结束token
                if next_token.item() == self.eos_token_id:
                    print(f"Generated EOS token at step {step}")
                    break
                
                # 打印进度
                if step % 100 == 0:
                    print(f"Generated {step + 1} tokens...")
            
            print(f"Generation completed! Total tokens: {len(generated_tokens)}")
            return generated_tokens
    
    def tokens_to_midi(self, tokens: List[int], output_path: str):
        """
        将生成的tokens转换为MIDI文件
        
        Args:
            tokens: 生成的token序列
            output_path: 输出MIDI文件路径
        """
        try:
            # 使用分词器将tokens转换为MIDI
            midi = self.tokenizer.tokens_to_midi([tokens])
            midi.dump(output_path)
            print(f"MIDI file saved to: {output_path}")
        except Exception as e:
            print(f"Error converting tokens to MIDI: {e}")
    
    def generate_and_save(self, 
                         output_path: str,
                         prompt_tokens: Optional[List[int]] = None,
                         max_length: int = 1000,
                         temperature: float = 1.0,
                         top_k: int = 50,
                         top_p: float = 0.9,
                         do_sample: bool = True):
        """
        生成音乐并直接保存为MIDI文件
        
        Args:
            output_path: 输出MIDI文件路径
            prompt_tokens: 起始tokens
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: nucleus采样参数
            do_sample: 是否使用采样
        """
        # 生成tokens
        generated_tokens = self.generate_music(
            prompt_tokens=prompt_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample
        )
        
        # 保存为MIDI文件
        self.tokens_to_midi(generated_tokens, output_path)

def main():
    parser = argparse.ArgumentParser(description="RWKV-7 Music Generation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model file (.pt)")
    parser.add_argument("--tokenizer_config", type=str, required=True,
                       help="Path to the tokenizer config file")
    parser.add_argument("--output_path", type=str, default="generated_music.mid",
                       help="Output MIDI file path")
    parser.add_argument("--prompt_tokens", type=str, default=None,
                       help="Comma-separated prompt tokens (e.g., '1,2,3')")
    parser.add_argument("--max_length", type=int, default=1000,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus sampling parameter")
    parser.add_argument("--do_sample", action="store_true", default=True,
                       help="Use sampling instead of greedy decoding")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # 解析prompt tokens
    prompt_tokens = None
    if args.prompt_tokens:
        prompt_tokens = [int(x.strip()) for x in args.prompt_tokens.split(",")]
    
    # 创建生成器
    generator = MusicGenerator(
        model_path=args.model_path,
        tokenizer_config_path=args.tokenizer_config,
        device=args.device
    )
    
    # 生成音乐
    generator.generate_and_save(
        output_path=args.output_path,
        prompt_tokens=prompt_tokens,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample
    )

if __name__ == "__main__":
    main() 
