import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List

from transformers.modeling_outputs import CausalLMOutputWithPast
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Model, RWKV7ForCausalLM, Cache
from rwkvfla.models.rwkv7.configuration_rwkv7 import RWKV7Config

# Generation imports
from transformers.generation import GenerationMixin, LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.generation.streamers import BaseStreamer


class RWKV7MIDIConfig(RWKV7Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # MIDI 词汇表大小
        self.midi_vocab_size = kwargs.get("midi_vocab_size", 413)
        # 轨道数量
        self.num_tracks = 16 #MIDI 文件中独立的乐器轨道数量。每个轨道对应一种乐器或音效
        # 长度归一化损失
        self.length_normalized_loss = kwargs.get("length_normalized_loss", True)
        # 标签平滑权重
        self.lsm_weight = kwargs.get("lsm_weight", 0.0)
        # 丢弃率
        self.drop_ratio = kwargs.get("drop_ratio", 0.0)
        # MIDI 填充标记
        self.midi_pad_token = kwargs.get("midi_pad_token", 0)
        
        self.track_min_tokens = kwargs.get("track_min_tokens", [0] * self.num_tracks)
        self.track_max_tokens = kwargs.get("track_max_tokens", [self.midi_vocab_size] * self.num_tracks)

class CustomMIDIGenerationMixin(GenerationMixin):
    """
    Custom GenerationMixin to provide a bespoke _sample method for the RWKV7MIDILM model.
    """

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateDecoderOnlyOutput, torch.LongTensor]:
        # 初始化参数
        midi_pad_idx = self.config.midi_pad_token
        midi_vocab_size = self.config.midi_vocab_size
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        output_scores = generation_config.output_scores
        return_dict_in_generate = generation_config.return_dict_in_generate

        batch_size, cur_len, tracks = input_ids.shape
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # --- 2. Decode Loop ---
        while True:
            # --- 2a. Prepare model inputs ---
            # 准备模型输入
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # --- 2b. Forward pass ---
            # 前向传播
            outputs = self(**model_inputs, return_dict=True)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # --- 2c. Get logits for all channels ---
            # 获取所有轨道的下一个标记的 logits
            next_token_logits = [logits[:, -1, :].clone().float() for logits in outputs.logits]

            # --- 2d. Constrain logits for each track ---
            # 对每个音轨的 logits 进行约束，确保只采样有效范围内的标记
            for i in range(tracks):
                # 获取当前音轨的最小和最大标记范围
                min_token = self.config.track_min_tokens[i]
                max_token = self.config.track_max_tokens[i]
                # 创建一个掩码，将有效范围外的位置置为 1
                mask = torch.ones_like(next_token_logits[i])
                mask[:, min_token : max_token] = 0
                # 将有效范围外的 logits 置为负无穷，这样在采样时就不会被选中
                next_token_logits[i].masked_fill_(mask.bool(), -float("inf"))

            # 对所有音轨的 logits 应用标准的 logits 处理器，如温度调整等
            # 应用标准的 logits 处理器
            next_token_scores = [logits_processor(input_ids[..., i], logits) for i, logits in enumerate(next_token_logits)]

            # --- 2e. Sample next tokens for all channels ---
            # 为所有轨道采样下一个标记
            next_tokens_list = []
            for i, channel_score in enumerate(next_token_scores):
                probs = F.softmax(channel_score, dim=-1)
                channel_ntk = torch.multinomial(probs, num_samples=1).squeeze(1)
                next_tokens_list.append(channel_ntk)
            next_tokens = torch.stack(next_tokens_list, dim=-1)
            

            # --- 2g. Update state for next iteration ---
            # 确保已完成的序列继续输出填充/结束标记
            if eos_token_id is not None:
                pad_or_eos_token = eos_token_id[0]
            else:
                pad_or_eos_token = 0  # 默认填充标记
            next_tokens[:, 0] = next_tokens[:, 0] * unfinished_sequences + pad_or_eos_token * (1 - unfinished_sequences)
            next_tokens[:, 1:] = next_tokens[:, 1:] * unfinished_sequences.unsqueeze(-1) + midi_pad_idx * (
                    1 - unfinished_sequences.unsqueeze(-1))

            # 更新input_ids
            input_ids = torch.cat([input_ids, next_tokens[:, None, :]], dim=1)
            if streamer is not None:
                streamer.put(next_tokens[:, 0].cpu())

            # 检查停止条件
            stopping_criteria_met = stopping_criteria(input_ids[..., 0], None)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria_met

            if unfinished_sequences.max() == 0:
                break
        
        # --- 3. Finalize and Return ---
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=input_ids)
        else:
            return input_ids


class RWKV7MIDILM(RWKV7ForCausalLM, CustomMIDIGenerationMixin):
    config_class = RWKV7MIDIConfig

    def __init__(self, config: RWKV7MIDIConfig):
        super().__init__(config)

        self.model = RWKV7Model(config)

        self.embs = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.criterions = nn.ModuleList()

        # 为每个轨道初始化嵌入层、头部层和损失函数
        for _ in range(config.num_tracks):
            self.embs.append(nn.Embedding(config.midi_vocab_size, config.hidden_size,
                                          padding_idx=config.midi_pad_token))
            self.heads.append(nn.Linear(config.hidden_size, config.midi_vocab_size))
            self.criterions.append(nn.CrossEntropyLoss(label_smoothing=config.lsm_weight, ignore_index=-100))

        self.dropout = nn.Dropout(config.drop_ratio) if config.drop_ratio > 0 else None

        self.post_init()

    def zero_embs(self):
        """
        Manually zero out the embedding vectors for the padding indices.
        """
        midi_pad_idx = self.config.midi_pad_token
        for i in range(self.config.num_tracks):
            if self.embs[i].padding_idx is not None:
                self.embs[i].weight.data[midi_pad_idx].zero_()
    
    def ensure_padding_embeddings_zero(self):
        """
        Ensure padding embeddings are always zero during training.
        This should be called periodically during training.
        """
        midi_pad_idx = self.config.midi_pad_token
        for i in range(self.config.num_tracks):
            if self.embs[i].padding_idx is not None:
                # 强制设置为0，并阻止梯度更新
                with torch.no_grad():
                    self.embs[i].weight.data[midi_pad_idx].zero_()
    
    def verify_padding_embeddings(self):
        """
        Verify that padding embeddings are zero and return status.
        """
        midi_pad_idx = self.config.midi_pad_token
        all_zero = True
        for i in range(self.config.num_tracks):
            if self.embs[i].padding_idx is not None:
                if not torch.all(self.embs[i].weight.data[midi_pad_idx] == 0):
                    all_zero = False
                    break
        return all_zero

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # (B, T, num_tracks)
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,  # (B, T, num_tracks)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None and input_ids is not None:
            if input_ids.dim() != 3 or input_ids.shape[2] != self.config.num_tracks:
                raise ValueError(f"input_ids must have shape (B, T, num_tracks), but got {input_ids.shape}")

            # 检查输入数据的有效性
            if torch.any(torch.isnan(input_ids)):
                raise ValueError("input_ids contains NaN values")
            if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.midi_vocab_size):
                raise ValueError(f"input_ids contains invalid token indices: min={input_ids.min()}, max={input_ids.max()}, vocab_size={self.config.midi_vocab_size}")
            
            B, T, num_tracks = input_ids.shape
            embeds_list = [self.embs[i](input_ids[:, :, i]) for i in range(num_tracks)]
            inputs_embeds = torch.stack(embeds_list, dim=0).sum(dim=0)
            
            # 检查合并后的embedding
            if torch.any(torch.isnan(inputs_embeds)):
                raise ValueError("NaN detected in combined embeddings")
            if torch.any(torch.isinf(inputs_embeds)):
                raise ValueError("Inf detected in combined embeddings")
            
            # 确保padding位置的embedding为0，避免梯度爆炸
            # 创建padding mask
            padding_mask = (input_ids == self.config.midi_pad_token).any(dim=2)  # (B, T)
            padding_mask = padding_mask.unsqueeze(-1).expand_as(inputs_embeds)  # (B, T, hidden_size)
            inputs_embeds = inputs_embeds.masked_fill(padding_mask, 0.0)

        if self.dropout is not None:
            inputs_embeds = self.dropout(inputs_embeds)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        total_loss = None
        all_logits = []

        if labels is not None:
            total_loss = 0
            for i in range(self.config.num_tracks):
                logits = self.heads[i](hidden_states)
                
                # 检查logits的有效性
                if torch.any(torch.isnan(logits)):
                    raise ValueError(f"NaN detected in channel {i} logits")
                if torch.any(torch.isinf(logits)):
                    raise ValueError(f"Inf detected in channel {i} logits")
                
                all_logits.append(logits)
                channel_labels = labels[:, :, i].view(-1)
                
                # 检查labels的有效性
                valid_labels = channel_labels != -100
                if valid_labels.sum() == 0:
                    # 如果所有labels都是-100，跳过这个通道的损失计算
                    continue
                
                loss = self.criterions[i](logits.view(-1, logits.shape[-1]), channel_labels)
                
                # 检查损失的有效性
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Invalid loss detected in channel {i}: {loss}")
                
                total_loss += loss
        else:
            # 在推理时，仍然计算所有轨道的 logits
            for i in range(self.config.num_tracks):
                logits = self.heads[i](hidden_states)
                all_logits.append(logits)

        if not return_dict:
            output = (all_logits,) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=all_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
