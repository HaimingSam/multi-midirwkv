"""Dataset class to train models for MMM."""

from __future__ import annotations

from random import choice, random, sample, uniform
from typing import TYPE_CHECKING
from copy import deepcopy

import numpy as np
from miditok import TokSequence
from miditok.attribute_controls import BarAttributeControl
from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.data_augmentation.data_augmentation import (
    _filter_offset_tuples_to_score,
    augment_score,
)
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import get_bars_ticks
from symusic import Score
from torch import LongTensor, isin
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from datasets import Dataset
    from miditok import MMM


def concat_tokseq(sequences: list[TokSequence]) -> TokSequence:
    """
    Concatenate a sequence of :class:`miditok.TokSequence`.

    :param sequences: :class:`miditok.TokSequence`s to concatenate.
    :return: the concatenated ``sequences``.
    """
    tokseq = sequences.pop(0)
    for seq in sequences:
        tokseq += seq
    return tokseq


class DataCollatorNoneFilter:
    def __init__(self, pad_token_id=0, max_length=2048):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.collator = DataCollator(pad_token_id)
    
    def __call__(self, batch):
        collated_batch = self.collator(batch)
        
        # Get the current sequence length
        input_ids = collated_batch["input_ids"]
        labels = collated_batch["labels"]
        batch_size, seq_len = input_ids.size()
        labels_bsz, labels_seq_len = labels.size()

        # Pad to the fixed length if needed
        if seq_len < self.max_length:
            padding_len = self.max_length - seq_len           
            padding = torch.full((batch_size, padding_len), self.pad_token_id, 
                                dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        elif seq_len > self.max_length:
            # Truncate if sequence is longer than max_length
            input_ids = input_ids[:, :self.max_length]

        if labels_seq_len < self.max_length:
            padding_len = self.max_length - labels_seq_len
            label_padding = torch.full((labels_bsz, padding_len), -100, 
                                      dtype=labels.dtype, device=labels.device)
            labels = torch.cat([labels, label_padding], dim=1)
        elif labels_seq_len > self.max_length:
            # Truncate if sequence is longer than max_length
            labels = labels[:, :self.max_length]

        # Return just the input_ids and labels as a tuple
        return input_ids, labels


class AutoregressiveMIDIDataset(DatasetMIDI):
    """
    自回归多轨MIDI数据集
    用于训练生成多轨音乐而非填充
    """
    
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: MMM,
        max_seq_len: int,
        ratio_random_tracks_range: tuple[float, float] = (0.4, 1.0),
        data_augmentation_offsets: tuple[int, int, int] = (6, 2, 0),
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        sample_key_name: str = "input_ids",
        labels_key_name: str = "labels",
    ) -> None:
        self._dataset = dataset
        self.ratio_random_tracks_range = ratio_random_tracks_range
        pitch_offsets = data_augmentation_offsets[0]
        self.pitch_offsets = list(range(-pitch_offsets, pitch_offsets + 1))
        velocity_offsets = data_augmentation_offsets[1]
        self.velocity_offsets = list(range(-velocity_offsets, velocity_offsets + 1))
        duration_offsets = data_augmentation_offsets[2]
        self.duration_offsets = list(range(-duration_offsets, duration_offsets + 1))
        
        # 特殊token ID
        self._track_start_token_id = tokenizer.vocab["Track_Start"]
        self._track_end_token_id = tokenizer.vocab["Track_End"]
        
        # Token ids that should be masked from the "labels" entry
        self._token_ids_no_loss = LongTensor([
            tokenizer.vocab["Track_Start"],
            tokenizer.vocab["Track_End"],
        ])

        max_seq_len -= sum([1 for t in [bos_token_id, eos_token_id] if t is not None])
        super().__init__(
            [],
            tokenizer,
            max_seq_len,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pre_tokenize=False,
            func_to_get_labels=None,
            sample_key_name=sample_key_name,
            labels_key_name=labels_key_name,
        )
    
    def __getitem__(self, idx: int) -> dict[str, LongTensor]:
        """
        获取训练样本
        直接从JSONL中读取token序列（已预处理为固定长度）
        """
        try:
            # 直接从JSONL中获取token序列
            sample = self._dataset[idx]
            token_strings = sample["midi_tokens"]
            
            # 将token字符串转换为ID
            tokens = []
            for token_str in token_strings:
                if token_str in self.tokenizer.vocab:
                    tokens.append(self.tokenizer.vocab[token_str])
                else:
                    # 如果token不在词汇表中，跳过
                    print(f"Warning: token '{token_str}' not in vocabulary")
                    continue
            
            if len(tokens) == 0:
                return {self.sample_key_name: None, self.labels_key_name: None}
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return {self.sample_key_name: None, self.labels_key_name: None}

        # 直接转换为tensor（假设长度已经预处理为max_seq_len）
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # 创建input_ids和labels（自回归：输入和标签相同）
        input_ids = tokens_tensor[:-1]  # 除了最后一个token
        labels = tokens_tensor[1:]      # 除了第一个token
        
        item = {self.sample_key_name: input_ids, self.labels_key_name: labels}
        
        # Set ids of elements to discard to -100
        idx_tokens_to_discard = isin(
            item[self.labels_key_name], self._token_ids_no_loss
        )
        item[self.labels_key_name][idx_tokens_to_discard] = -100
        
        return item
    
    def _tokenize_score(self, score: Score) -> list[int] | None:
        """
        自回归模式的tokenization
        保持多轨结构，不进行填充任务
        """
        # 删除未使用的元素
        score.markers = []
        score.key_signatures = []
        for track in score.tracks:
            track.controls = []
            track.lyrics = []
            if not self.tokenizer.config.use_sustain_pedals:
                track.pedals = []
            if not self.tokenizer.config.use_pitch_bends:
                track.pitch_bends = []

        # 随机选择轨道数量（数据增强）
        num_tracks_to_keep = max(
            1, round(len(score.tracks) * uniform(*self.ratio_random_tracks_range))
        )
        bars_ticks = np.array(get_bars_ticks(score))
        tracks_idx_ok = [
            idx
            for idx in range(len(score.tracks))
            if len(score.tracks[idx].notes) > 0
               and len(bars_ticks) > 1
               and score.tracks[idx].notes[-1].time > bars_ticks[1]
        ]

        if len(tracks_idx_ok) == 0:
            return None

        # 随机选择轨道
        score.tracks = [
            score.tracks[idx]
            for idx in sample(
                tracks_idx_ok, k=min(num_tracks_to_keep, len(tracks_idx_ok))
            )
        ]

        # 移除时间签名和速度变化
        max_note_time = 0
        for track in score.tracks:
            if len(track.notes) > 0:
                max_note_time = max(max_note_time, track.notes[-1].time)
        for ti in reversed(range(len(score.time_signatures))):
            if score.time_signatures[ti].time > max_note_time:
                del score.time_signatures[ti]
            else:
                break
        for ti in reversed(range(len(score.tempos))):
            if score.tempos[ti].time > max_note_time:
                del score.tempos[ti]
            else:
                break

        # 数据增强和预处理
        score = self.augment_and_preprocess_score(score)
        
        if len(score.tracks) == 0:
            return None

        # 编码为多轨token序列（保持结构）
        tokens = self.tokenizer.encode(score, concatenate_track_sequences=False)
        
        if tokens is None or len(tokens) == 0:
            return None

        # 合并为单一序列用于自回归训练
        # 格式: [BOS] [Track_Start] [Program_X] [轨道内容] [Track_End] [Track_Start] [Program_Y] [轨道内容] [Track_End] ... [EOS]
        combined_tokens = []
        
        if self.bos_token_id is not None:
            combined_tokens.append(self.bos_token_id)
        
        for track_idx, track_tokens in enumerate(tokens):
            # 添加轨道开始标记
            combined_tokens.append(self.tokenizer.vocab["Track_Start"])
            
            # 添加程序号
            program = score.tracks[track_idx].program
            combined_tokens.append(self.tokenizer.vocab[f"Program_{program}"])
            
            # 添加轨道内容
            combined_tokens.extend(track_tokens.ids)
            
            # 添加轨道结束标记
            combined_tokens.append(self.tokenizer.vocab["Track_End"])
        
        if self.eos_token_id is not None:
            combined_tokens.append(self.eos_token_id)
        
        return combined_tokens

    def augment_and_preprocess_score(self, score: Score) -> Score:
        """
        Augment a ``symusic.Score`` and preprocess it with the tokenizer.

        :param score: score to augment and preprocess.
        :return: the augmented and preprocessed track.
        """
        pitch_offsets = _filter_offset_tuples_to_score(
            self.pitch_offsets.copy(),
            score,
            restrict_on_program_tessitura=True,
        )
        if len(pitch_offsets) > 0:
            score = augment_score(
                score,
                choice(pitch_offsets),
                choice(self.velocity_offsets),
                choice(self.duration_offsets),
            )
        return self.tokenizer.preprocess_score(score)

    def __len__(self) -> int:
        """
        Return the size of the dataset.

        :return: number of elements in the dataset.
        """
        return len(self._dataset)
