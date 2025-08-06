import json
import logging
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def intelligent_split_sequences(input_jsonl_path, output_jsonl_path, max_seq_len=4096, min_seq_len=512):
    """
    智能切分序列，处理长度不足的情况，并确保所有输出序列都有完整的开始/结束标记。
    """
    logging.info(f"开始智能切分序列...")
    logging.info(f"输入文件: {input_jsonl_path}")
    logging.info(f"输出文件: {output_jsonl_path}")
    logging.info(f"最大序列长度: {max_seq_len}")
    logging.info(f"最小序列长度: {min_seq_len}")
    
    # 统计变量
    total_files = 0
    split_files = 0
    skipped_short = 0
    length_stats = []
    
    with open(input_jsonl_path, 'r') as input_file, open(output_jsonl_path, 'w') as output_file:
        
        for line in tqdm(input_file, desc="Processing files"):
            try:
                data = json.loads(line.strip())
                tokens = data['midi_tokens']
                total_files += 1
                
                # 如果序列太短，跳过
                if len(tokens) < min_seq_len:
                    logging.warning(f"序列长度 {len(tokens)} 小于最小长度 {min_seq_len}，跳过")
                    skipped_short += 1
                    continue
                
                # 所有序列（包括原始长度合适的和经过切分的）都经过 ensure_complete_structure 处理
                # 然后再进行长度检查和写入
                
                # 对原始序列进行智能切分，这会返回一个或多个片段
                # 即使原始序列长度合适，也会通过 intelligent_split 生成一个片段
                splits = intelligent_split(tokens, max_seq_len, min_seq_len)
                
                # 遍历所有生成的片段，并写入文件
                for i, split in enumerate(splits):
                    # ensure_complete_structure 已经在 intelligent_split 内部调用了
                    # 确保片段达到最小长度
                    if len(split) >= min_seq_len:
                        output_file.write(json.dumps({"midi_tokens": split}) + '\n')
                        split_files += 1
                        length_stats.append(len(split))
                    else:
                        logging.warning(f"智能切分后片段 {i+1} 长度 {len(split)} 小于最小长度，跳过")
                
                # 检查原始序列是否被切分过
                if len(splits) > 1:
                    logging.info(f"将长度为 {len(tokens)} 的序列切分为 {len(splits)} 个片段")
                
            except Exception as e:
                logging.error(f"Error processing line: {e}")
                continue
    
    # 打印统计信息
    if length_stats:
        stats = {
            'total_original': total_files,
            'total_after_split': split_files,
            'skipped_short': skipped_short,
            'avg_length': np.mean(length_stats),
            'median_length': np.median(length_stats),
            'min_length': np.min(length_stats),
            'max_length': np.max(length_stats),
            'length_distribution': {
                '512-1K': sum(1 for x in length_stats if 512 <= x <= 1000),
                '1K-2K': sum(1 for x in length_stats if 1000 < x <= 2000),
                '2K-3K': sum(1 for x in length_stats if 2000 < x <= 3000),
                '3K-4K': sum(1 for x in length_stats if 3000 < x <= 4000),
                '4K+': sum(1 for x in length_stats if x > 4000)
            }
        }
        
        logging.info(f"\n=== 切分统计 ===")
        logging.info(f"原始文件数: {stats['total_original']}")
        logging.info(f"切分后文件数: {stats['total_after_split']}")
        logging.info(f"跳过短序列数: {stats['skipped_short']}")
        logging.info(f"平均长度: {stats['avg_length']:.1f}")
        logging.info(f"中位数长度: {stats['median_length']:.1f}")
        logging.info(f"长度范围: {stats['min_length']} - {stats['max_length']}")
        logging.info(f"\n长度分布:")
        for range_name, count in stats['length_distribution'].items():
            percentage = count / stats['total_after_split'] * 100
            logging.info(f"  {range_name}: {count} ({percentage:.1f}%)")
        
        # 检查是否有超过4096的序列
        over_limit = [x for x in length_stats if x > max_seq_len]
        if over_limit:
            logging.warning(f"发现 {len(over_limit)} 个超过 {max_seq_len} 的序列:")
            logging.warning(f"  最大长度: {max(over_limit)}")
            logging.warning(f"  平均超长: {np.mean(over_limit):.1f}")

def ensure_complete_structure(tokens, max_len=4096):
    """
    确保片段有完整的音乐结构，添加BOS_None、Bar_None和EOS_None
    注意：tokens是数字ID，不是字符串
    结尾必须是EOS_None，不能是Bar_None
    """
    # 检查是否已经有BOS_None (ID=1)
    has_bos = tokens and tokens[0] == 1
    has_bar = tokens and tokens[0] == 4  # Bar_None的ID是4
    
    # 添加开始标记
    if not has_bos:
        # 如果没有BOS_None，添加BOS_None和Bar_None
        if not has_bar:
            # 既没有BOS_None也没有Bar_None
            tokens = [1, 4] + tokens  # BOS_None=1, Bar_None=4
        else:
            # 有Bar_None但没有BOS_None
            tokens = [1] + tokens  # 只添加BOS_None
    
    # 处理结束标记：确保结尾是EOS_None而不是Bar_None
    if not tokens:
        tokens.append(2)  # 空序列，直接添加EOS_None
    elif tokens[-1] == 2:  # 已经是EOS_None，不需要修改
        pass
    elif tokens[-1] == 4:  # 如果结尾是Bar_None
        # 移除Bar_None，添加EOS_None
        tokens = tokens[:-1] + [2]
        logging.info(f"将结尾的Bar_None替换为EOS_None")
    else:  # 其他情况，直接添加EOS_None
        tokens.append(2)
    
    return tokens

def intelligent_split(tokens, max_len, min_len):
    """
    改进的智能切分策略，确保每个片段都有完整的音乐结构
    """
    splits = []
    current_split = []
    
    for i, token in enumerate(tokens):
        current_split.append(token)
        
        # 检查是否达到最大长度
        if len(current_split) >= max_len:
            # 寻找最佳分割点
            split_point = find_best_split_point(current_split)
            
            if split_point > 0:
                # 在合适位置分割
                split_tokens = current_split[:split_point]
                if len(split_tokens) >= min_len:
                    # 确保片段有完整的开始结构，预留空间给BOS_None和Bar_None
                    split_tokens = ensure_complete_structure(split_tokens, max_len)
                    splits.append(split_tokens)
                current_split = current_split[split_point:]
            else:
                # 强制分割
                if len(current_split) >= min_len:
                    current_split = ensure_complete_structure(current_split, max_len)
                    splits.append(current_split)
                current_split = []
    
    # 处理剩余部分
    if current_split and len(current_split) >= min_len:
        current_split = ensure_complete_structure(current_split, max_len)
        splits.append(current_split)
    
    return splits

def find_best_split_point(tokens):
    """
    寻找最佳分割点，优先级：
    1. 小节结束 (Bar_None, ID=4)
    2. 段落结束 (EOS_None, ID=2)  
    3. 持续时间token (Duration_*, ID范围)
    4. 音符结束
    """
    # 从后往前搜索，优先选择音乐结构边界
    for i in range(len(tokens) - 1, max(0, len(tokens) - 500), -1):
        token_id = tokens[i]
        
        # 小节结束
        if token_id == 4:  # Bar_None
            return i + 1
        
        # 段落结束
        if token_id == 2:  # EOS_None
            return i + 1
    
    # 寻找持续时间token (ID范围: 126-189)
    for i in range(len(tokens) - 1, max(0, len(tokens) - 100), -1):
        token_id = tokens[i]
        if 126 <= token_id <= 189:  # Duration_* 的ID范围
            return i + 1
    
    # 寻找音符结束（Pitch后面跟着Velocity）
    for i in range(len(tokens) - 2, max(0, len(tokens) - 50), -1):
        if (5 <= tokens[i] <= 93 and  # Pitch_* 的ID范围: 5-93
            94 <= tokens[i+1] <= 125):  # Velocity_* 的ID范围: 94-125
            return i + 2
    
    return -1  # 没有找到合适的分割点

def analyze_split_quality(input_jsonl_path, sample_size=10):
    """
    分析切分质量，检查分割点是否合理
    """
    logging.info(f"分析切分质量（样本数: {sample_size}）...")
    
    with open(input_jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
                
            try:
                data = json.loads(line.strip())
                tokens = data['midi_tokens']
                
                # 检查开始和结束标记
                start_markers = []
                end_markers = []
                if tokens:
                    start_markers.append(tokens[0])  # 直接使用数字ID
                    if len(tokens) > 1:
                        start_markers.append(tokens[1])
                    end_markers.append(tokens[-1])  # 结束标记
                
                # 分析token分布
                token_types = {}
                for token_id in tokens:
                    if token_id == 4:  # Bar_None
                        token_types["Bar"] = token_types.get("Bar", 0) + 1
                    elif 190 <= token_id <= 221:  # Position_* 的ID范围
                        token_types["Position"] = token_types.get("Position", 0) + 1
                    elif 298 <= token_id <= 426:  # Program_* 的ID范围
                        token_types["Program"] = token_types.get("Program", 0) + 1
                    elif 5 <= token_id <= 93:  # Pitch_* 的ID范围
                        token_types["Pitch"] = token_types.get("Pitch", 0) + 1
                    elif 126 <= token_id <= 189:  # Duration_* 的ID范围
                        token_types["Duration"] = token_types.get("Duration", 0) + 1
                
                logging.info(f"样本 {i+1}: 长度={len(tokens)}, 开始标记={start_markers}, 结束标记={end_markers}, "
                           f"小节数={token_types.get('Bar', 0)}, 位置数={token_types.get('Position', 0)}")
                
            except Exception as e:
                logging.error(f"分析样本 {i+1} 时出错: {e}")

def validate_split_results(output_jsonl_path, max_seq_len=4096):
    """
    验证切分结果，确保所有序列都有正确的开始和结束标记
    """
    logging.info(f"验证切分结果...")
    
    total_count = 0
    valid_count = 0
    invalid_start_count = 0
    invalid_end_count = 0
    too_long_count = 0
    too_short_count = 0
    length_stats = []
    
    with open(output_jsonl_path, 'r') as f:
        for line in tqdm(f, desc="Validating results"):
            try:
                data = json.loads(line.strip())
                tokens = data['midi_tokens']
                total_count += 1
                length_stats.append(len(tokens))
                
                # 检查长度
                if len(tokens) > max_seq_len:
                    too_long_count += 1
                    logging.warning(f"发现过长序列: {len(tokens)} tokens (超过 {max_seq_len})")
                elif len(tokens) < 512:
                    too_short_count += 1
                    logging.warning(f"发现过短序列: {len(tokens)} tokens")
                else:
                    # 检查开始标记
                    if not tokens or tokens[0] != 1:  # BOS_None
                        invalid_start_count += 1
                        logging.warning(f"序列缺少BOS_None开始标记")
                    elif len(tokens) < 2 or tokens[1] != 4:  # Bar_None
                        invalid_start_count += 1
                        logging.warning(f"序列缺少Bar_None标记")
                    # 检查结束标记
                    elif not tokens or tokens[-1] != 2:  # EOS_None
                        invalid_end_count += 1
                        logging.warning(f"序列缺少EOS_None结束标记")
                    else:
                        valid_count += 1
                    
            except Exception as e:
                logging.error(f"验证时出错: {e}")
    
    logging.info(f"\n=== 验证结果 ===")
    logging.info(f"总序列数: {total_count}")
    logging.info(f"有效序列数: {valid_count}")
    logging.info(f"无效开始标记数: {invalid_start_count}")
    logging.info(f"无效结束标记数: {invalid_end_count}")
    logging.info(f"过长序列数: {too_long_count}")
    logging.info(f"过短序列数: {too_short_count}")
    logging.info(f"有效比例: {valid_count/total_count*100:.2f}%")
    
    if length_stats:
        logging.info(f"长度统计:")
        logging.info(f"  平均长度: {np.mean(length_stats):.1f}")
        logging.info(f"  中位数长度: {np.median(length_stats):.1f}")
        logging.info(f"  最小长度: {np.min(length_stats)}")
        logging.info(f"  最大长度: {np.max(length_stats)}")
        
        # 检查是否有超过4096的序列
        over_limit = [x for x in length_stats if x > max_seq_len]
        if over_limit:
            logging.warning(f"发现 {len(over_limit)} 个超过 {max_seq_len} 的序列:")
            logging.warning(f"  最大长度: {max(over_limit)}")
            logging.warning(f"  平均超长: {np.mean(over_limit):.1f}")

if __name__ == "__main__":
    input_file = "/home/rwkv/cai-RWKV/Multi-midirwkv/gigamidi-multi-track-one.jsonl"
    output_file = "/home/rwkv/cai-RWKV/Multi-midirwkv/gigamidi-split.jsonl"
    
    # 第一步：智能切分序列
    logging.info("=== 第一步：智能切分序列 ===")
    intelligent_split_sequences(
        input_jsonl_path=input_file,
        output_jsonl_path=output_file,
        max_seq_len=4096,
        min_seq_len=512
    )
    
    # 第二步：分析切分质量
    logging.info("\n=== 第二步：分析切分质量 ===")
    analyze_split_quality(output_file, sample_size=5)
    
    # 第三步：验证切分结果
    logging.info("\n=== 第三步：验证切分结果 ===")
    validate_split_results(output_file, max_seq_len=4096)
    
    logging.info("\n=== 处理完成 ===")
    logging.info(f"最终训练数据文件: {output_file}")
