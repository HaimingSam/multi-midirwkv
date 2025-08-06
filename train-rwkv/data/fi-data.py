import json
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def filter_multi_track_songs(input_jsonl_path, output_jsonl_path):
    """
    从JSONL文件中过滤出多轨道的曲子（至少2个不同的程序/乐器）
    """
    processed_count = 0
    skipped_single_track = 0
    skipped_no_programs = 0
    total_count = 0
    
    # 首先计算总行数用于进度条
    with open(input_jsonl_path, 'r') as f:
        total_count = sum(1 for line in f)
    
    with open(input_jsonl_path, 'r') as input_file, open(output_jsonl_path, 'w') as output_file:
        for line_num, line in enumerate(tqdm(input_file, total=total_count, desc="Processing songs")):
            try:
                data = json.loads(line.strip())
                midi_tokens = data.get('midi_tokens', [])
                
                if not midi_tokens:
                    logging.warning(f"Line {line_num}: No tokens found, skipping.")
                    skipped_no_programs += 1
                    continue
                
                # 检查是否包含多个不同的程序
                programs_found = set()
                
                for token in midi_tokens:
                    # 检查是否是程序变化token (Program_0 到 Program_127, Program_-1)
                    if isinstance(token, int):
                        # 根据您的词汇表，Program tokens的范围是298-426
                        if 298 <= token <= 426:
                            # 将token ID转换为程序号
                            if token == 426:  # Program_-1
                                program_num = -1
                            else:
                                program_num = token - 298  # Program_0 = 298, Program_1 = 299, etc.
                            programs_found.add(program_num)
                
                # 检查是否至少有2个不同的程序
                if len(programs_found) >= 2:
                    output_file.write(line)
                    processed_count += 1
                    if processed_count <= 5:  # 显示前5个文件的程序信息
                        logging.info(f"Line {line_num}: Found {len(programs_found)} programs: {sorted(programs_found)}")
                else:
                    logging.warning(f"Line {line_num}: Only {len(programs_found)} program(s) found: {sorted(programs_found)}, skipping.")
                    skipped_single_track += 1
                    
            except json.JSONDecodeError as e:
                logging.error(f"Line {line_num}: JSON decode error: {e}")
                continue
            except Exception as e:
                logging.error(f"Line {line_num}: Unexpected error: {e}")
                continue
    
    logging.info(f"\n--- Filtering Summary ---")
    logging.info(f"Total songs processed: {total_count}")
    logging.info(f"Multi-track songs kept: {processed_count}")
    logging.info(f"Skipped single-track songs: {skipped_single_track}")
    logging.info(f"Skipped songs with no programs: {skipped_no_programs}")
    logging.info(f"Retention rate: {processed_count/total_count*100:.2f}%")

def analyze_program_distribution(input_jsonl_path, sample_size=100):
    """
    分析程序分布情况，帮助了解数据特征
    """
    program_counts = {}
    total_songs = 0
    
    with open(input_jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
                
            try:
                data = json.loads(line.strip())
                midi_tokens = data.get('midi_tokens', [])
                
                programs_found = set()
                for token in midi_tokens:
                    if isinstance(token, int) and 298 <= token <= 426:
                        if token == 426:
                            program_num = -1
                        else:
                            program_num = token - 298
                        programs_found.add(program_num)
                
                program_count = len(programs_found)
                program_counts[program_count] = program_counts.get(program_count, 0) + 1
                total_songs += 1
                
            except Exception as e:
                continue
    
    logging.info(f"\n--- Program Distribution Analysis (Sample of {sample_size}) ---")
    for count in sorted(program_counts.keys()):
        percentage = program_counts[count] / total_songs * 100
        logging.info(f"Songs with {count} program(s): {program_counts[count]} ({percentage:.1f}%)")

if __name__ == "__main__":
    input_file = "/home/rwkv/cai-RWKV/Multi-midirwkv/gigamidi-one31.jsonl"
    output_file = "/home/rwkv/cai-RWKV/Multi-midirwkv/gigamidi-multi-track-one.jsonl"
    
    # 首先分析数据分布
    logging.info("Analyzing program distribution...")
    analyze_program_distribution(input_file)
    
    # 执行过滤
    logging.info("\nStarting multi-track filtering...")
    filter_multi_track_songs(input_file, output_file)
    
    logging.info(f"\nFiltering completed!")
    logging.info(f"Input file: {input_file}")
    logging.info(f"Output file: {output_file}")
