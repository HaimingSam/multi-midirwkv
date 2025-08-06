import pandas as pd
import json
import os
from miditok import REMI, TokenizerConfig
from tqdm import tqdm
import tempfile
import mido # Import mido to validate MIDI files more robustly
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def convert_parquet_to_jsonl(parquet_path, output_dir, tokenizer):
    df = pd.read_parquet(parquet_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gigamidi-one31.jsonl")

    processed_count = 0
    skipped_invalid_header = 0
    skipped_parsing_error = 0
    skipped_no_tokens = 0

    with open(output_path, "w") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            midi_data = row["music"]

            # --- Robust MIDI Header and File Validation ---
            if len(midi_data) < 4 or midi_data[:4] != b'MThd':
                logging.warning(f"Row {idx}: Invalid MIDI file header! Skipping.")
                skipped_invalid_header += 1
                continue

            try:
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as temp_midi_file:
                    temp_midi_file.write(midi_data)
                    temp_midi_file.flush()

                    # Use mido to perform a more thorough MIDI file validation
                    try:
                        mido.MidiFile(temp_midi_file.name)
                    except Exception as e:
                        logging.warning(f"Row {idx}: Mido validation failed: {e}. Skipping.")
                        skipped_parsing_error += 1
                        continue

                    # Core: Get tokenizer encoding results (single stream)
                    tokens = tokenizer.encode(temp_midi_file.name)
                    
                    # Since one_token_stream=True, tokens.ids contains the single token stream
                    if tokens.ids and len(tokens.ids) > 0:  # Check if the token stream is not empty
                        json_obj = {"midi_tokens": tokens.ids}
                        f.write(json.dumps(json_obj) + "\n")
                        processed_count += 1
                    else:
                        logging.warning(f"Row {idx}: No valid tokens generated after encoding. Skipping.")
                        skipped_no_tokens += 1

            except Exception as e:
                # Catch general exceptions during processing, including miditok's internal errors
                logging.error(f"Row {idx}: Error during tokenization: {e}. Skipping.")
                skipped_parsing_error += 1
    
    logging.info(f"\n--- Processing Summary ---")
    logging.info(f"Total rows processed successfully: {processed_count}")
    logging.info(f"Skipped due to invalid MIDI header: {skipped_invalid_header}")
    logging.info(f"Skipped due to parsing/tokenization errors: {skipped_parsing_error}")
    logging.info(f"Skipped due to no valid tokens generated: {skipped_no_tokens}")


def save_tokenizer_vocab(tokenizer, output_path):
    """Save the tokenizer's vocabulary to a JSON file"""
    vocab = tokenizer.vocab
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=4)

if __name__ == "__main__":
    # Full tokenizer configuration
    config = TokenizerConfig(
        use_programs=True,  # Enable instrument program numbers
        programs=list(range(128)) + [-1],  # Include all 128 standard MIDI program numbers and -1 (drum set)
        # one_token_stream_for_programs=True,  # merge all tracks into one token stream
        program_changes=True,  # Allow instrument changes
        one_token_stream=True, # Single stream mode
        use_chords=True,
        use_pitchdrum_tokens=True
    )
    
    # Initialize tokenizer
    tokenizer = REMI(config)
    
    # Explicitly save tokenizer configuration
    tokenizer.save("/home/rwkv/cai-RWKV/Multi-midirwkv/tokenizer_params.json")
    
    # Save vocabulary
    save_tokenizer_vocab(tokenizer, "/home/rwkv/cai-RWKV/Multi-midirwkv/tokenizer_vocab.json")

    # Process data
    convert_parquet_to_jsonl(
        parquet_path="/home/rwkv/cai-RWKV/hf-mirrors/GigaMIDI/v1.0.0/train.parquet",
        output_dir="/home/rwkv/cai-RWKV/Multi-midirwkv",
        tokenizer=tokenizer
    )
