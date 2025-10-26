#!/usr/bin/env python3
"""
Script to download and process the system-prompt-leakage dataset from Hugging Face.
Downloads 100 rows and formats them for the victim_prompts directory.
"""

import json
import os
import pandas as pd
from typing import List, Dict, Any

def download_system_prompt_leakage(num_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Download the system-prompt-leakage dataset from Hugging Face.
    
    Args:
        num_samples: Number of samples to download (default: 100)
    
    Returns:
        List of dictionaries containing the dataset samples
    """
    print(f"Downloading {num_samples} samples from system-prompt-leakage dataset...")
    
    try:
        # Define splits
        splits = {'train': 'train.parquet', 'test': 'test.parquet'}
        
        # Load the dataset using pandas
        df = pd.read_parquet("hf://datasets/gabrielchua/system-prompt-leakage/" + splits["test"])
        
        # Take the first num_samples rows
        samples_df = df.head(min(num_samples, len(df)))
        
        # Convert to list of dictionaries
        samples = samples_df.to_dict('records')
        
        print(f"Successfully downloaded {len(samples)} samples")
        return samples
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def process_for_victim_prompts(samples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Process the downloaded samples to match the victim_prompts format.
    
    Args:
        samples: Raw samples from the dataset
    
    Returns:
        List of dictionaries formatted for victim_prompts
    """
    processed_samples = []
    
    for i, sample in enumerate(samples):
        # Create a victim prompt entry using the system_prompt as the instruction
        # and the content as additional context
        victim_prompt = {
            "id": f"system-prompt-leakage-{i:04d}",
            "instruction": sample["system_prompt"],
            "content": sample["content"],
            "leakage": sample["leakage"]
        }
        processed_samples.append(victim_prompt)
    
    return processed_samples

def save_to_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        filepath: Path to save the file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} samples to {filepath}")

def main():
    """Main function to download and process the dataset."""
    # Configuration
    num_samples = 100
    output_file = "/Users/Hussein Jawad/Desktop/projects/SPE-SC/data/victim_prompts/system-prompt-leakage.jsonl"
    
    try:
        # Download the dataset
        raw_samples = download_system_prompt_leakage(num_samples)
        
        # Process for victim_prompts format
        processed_samples = process_for_victim_prompts(raw_samples)
        
        # Save to file
        save_to_jsonl(processed_samples, output_file)
        
        print(f"\nDataset processing complete!")
        print(f"Downloaded: {len(raw_samples)} samples")
        print(f"Saved to: {output_file}")
        
        # Print a few examples
        print("\nFirst 3 examples:")
        for i, sample in enumerate(processed_samples[:3]):
            print(f"\nExample {i+1}:")
            print(f"ID: {sample['id']}")
            print(f"Instruction: {sample['instruction'][:100]}...")
            print(f"Content: {sample['content'][:100]}...")
            print(f"Leakage: {sample['leakage']}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
