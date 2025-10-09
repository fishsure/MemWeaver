import json
import os
from pathlib import Path

def load_and_show_samples(file_path, file, num_samples=5, max_lines_per_sample=10):
    """Load and display samples from a JSON file, write to file handle"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        file.write(f"\n{'='*50}\n")
        file.write(f"File: {file_path}\n")
        file.write(f"{'='*50}\n")
        
        # Handle both list and dictionary formats
        if isinstance(data, list):
            samples = data[:num_samples]
        elif isinstance(data, dict):
            # If it's a dictionary, take first few items
            samples = list(data.items())[:num_samples]
        else:
            file.write(f"Unexpected data format in {file_path}\n")
            return
            
        file.write(f"Showing {len(samples)} samples:\n")
        for i, sample in enumerate(samples, 1):
            file.write(f"\nSample {i}:\n")
            # Convert sample to JSON string and split into lines
            sample_str = json.dumps(sample, indent=2, ensure_ascii=False)
            lines = sample_str.split('\n')
            # Limit the number of lines
            if len(lines) > max_lines_per_sample:
                lines = lines[:max_lines_per_sample]
                lines.append("... (truncated)")
            file.write('\n'.join(lines) + "\n")
            
    except Exception as e:
        file.write(f"Error reading {file_path}: {str(e)}\n")

def main():
    base_path = "data"
    output_path = os.path.join(base_path, "LaMP_samples.txt")
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # Process each LaMP dataset
        for i in range(1, 8):
            dataset_name = f"LaMP_{i}"
            
            # Regular LaMP files
            files = [
                f"{base_path}/{dataset_name}/train/train_questions.json",
                f"{base_path}/{dataset_name}/train/train_outputs.json",
                f"{base_path}/{dataset_name}/dev/dev_questions.json",
                f"{base_path}/{dataset_name}/dev/dev_outputs.json",
                
                # Time-based LaMP files
                f"{base_path}/{dataset_name}_time/train/train_questions.json",
                f"{base_path}/{dataset_name}_time/train/train_outputs.json",
                f"{base_path}/{dataset_name}_time/dev/dev_questions.json",
                f"{base_path}/{dataset_name}_time/dev/dev_outputs.json"
            ]
            
            out_file.write(f"\n{'#'*80}\n")
            out_file.write(f"Dataset: {dataset_name}\n")
            out_file.write(f"{'#'*80}\n")
            
            for file_path in files:
                if os.path.exists(file_path):
                    load_and_show_samples(file_path, out_file)
                else:
                    out_file.write(f"\nFile not found: {file_path}\n")

if __name__ == "__main__":
    main() 