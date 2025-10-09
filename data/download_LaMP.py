import os
import requests
from pathlib import Path
from tqdm import tqdm

def create_directories(base_path, dataset_name):
    """Create directory structure for a dataset"""
    paths = [
        f"{base_path}/{dataset_name}/train",
        f"{base_path}/{dataset_name}/dev",
        f"{base_path}/{dataset_name}_time/train",
        f"{base_path}/{dataset_name}_time/dev"
    ]
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def download_file(url, save_path):
    """Download a file from URL and save it to the specified path with progress bar and resume capability"""
    try:
        # Check if file already exists
        if os.path.exists(save_path):
            print(f"File already exists: {save_path}")
            return

        # Get file size if it exists
        initial_pos = 0
        if os.path.exists(save_path):
            initial_pos = os.path.getsize(save_path)

        # Set up headers for resume
        headers = {'Range': f'bytes={initial_pos}-'} if initial_pos > 0 else {}
        
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0)) + initial_pos
        
        # Create progress bar
        progress_bar = tqdm(
            total=total_size,
            initial=initial_pos,
            unit='iB',
            unit_scale=True,
            desc=f"Downloading {os.path.basename(save_path)}"
        )
        
        # Open file in append mode if resuming, write mode if new
        mode = 'ab' if initial_pos > 0 else 'wb'
        with open(save_path, mode) as f:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                progress_bar.update(size)
        
        progress_bar.close()
        print(f"Successfully downloaded: {save_path}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        # If download fails, keep the partial file for resume
        if not os.path.exists(save_path):
            print("Download failed, partial file will be used for resume")

def get_file_urls(dataset_name, base_url):
    """Get the appropriate URLs for a dataset, handling special cases"""
    if dataset_name == "LaMP_2":
        # Special case for LaMP_2 with "new" directory
        return [
            # Regular LaMP files
            f"{base_url}/{dataset_name}/new/train/train_outputs.json",
            f"{base_url}/{dataset_name}/new/train/train_questions.json",
            f"{base_url}/{dataset_name}/new/dev/dev_outputs.json",
            f"{base_url}/{dataset_name}/new/dev/dev_questions.json",
            
            # Time-based LaMP files with "new" directory
            f"{base_url}/time/{dataset_name}/new/train/train_outputs.json",
            f"{base_url}/time/{dataset_name}/new/train/train_questions.json",
            f"{base_url}/time/{dataset_name}/new/dev/dev_outputs.json",
            f"{base_url}/time/{dataset_name}/new/dev/dev_questions.json"
        ]
    else:
        # Standard URL structure for other datasets
        return [
            # Regular LaMP files
            f"{base_url}/{dataset_name}/train/train_outputs.json",
            f"{base_url}/{dataset_name}/train/train_questions.json",
            f"{base_url}/{dataset_name}/dev/dev_outputs.json",
            f"{base_url}/{dataset_name}/dev/dev_questions.json",
            
            # Time-based LaMP files
            f"{base_url}/time/{dataset_name}/train/train_outputs.json",
            f"{base_url}/time/{dataset_name}/train/train_questions.json",
            f"{base_url}/time/{dataset_name}/dev/dev_outputs.json",
            f"{base_url}/time/{dataset_name}/dev/dev_questions.json"
        ]

def main():
    base_path = "data"
    base_url = "https://ciir.cs.umass.edu/downloads/LaMP"
    
    # Create base directory if it doesn't exist
    Path(base_path).mkdir(exist_ok=True)
    
    # Download files for LaMP_1 through LaMP_7
    for i in range(1, 8):
        dataset_name = f"LaMP_{i}"
        
        # Create directories
        create_directories(base_path, dataset_name)
        
        # Files to download for each dataset
        files = get_file_urls(dataset_name, base_url)
        
        # Corresponding save paths
        save_paths = [
            f"{base_path}/{dataset_name}/train/train_outputs.json",
            f"{base_path}/{dataset_name}/train/train_questions.json",
            f"{base_path}/{dataset_name}/dev/dev_outputs.json",
            f"{base_path}/{dataset_name}/dev/dev_questions.json",
            f"{base_path}/{dataset_name}_time/train/train_outputs.json",
            f"{base_path}/{dataset_name}_time/train/train_questions.json",
            f"{base_path}/{dataset_name}_time/dev/dev_outputs.json",
            f"{base_path}/{dataset_name}_time/dev/dev_questions.json"
        ]
        
        # Download each file
        for url, save_path in zip(files, save_paths):
            download_file(url, save_path)

if __name__ == "__main__":
    main()