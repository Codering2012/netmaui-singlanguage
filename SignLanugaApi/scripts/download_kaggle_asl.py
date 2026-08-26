import os
import zipfile
import subprocess

def download_asl_dataset():
    """
    Downloads the Google American Sign Language Fingerspelling Recognition dataset from Kaggle.
    Requires kaggle python package and credentials (~/.kaggle/kaggle.json)
    """
    dataset = "asl-fingerspelling"
    print(f"Checking for kaggle API...")
    
    try:
        import kaggle
    except ImportError:
        print("Kaggle API not found. Installing...")
        subprocess.run(["pip", "install", "kaggle"], check=True)
        import kaggle

    # Create target directory
    target_dir = os.path.join(os.getcwd(), "DATASETS", "asl_fingerspelling")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    print(f"Downloading dataset '{dataset}' to {target_dir}...")
    
    # Download
    subprocess.run([
        "kaggle", "competitions", "download", "-c", dataset, 
        "-p", target_dir
    ], check=True)
    
    # Unzip
    zip_path = os.path.join(target_dir, f"{dataset}.zip")
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(zip_path)
        print("Extraction complete.")
    else:
        print(f"Zip file not found at {zip_path}. Check if download was successful.")

if __name__ == "__main__":
    download_asl_dataset()
