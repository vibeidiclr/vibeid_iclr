import subprocess
import sys
import argparse
import os
import json


def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)


required_packages = [
    "argparse",
    "kaggle",
    "os",
    "subprocess",
    "json"
]

for package in required_packages:
    install_and_import(package)


def create_kaggle_json(username, key):
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    kaggle_json_content = {
        "username": username,
        "key": key
    }

    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json_path, 'w') as f:
        json.dump(kaggle_json_content, f)

    os.chmod(kaggle_json_path, 0o600)
    return kaggle_json_path


def download_data_from_kaggle(username, key, kaggle_dataset):
    output_dir = os.getcwd()  # Current working directory

    dataset_name = kaggle_dataset.split('/')[-1]
    dataset_dir = os.path.join(output_dir, dataset_name)

    if os.path.exists(dataset_dir):
        print(f"Data already exists in {dataset_dir}. Skipping download.")
        return

    print(f"Downloading data from Kaggle: {kaggle_dataset} to {output_dir}")
    create_kaggle_json(username, key)

    subprocess.run(["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", output_dir, "--unzip"], check=True)
    print("Download completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download data from Kaggle and unzip it')
    parser.add_argument('--username', type=str, default="mainakml", help='Kaggle username')
    parser.add_argument('--key', type=str, default="eecece63dc6b1a4335312b903042113c", help='Kaggle API key')
    parser.add_argument('--kaggle_dataset', type=str, required=True, help='Kaggle dataset identifier (e.g., username/dataset-name)')

    args = parser.parse_args()

    print(f"Arguments: {args}")

    download_data_from_kaggle(args.username, args.key, args.kaggle_dataset)
