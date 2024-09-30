# VIBeID: A Structural **VIB**ration-based Soft Biometric Dataset and Benchmark for Person **ID**entification
This repository provides a script to download Pre-processed  VIBeID datasets, create DataLoaders for training and testing, and train a ResNet-18 and ResNet-50 model using PyTorch.

![1717854965704](image/README/1717854965704.png)
## Requirements
- Python 3.x
- `pip` (Python package installer)
- Kaggle API key (`kaggle.json`) [optional]

## Arguments

### `--kaggle_dataset`
- **Description**: Kaggle dataset identifier in the format `mainakml/dataset-name`.
- **Type**: `str`
- **Default**: `'mainakml/vibeid-a-4-1'`
- **Example**: `--kaggle_dataset yourusername/yourdataset`
- **Note**: This argument specifies which dataset to download from Kaggle.

### `--output_dir`
- **Description**: Directory to download and unzip the Kaggle dataset.
- **Type**: `str`
- **Default**: `'vibeid-a-4-1/VIBeID_A_4_1'`
- **Example**: `--output_dir /path/to/output_dir`
- **Note**: The script will create this directory if it does not exist and will store the downloaded dataset here.

### `--batch_size`
- **Description**: Batch size for the DataLoader.
- **Type**: `int`
- **Default**: `16`
- **Example**: `--batch_size 16`
- **Note**: This determines the number of samples that will be propagated through the network at once.

### `--num_workers`
- **Description**: Number of worker threads to use for loading the data.
- **Type**: `int`
- **Default**: `2`
- **Example**: `--num_workers 4`
- **Note**: This is used to speed up data loading by using multiple threads.

### `--num_epochs`
- **Description**: Number of epochs to train the model.
- **Type**: `int`
- **Default**: `50`
- **Example**: `--num_epochs 30`
- **Note**: One epoch means that each sample in the dataset has had an opportunity to update the internal model parameters once.

### `--model`
- **Description**: Model type to use for training.
- **Type**: `str`
- **Choices**: `['resnet18', 'resnet50']`
- **Default**: `resnet18`
- **Example**: `--model resnet50`
- **Note**: Specifies which ResNet model architecture to use.

### `--num_classes`
- **Description**: Number of output classes for the model.
- **Type**: `int`
- **Default**: `15`
- **Example**: `--num_classes 15/30/40/100`
- **Note**: This should match the number of classes in your dataset.

### `--three_all`
- **Description**: Fine tune last 3 layers or all layers
- **Type**: `int`
- **Default**: `0`
- **Example**: `--three_all 0/1`
- **Note**: 0:Fine tune last 3 layers, 1: Fine tune all layers.


## Step-by-Step guide
## Convert Signal to CWT images
- Run spec_maker.py 
``` python spec_maker.py --file_path "A2_2_30p.mat" --notebook_path "folder_to_save_CWT _images"```
- Run train test file
``` python train_test.py --data_dir "CWT_image_folder_name" --output_dir "folder_to_save" --test_size 0.2```
<<<<<<< HEAD
=======

>>>>>>> 8fba5b9531fdd10712358e24f2af79ec370a2980
## Person Identification using Deep learning 
### Quick Run 
- Run Multi-class Classification (Single Image) - [single_run_demo.ipynb](https://github.com/vibeidiclr/vibeid_iclr/blob/main/single_run_demo.ipynb)
- Run Multi-class Classification (Multi Image)- [multi_run_demo.ipynb](https://github.com/vibeidiclr/vibeid_iclr/blob/main/multi_run_demo.ipynb)
- Run Domain Adaptation demo - [domain_adaptation_demo.ipynb](https://github.com/vibeidiclr/vibeid_iclr/blob/main/domain_adaptation_demo.ipynb)
### STEP 1: Install Libraries:
python install_libraries.py


### STEP 2: Download the Datasets
You can download the datasets from the Kaggle (dataset is public)

1. vibeid-a1 [A1](https://www.kaggle.com/datasets/mainakml/vibeid-a1)
2. vibeid-a2 [A2](https://www.kaggle.com/datasets/mainakml/vibeid-a2)
3. vibeid-a3 [A3](https://www.kaggle.com/datasets/mainakml/vibeid-a3)
4. vibeid-a4 [A4](https://www.kaggle.com/datasets/mainakml/vibeid-a-4-1)

OR 
run 

```python kaggle_dataset_download.py --kaggle_dataset "mainakml/dataset link"```

Quick  Run 
```python kaggle_dataset_download.py --kaggle_dataset "mainakml/vibeid-a-4-1"```

change the dataset link as your requirement
1. mainakml/vibeid-a1
2. mainakml/vibeid-a2
3. mainakml/vibeid-a3
4. mainakml/vibeid-a-4-1


### STEP 3: Quick Run

```python single_run.py --output_dir C:\Users\mainak\Documents\GitHub\VIBEID\VIBeID_A_4_1 --batch_size 16 --num_epochs 100 --model resnet18 --num_classes 15```

### STEP 4: Run dataset as per your requirement

### single_image_run
```python single_run.py --output_dir "add dataset link which contains train and test" --batch_size 16 --num_epochs 100 --model resnet18 --num_classes 15/30/40/100```


### multi_image_run
```python multi_run.py --output_dir "add dataset link which contains train and test" --batch_size 16 --num_epochs 100 --model resnet18 --num_classes 15/30/40/100```

## Domain Adaptation using Deep learning 
- Pretrained models are available in the folder
- change the path to the target test and val directory
- update three_all parameter
```python domain_run.py --model_path resnet_18_RGB_A3.1_100.pth --target_train_dir "add path to test"\test --target_test_dir  "add path to val"\val --three_all 0/1```
---

