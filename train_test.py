<<<<<<< HEAD
import os
import shutil
from sklearn.model_selection import train_test_split
import argparse

def split_data(data_dir, output_dir, test_size):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all subdirectories (class-wise folders) in the data directory
    classes = os.listdir(data_dir)

    # Iterate over each class directory
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)

        # List all files in the class directory
        files = os.listdir(class_dir)

        # Split files into train and test sets
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)

        # Create directories for train and test data if they don't exist
        train_dir = os.path.join(output_dir, 'train', class_name)
        test_dir = os.path.join(output_dir, 'test', class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Move train files to train directory
        for file in train_files:
            src = os.path.join(class_dir, file)
            dest = os.path.join(train_dir, file)
            shutil.copy(src, dest)

        # Move test files to test directory
        for file in test_files:
            src = os.path.join(class_dir, file)
            dest = os.path.join(test_dir, file)
            shutil.copy(src, dest)

    print("Data separated and saved into train and test directories.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing class-wise folders.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory where you want to save the train and test data.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio.')

    args = parser.parse_args()
    split_data(args.data_dir, args.output_dir, args.test_size)
=======
import os
import shutil
from sklearn.model_selection import train_test_split
import argparse

def split_data(data_dir, output_dir, test_size):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all subdirectories (class-wise folders) in the data directory
    classes = os.listdir(data_dir)

    # Iterate over each class directory
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)

        # List all files in the class directory
        files = os.listdir(class_dir)

        # Split files into train and test sets
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)

        # Create directories for train and test data if they don't exist
        train_dir = os.path.join(output_dir, 'train', class_name)
        test_dir = os.path.join(output_dir, 'test', class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Move train files to train directory
        for file in train_files:
            src = os.path.join(class_dir, file)
            dest = os.path.join(train_dir, file)
            shutil.copy(src, dest)

        # Move test files to test directory
        for file in test_files:
            src = os.path.join(class_dir, file)
            dest = os.path.join(test_dir, file)
            shutil.copy(src, dest)

    print("Data separated and saved into train and test directories.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing class-wise folders.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory where you want to save the train and test data.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio.')

    args = parser.parse_args()
    split_data(args.data_dir, args.output_dir, args.test_size)
>>>>>>> 8fba5b9531fdd10712358e24f2af79ec370a2980
