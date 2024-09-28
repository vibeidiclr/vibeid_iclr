import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
import numpy as np
import cv2
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_and_import(package):
    package_name = package if package != "cv2" else "opencv-python"
    try:
        __import__(package)
    except ImportError:
        logging.info(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    finally:
        globals()[package] = __import__(package)

required_packages = [
    "argparse",
    "torch",
    "torch.nn",
    "torch.optim",
    "torch.utils.data",
    "torchvision.transforms",
    "torchvision.models",
    "cv2",
    "numpy"
]

for package in required_packages:
    install_and_import(package)

def load_images_from_folder(data_dir, num_channels, width, height, labels):
    X = []
    y = []
    extension = [".png", ".jpg", ".jpeg"]
    for label_index, label_dir in enumerate(sorted(os.listdir(data_dir))):
        if not label_dir.startswith("."):
            label_path = os.path.join(data_dir, label_dir)
            i = 0
            channel_data = []

            for fname in sorted(os.listdir(label_path)):
                if any(fname.endswith(ext) for ext in extension) and not fname.startswith("."):
                    i += 1
                    path = os.path.join(label_path, fname)
                    img = cv2.imread(path, cv2.IMREAD_COLOR if num_channels == 3 else cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (width, height))
                    channel_data.append(img)

                    if i % 5 == 0:
                        if num_channels == 1:
                            feature = np.stack(channel_data, axis=-1)
                        elif num_channels == 3:
                            feature = np.concatenate(channel_data, axis=-1)
                        feature = np.transpose(feature, (2, 0, 1))
                        X.append(feature)
                        y.append(label_index)
                        channel_data = []

    return np.asarray(X), np.asarray(y)

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        resnet =  models.resnet18(weights=None)
        for param in resnet.parameters():
          param.requires_grad = True
        resnet.conv1 =  nn.Conv2d(15, 64, kernel_size=7, stride=2, padding=3)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(nn.Linear(num_ftrs,40))
        self.resnet = resnet
    def forward(self, x):
        return self.resnet(x)
custom_resnet = CustomResNet()

def train_and_test_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs):
    logging.info(f"Training the model for {num_epochs} epochs.")
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = test_correct / test_total

        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        if scheduler is not None:
            scheduler.step(test_loss)
    logging.info("Training completed!")

def prepare_data_and_train_model(three_all, target_train_dir, target_test_dir, model_path, num_channels, width, height, num_classes, device, labels, num_epochs=10, batch_size=16):
    target_X_train, target_y_train = load_images_from_folder(target_train_dir, num_channels, width, height, labels)
    target_X_test, target_y_test = load_images_from_folder(target_test_dir, num_channels, width, height, labels)

    target_X_train_tensor = torch.tensor(target_X_train, dtype=torch.float32)
    target_y_train_tensor = torch.tensor(target_y_train, dtype=torch.long)
    target_X_test_tensor = torch.tensor(target_X_test, dtype=torch.float32)
    target_y_test_tensor = torch.tensor(target_y_test, dtype=torch.long)

    target_train_loader = DataLoader(TensorDataset(target_X_train_tensor, target_y_train_tensor), batch_size=batch_size, shuffle=True)
    target_test_loader = DataLoader(TensorDataset(target_X_test_tensor, target_y_test_tensor), batch_size=batch_size, shuffle=False)

    test_correct = 0
    test_total = 0
    test_loss = 0.0
    loaded_model = CustomResNet()
    loaded_model.to(device)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in target_test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = loaded_model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss /= len(target_test_loader)
    test_accuracy = test_correct / test_total

    print(f"Source Accuracy: {test_accuracy:.4f}")

    loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    if three_all==0:
        print(" Finetuning last three layers ")
        for param in loaded_model.resnet.fc.parameters():
            param.requires_grad = True
    else:
        print(" Finetuning all layers ")
        for name, param in loaded_model.named_parameters():
            if 'resnet.layer4' in name or 'resnet.fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(loaded_model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=3)

    train_and_test_model(loaded_model, target_train_loader, target_test_loader, criterion, optimizer, scheduler, device, num_epochs=num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and Testing Script")
    parser.add_argument("--model_path", required=True, help="Path to the pre-trained model")
    parser.add_argument("--target_train_dir", required=True, help="Path to the training data directory")
    parser.add_argument("--target_test_dir", required=True, help="Path to the testing data directory")
    parser.add_argument("--three_all",required=True, type=int, default=1, help="three_layers=0 and all_layers=1")
    parser.add_argument("--num_channels", type=int, default=3, help="Number of channels in the images")
    parser.add_argument("--width", type=int, default=128, help="Width of the images")
    parser.add_argument("--height", type=int, default=128, help="Height of the images")
    parser.add_argument("--num_classes", type=int, default=40, help="Number of classes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the training on")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CustomResNet.to(device)

    logging.info("Device set to: %s", device)

    prepare_data_and_train_model(args.three_all, args.target_train_dir, args.target_test_dir, args.model_path, args.num_channels, args.width, args.height, args.num_classes, args.device, labels=None, num_epochs=args.num_epochs, batch_size=args.batch_size)
