import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

def create_dataloaders(output_dir, num_classes, batch_size=16, num_workers=2, num_channels=3, width=256, height=256):
    logging.info(f"Creating dataloaders with batch size {batch_size}, {num_workers} workers, {num_channels} channels, width {width}, and height {height}.")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    labels = sorted(os.listdir(train_dir))[:num_classes]
    X_train, y_train = load_images_from_folder(train_dir, num_channels, width, height, labels)
    X_test, y_test = load_images_from_folder(test_dir, num_channels, width, height, labels)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # logging.info("Train data shape (X_train): %s", X_train_tensor.shape)
    # logging.info("Train data labels shape (y_train): %s", y_train_tensor.shape)
    # logging.info("Test data shape (X_test): %s", X_test_tensor.shape)
    # logging.info("Test data labels shape (y_test): %s", y_test_tensor.shape)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    logging.info("Data loaders created.")
    return train_loader, test_loader

class CustomResNet(nn.Module):
    def __init__(self, model_type='resnet50'):
        super(CustomResNet, self).__init__()
        if model_type == 'resnet18':
            resnet = models.resnet18(weights=None)
        elif model_type == 'resnet50':
            resnet = models.resnet50(weights=None)
        else:
            raise ValueError("Unsupported model_type. Choose either 'resnet18' or 'resnet50'.")

        for param in resnet.parameters():
            param.requires_grad = True
        resnet.conv1 = nn.Conv2d(15, 64, kernel_size=7, stride=2, padding=3)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(nn.Linear(num_ftrs, 30))
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)

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

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"Model saved at epoch {epoch + 1} with accuracy {test_accuracy:.4f}")

    logging.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test a PyTorch model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train the model')
    parser.add_argument('--num_classes', type=int, default=15, help='Number of classes for the output layer')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of channels for input images')
    parser.add_argument('--width', type=int, default=256, help='Width of input images')
    parser.add_argument('--height', type=int, default=256, help='Height of input images')
    parser.add_argument('--model_type', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='Model type to use (resnet18 or resnet50)')

    args = parser.parse_args()

    logging.info(f"Arguments: {args}")

    train_loader, test_loader = create_dataloaders(args.output_dir, args.num_classes, args.batch_size, args.num_workers, args.num_channels, args.width, args.height)

    custom_resnet = CustomResNet(model_type=args.model_type)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(custom_resnet.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_resnet.to(device)
    logging.info("Device set to: %s", device)

    train_and_test_model(custom_resnet, train_loader, test_loader, criterion, optimizer, scheduler, device, args.num_epochs)
