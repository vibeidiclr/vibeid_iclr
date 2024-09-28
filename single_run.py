import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

def create_dataloaders(output_dir, batch_size=16, num_workers=2):
    print(f"Creating dataloaders with batch size {batch_size} and {num_workers} workers.")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Data loaders created.")
    return train_loader, test_loader

def train_and_test_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs):
    print(f"Training the model for {num_epochs} epochs.")
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

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        if scheduler is not None:
            scheduler.step(test_loss)

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved at epoch {epoch + 1} with accuracy {test_accuracy:.4f}")

    print("Training completed!")

def get_model(model_name, num_classes):
    print(f"Loading model: {model_name} with {num_classes} output classes.")
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for param in model.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes))
    print("Model loaded.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test a PyTorch model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the model')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='Model type to use (resnet18 or resnet50)')
    parser.add_argument('--num_classes', type=int, default=15, help='Number of classes for the output layer')

    args = parser.parse_args()

    print(f"Arguments: {args}")

    train_loader, test_loader = create_dataloaders(args.output_dir, args.batch_size, args.num_workers)

    custom_resnet = get_model(args.model, args.num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(custom_resnet.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_resnet.to(device)
    print("Device set to:", device)

    train_and_test_model(custom_resnet, train_loader, test_loader, criterion, optimizer, scheduler, device, args.num_epochs)
