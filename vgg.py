import os
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import vgg19, VGG19_Weights
from scipy.io import loadmat
from PIL import Image
import sys

sys.stdout.flush()  # Force flush for this print


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to store the dataset
data_dir = 'jpg/'  # Replace with your dataset path

# Load the image labels from the .mat file
label_file =  'imagelabels.mat'
labels_data = loadmat(label_file)

# Assuming the .mat file contains 'labels' and 'file_names'
image_labels = labels_data['labels'].flatten()  # Flatten to 1D array

# Convert paths from numpy array to list of strings
image_paths = [os.path.join(data_dir, f"image_{i + 1:05d}.jpg") for i in range(image_labels.shape[0])]
image_labels = image_labels - 1  # Subtract 1 to make labels 0-indexed

# Define a custom dataset class
class Oxford102Dataset(Dataset):
    def __init__(self, image_paths, image_labels, transform=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.image_labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Define image transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Create the dataset
dataset = Oxford102Dataset(image_paths, image_labels, transform=data_transforms)

# Split dataset into train, validation, and test sets
train_size = int(0.5 * len(dataset))  # 50% for training
val_size = int(0.25 * len(dataset))  # 25% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 25% for testing

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

BATCH_SIZE = 128
# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the pretrained VGG19 model
model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

# Modify the classifier for the Oxford 102 dataset
NUM_CLASSES = 102
model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Variables to store the loss and accuracy values for plotting
train_losses, val_losses, test_losses = [], [], []
train_accuracies, val_accuracies, test_accuracies = [], [], []

# Training function
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_steps = len(train_loader)

        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % (total_steps // 10) == 0 or step == total_steps - 1:
                epoch_loss = running_loss / total
                epoch_acc = correct / total
                print(f"Step [{step + 1}/{total_steps}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        total_steps = len(val_loader)

        with torch.no_grad():
            for step, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if (step + 1) % (total_steps // 10) == 0 or step == total_steps - 1:
                    epoch_loss = running_loss / total
                    epoch_acc = correct / total
                    print(f"Step [{step + 1}/{total_steps}], Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        val_losses.append(epoch_loss)
        val_accuracies.append(epoch_acc)

        # Test phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        total_steps = len(test_loader)

        with torch.no_grad():
            for step, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if (step + 1) % (total_steps // 10) == 0 or step == total_steps - 1:
                    epoch_loss = running_loss / total
                    epoch_acc = correct / total
                    print(f"Step [{step + 1}/{total_steps}], Test Loss: {epoch_loss:.4f}, Test Acc: {epoch_acc:.4f}")

        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_acc = correct / total
        print(f"Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_acc)

    return model


# Train the model
model = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=100)


# Plotting function for accuracy and loss (including test)
def plot_metrics(train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies, num_epochs):
    epochs = range(1, num_epochs + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.plot(epochs, test_losses, label='Test Loss', color='green')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()


# Create and save the graphs
plot_metrics(train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies, num_epochs=100)


# Evaluate on the test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")


evaluate_model(model, test_loader)
