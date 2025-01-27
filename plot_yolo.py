import matplotlib.pyplot as plt

import pandas as pd

# Load the CSV file to examine its structure
file_path = 'results.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head(), data.columns



# Clean up column names by stripping leading/trailing spaces
data.columns = data.columns.str.strip()

# Convert columns to appropriate numeric types (if needed)
data['epoch'] = data['epoch'].astype(int)
data['train/loss'] = pd.to_numeric(data['train/loss'], errors='coerce')
data['test/loss'] = pd.to_numeric(data['test/loss'], errors='coerce')
data['metrics/accuracy_top1'] = pd.to_numeric(data['metrics/accuracy_top1'], errors='coerce')

# Plot the Cross-Entropy loss for train and test datasets
plt.figure(figsize=(12, 6))
plt.plot(data['epoch'], data['train/loss'], label='Train Loss', marker='o')
plt.plot(data['epoch'], data['test/loss'], label='Test Loss', marker='o')
plt.title('Cross-Entropy Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('yolo_loss.png')
plt.close()

# Plot the accuracy (top-1) for train and test datasets
plt.figure(figsize=(12, 6))
plt.plot(data['epoch'], data['metrics/accuracy_top1'], label='Test Accuracy (Top-1)', marker='o')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('yolo_accuracy.png')
plt.close()
