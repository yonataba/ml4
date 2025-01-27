import os
import shutil
import scipy.io
from tqdm import tqdm

# Paths to your files and directories
mat_file = "imagelabels.mat"  # Path to the .mat file
images_dir = "jpg"            # Directory containing images
output_dir = "output"         # Output directory for classification dataset

# Create output directories
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# Split ratios
train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25

# Load the .mat file
mat_data = scipy.io.loadmat(mat_file)
labels = mat_data['labels'][0]  # Extract labels as a 1D array
num_classes = len(set(labels))

# Ensure class directories exist
for cls in range(1, num_classes + 1):  # Assuming class IDs start at 1
    os.makedirs(os.path.join(train_dir, f"class_{cls}"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, f"class_{cls}"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, f"class_{cls}"), exist_ok=True)

# Get all image files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])

# Sanity check
if len(image_files) != len(labels):
    raise ValueError(f"Number of images ({len(image_files)}) does not match number of labels ({len(labels)})!")

# Process images and assign to train/val/test
for idx, image_file in enumerate(tqdm(image_files, desc="Processing images")):
    # Get the label (1-based index)
    label = labels[idx]

    # Determine the split using the hash of the filename
    hash_value = hash(image_file) % 100
    if hash_value < train_ratio * 100:
        subset = "train"
        subset_dir = train_dir
    elif hash_value < (train_ratio + val_ratio) * 100:
        subset = "val"
        subset_dir = val_dir
    else:
        subset = "test"
        subset_dir = test_dir

    # Copy the image to the corresponding class folder
    class_dir = os.path.join(subset_dir, f"class_{label}")
    shutil.copy(os.path.join(images_dir, image_file), os.path.join(class_dir, image_file))

print("Dataset creation complete!")
