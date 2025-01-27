# README for Assignment No. 4: Flower Classification using CNNs

## Project Overview
This project involves the classification of flowers into their respective categories using two pre-trained convolutional neural networks (CNNs): YOLOv5 and VGG19. The assignment demonstrates the application of transfer learning and deep learning techniques to achieve high accuracy in flower image classification. The Oxford 102 Flowers dataset was utilized for training, validation, and testing.

---

## Submission Details
**Assignment No.:** 4  
**Submission Date:** 2/2/2025  

---

## Folder Structure
```
ml4/
├── create_dir.py       # Script to create directory structures for YOLO classification
├── imagelabels.mat     # Label mapping file for the Oxford dataset
├── jpg/                # Folder containing flower images
├── models/             # Pre-trained YOLOv5 and VGG19 models
├── output/             # Directory the directories for YOLO training
├── plots/              # Directory containing accuracy and loss graphs
├── plot_yolo.py        # Script for plotting YOLOv5 results
├── results.csv         # CSV file summarizing classification results
├── run_me_vgg.sh       # Shell script to run VGG19 model
├── run_me_yolo.sh      # Shell script to run YOLOv5 model
├── vgg.py              # Python script for VGG19 implementation
```

---

## Preprocessing
The following steps were performed to preprocess the images:
1. **Resizing:** All images were resized to 224x224 pixels for compatibility with VGG19 and YOLOv5.
2. **Normalization:** Pixel values were scaled to the range [0, 1].
3. **Data Augmentation:** Techniques such as random rotations, flips, and zooms were applied to improve generalization.
4. **Dataset Split:** The dataset was randomly divided into training (50%), validation (25%), and testing (25%) sets. The split was repeated twice to ensure robustness.

---

## Model Details
### 1. **VGG19**
- **Architecture:** The VGG19 model includes 19 layers with convolutional, max pooling, and fully connected layers.
- **Modifications:**
  - Replaced the top layer with a fully connected layer with 102 output nodes (one for each flower category).
  - Used a softmax activation function for probabilistic output.
- **Optimizer:** Adam optimizer with a learning rate of 0.001.
- **Loss Function:** Cross-Entropy Loss.

### 2. **YOLOv5**
- **Architecture:** YOLOv5 is a real-time object detection model used in classification mode for this project.
- **Modifications:**
  - Configured YOLOv5 for flower classification by retraining the classification layers.
- **Optimizer:** SGD optimizer with momentum.
- **Loss Function:** Cross-Entropy Loss for classification tasks.

---

## Results
### Performance Metrics
- **Accuracy:**
  - VGG19: Achieved >90% test accuracy.
  - YOLOv5: Achieved >90% test accuracy.
- **Graphs:**
  - Accuracy and Cross-Entropy Loss for training, validation, and testing were plotted as a function of epochs.
  - Graphs are stored in the `plots/` directory.

### Results Summary
- Results of YOLO are documented in `results.csv`.
- Key findings:
  - Both VGG19 and YOLOv5 achieved high accuracy (>90%).
  - Data augmentation, preprocessing, and hyperparameter tuning were crucial for performance improvements.

---

## Usage
1. Clone the repository from GitHub:
   ```
   git clone <GitHub Repository URL>
   ```
2. Navigate to the project directory:
   ```
   cd ml4
   ```
3. Prepare directories for YOLOv5 classification by running:
   ```
   python3 create_dir.py
   ```
4. Run the models:
   - For VGG19:
     ```
     ./run_me_vgg.sh
     ```
   - For YOLOv5:
     ```
     ./run_me_yolo.sh
     ```
5. View the directories for YOLO train in the `output/` directory and performance graphs in the `plots/` directory.
6. Setup
   - All models traind on rtx_6000

---

## Additional Resources
- **Datasets:**
  - [Oxford 102 Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)



