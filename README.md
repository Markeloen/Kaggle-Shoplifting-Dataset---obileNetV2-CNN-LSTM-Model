# Shoplifting Detection using Video Analysis

This notebook demonstrates a deep learning approach to detect shoplifting events from video data. It utilizes a MobileNetV2-based CNN combined with an LSTM layer to classify video sequences as either 'normal' or 'shoplifting'.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Dataset](#dataset)
3.  [Data Preprocessing](#data-preprocessing)
4.  [Model Architecture](#model-architecture)
5.  [Training and Evaluation](#training-and-evaluation)
6.  [Results and Analysis](#results-and-analysis)

## 1. Introduction
This project aims to build a video classification model capable of identifying shoplifting incidents. The approach involves extracting frames from video clips, processing them using a pre-trained Convolutional Neural Network (CNN) (MobileNetV2), and then feeding the features into a Long Short-Term Memory (LSTM) network to capture temporal dependencies.

## 2. Dataset
The dataset used is the 'Shoplifting Video Dataset' by 'kipshidze' from Kaggle. It consists of video clips categorized into:
*   `normal`: Videos depicting typical store activities.
*   `shoplifting`: Videos showing shoplifting events.

**Dataset Details:**
*   Total Videos: 182
*   Normal Videos: 90
*   Shoplifting Videos: 92

## 3. Data Preprocessing
The preprocessing steps include:
*   **Video Loading:** A custom `load_video` function extracts a fixed number of frames (16 frames) from each video.
*   **Resizing:** Frames are resized to 112x112 pixels.
*   **Normalization:** Pixel values are normalized to the range [0, 1].
*   **Splitting:** The dataset is split into training (80%) and testing (20%) sets, ensuring stratification to maintain class distribution.
*   **MobileNetV2 Preprocessing:** Input frames are scaled to [-1, 1] as required by the MobileNetV2 model.

## 4. Model Architecture
The model is a hybrid CNN-LSTM architecture:
*   **Base CNN:** A pre-trained `MobileNetV2` model (without the top classification layer) is used as a feature extractor. It processes each frame independently using a `TimeDistributed` layer.
*   **Temporal Layer:** An `LSTM` layer processes the sequence of features extracted by the CNN, capturing temporal patterns.
*   **Dense Layers:** Fully connected layers with ReLU activation and Dropout for regularization.
*   **Output Layer:** A single `Dense` layer with sigmoid activation for binary classification (normal vs. shoplifting).

**Compilation Details:**
*   **Optimizer:** Adam with a learning rate of 1e-4.
*   **Loss Function:** Binary Crossentropy.
*   **Metrics:** Accuracy, Precision, and Recall.

## 5. Training and Evaluation
The model is trained for 15 epochs with a batch size of 4. A validation split of 20% is used during training.

**Training History (`history_transfer` details):**
*   Epochs: 15
*   Batch Size: 4
*   Metrics monitored: Accuracy, Loss, Precision, Recall, and their validation counterparts.

After training, the model's performance is evaluated on the test set.

## 6. Results and Analysis
### Evaluation Metrics on Test Set:
*   **Test Loss:** 0.4148
*   **Test Accuracy:** 0.8378
*   **Test Precision:** 0.8421
*   **Test Recall:** 0.8421

### Confusion Matrix:
```
[[15  3]
 [ 3 16]]
```
*   **True Negatives (Normal correctly classified):** 15
*   **False Positives (Normal misclassified as Shoplifting):** 3
*   **False Negatives (Shoplifting misclassified as Normal):** 3
*   **True Positives (Shoplifting correctly classified):** 16

### Classification Report:
```
              precision    recall  f1-score   support

      normal       0.83      0.83      0.83        18
 shoplifting       0.84      0.84      0.84        19

    accuracy                           0.84        37
   macro avg       0.84      0.84      0.84        37
weighted avg       0.84      0.84      0.84        37
```

### Analysis of Wrong Predictions:
The notebook further identifies and visualizes the specific video segments that were incorrectly classified, showing the true label, predicted label, and the model's probability for shoplifting for each error case. This helps in understanding the model's failure modes and potential areas for improvement.
