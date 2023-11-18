# Transfer Learning: Image Classifier with MobileNetV2

This project implements a deep learning-based image classifier to distinguish between alpaca and non-alpaca images. The classifier uses transfer learning with MobileNetV2, a pre-trained Convolutional Neural Network (CNN) trained on the ImageNet dataset. The goal is to create an efficient and accurate model for alpaca recognition.

## Table of Contents
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Fine-tuning](#fine-tuning)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Project Objectives

### 1. Create Dataset:
   - Establish a dataset from a directory containing alpaca and non-alpaca images.
   - Implement data splitting into training and validation sets.

### 2. Preprocess and Augment Data:
   - Apply data augmentation techniques using the Sequential API to enhance diversity in the training set.
   - Utilize prefetching to prevent memory bottlenecks during data preprocessing.

### 3. Transfer Learning with MobileNetV2:
   - Explore the architecture of MobileNetV2, focusing on depthwise separable convolutions and bottleneck layers.
   - Delete the top layer of MobileNetV2, which contained ImageNet classification labels.
   - Integrate a new classifier layer for binary classification (alpaca or not alpaca).

### 4. Model Training:
   - Develop a custom model using the Functional API and MobileNetV2 base.
   - Fine-tune the model's final layers to improve accuracy.
   - Compile and train the model using Binary Crossentropy loss and Adam optimizer.

### 5. Fine-tuning:
   - Unfreeze selected layers of MobileNetV2 for fine-tuning.
   - Re-run the optimizer with a smaller learning rate to adapt the model to new data.
   - Explore the impact of fine-tuning on model accuracy.

## Key Features

- Efficient alpaca recognition using MobileNetV2.
- Transfer learning for leveraging pretrained model knowledge.
- Data augmentation for enhancing training set diversity.
- Fine-tuning for adapting the model to specific data.

## Dataset
The dataset consists of alpaca and non-alpaca images stored in the dataset/ directory. It is split into training and validation sets using the image_dataset_from_directory function.

## Model Architecture
The model architecture is based on MobileNetV2 with a customized classifier for binary classification. It utilizes depthwise separable convolutions and bottleneck layers to achieve efficient feature extraction.

## Training
The model is trained using the train_dataset with data augmentation and preprocessing. The training process involves fine-tuning the final layers for improved accuracy.

## Fine-tuning
Fine-tuning involves unfreezing selected layers of MobileNetV2 and re-running the optimizer with a smaller learning rate. This step adapts the model to new data and captures high-level details.

## Results
The trained model achieves accurate alpaca recognition with improved performance through fine-tuning.

## Dependencies
- TensorFlow
- Matplotlib
- NumPy

## Contributing
Feel free to contribute to this project by opening issues or submitting pull requests. Your feedback and contributions are highly appreciated.


