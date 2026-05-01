# TensorFlow - Main Directory

This folder contains a collection of Jupyter notebooks and Python scripts focused on TensorFlow fundamentals and practical implementations of neural networks.

## 📁 Notebooks

### Core TensorFlow Fundamentals

#### 00_Tensorflow_fundamentals.ipynb
Introduction to the fundamental concepts of tensors using TensorFlow:
- Introduction to tensors (scalars, vectors, matrices, tensors)
- Getting information from tensors (shape, rank, axis, size)
- Manipulating tensors
- Tensors & NumPy relationship
- Using `@tf.function` for performance optimization
- Using GPUs with TensorFlow (or TPUs)
- Practice exercises

---

### Neural Network Basics

#### TF_First_Neural_Network.ipynb
Your first neural network with TensorFlow - a beginner-friendly introduction to building and training neural networks.

#### TF_3_Layers.ipynb
Building a neural network with 3 layers - exploring deeper network architectures and their capabilities.

---

### Regression Problems

#### TF_Boston_Housing_Regression.ipynb
Predicting housing prices using neural networks with the Boston Housing dataset:
- Data preprocessing for regression
- Building regression models
- Evaluating regression performance

#### 01_neural_network_regression_with_tensorflow_video - Copy.ipynb
Introduction to regression problems with neural networks in TensorFlow.

---

### Classification Problems

#### TF_MNIST_Classification.ipynb
Handwritten digit classification using the MNIST dataset:
- Loading and preprocessing MNIST data
- Building classification neural networks
- Evaluating classification accuracy

#### 02_neural_network_classification_with_tensorflow.ipynb
Introduction to classification problems with neural networks in TensorFlow.

---

### Computer Vision

#### 03-Introduction-to-computer-vision-with-tensorflow.ipynb
Introduction to computer vision concepts and implementations with TensorFlow.

#### CNN_Cifar_10.ipynb
Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset:
- CNN architecture design
- Image classification with CNNs
- Model training and evaluation

---

### Testing & Environment

#### test.ipynb
A test notebook for experimenting with TensorFlow code and testing environment setup.

#### tf_startup.py
Python script for TensorFlow environment initialization and configuration.

#### test_env_completo.py
Comprehensive environment testing script to verify all dependencies and configurations.

---

### TinyML Subdirectory

#### TinyML/
Contains notebooks focused on Tiny Machine Learning:
- `Exploring_Loss_Cost_Function.ipynb` - Understanding loss functions
- `IESTI01_data_augmentation.ipynb` - Data augmentation techniques

---

### untitled_project/
Contains Keras Tuner configuration files:
- `oracle.json` - Tuner oracle configuration
- `tuner0.json` - Tuner state

---

## 🛠️ Support Files

- `.gitignore` - Git ignore configuration for this directory
- `tf_startup.py` - TensorFlow startup/initialization script
- `test_env_completo.py` - Complete environment testing script

## 📝 Requirements

To run these notebooks, you will need:
- Python 3.7+
- TensorFlow 2.x
- Jupyter Notebook/JupyterLab
- Additional libraries: numpy, pandas, matplotlib, scikit-learn

## 🚀 How to Use

1. Navigate to this folder
2. Install dependencies: `pip install -r ../requirements.txt` (from project root)
3. Open any notebook in Jupyter: `jupyter notebook notebook_name.ipynb`
4. Run cells sequentially to execute the code

## 📚 Recommended Study Order

1. `00_Tensorflow_fundamentals.ipynb` - Start with TensorFlow basics
2. `TF_First_Neural_Network.ipynb` - Build your first neural network
3. `TF_3_Layers.ipynb` - Explore deeper networks
4. `TF_Boston_Housing_Regression.ipynb` - Learn regression
5. `TF_MNIST_Classification.ipynb` - Learn classification
6. `03-Introduction-to-computer-vision-with-tensorflow.ipynb` - Computer vision basics
7. `CNN_Cifar_10.ipynb` - Advanced computer vision with CNNs

## 🔗 Related Resources

### Files_and_notebooks/ Subdirectory

This subdirectory contains organized course materials with more advanced topics:

#### 00_fundamentals/
- `00_Tensorflow_fundamentals.ipynb` - TensorFlow basics

#### 01_neural_network_regression/
- `01_neural_network_regression_with_tensorflow_video - Copy.ipynb` - Regression with neural networks

#### 02_neural_network_classification/
- `02_neural_network_classification_with_tensorflow.ipynb` - Classification with neural networks

#### 03_Intro_to_computer_vision/
- `03-Introduction-to-computer-vision-with-tensorflow.ipynb` - Computer vision introduction

#### 04_transfer_learning/
- `04-transfer-learning-in-tensorflow-part-1-feature-extraction.ipynb` - Transfer learning basics

#### 05_transfer_learning_fine_tuning/
- `05_transfer_learning_in_tensorflow_part_2_fine_tuning.ipynb` - Fine-tuning techniques

#### 06_Transfer_learning_scaling_up/
- `06_transfer_learning_in_tensorflow_part_3_scaling_up.ipynb` - Scaling up transfer learning

#### 07_Project_1_Food_Vision_Big/
- `Project_1_Food_Vision_Big.ipynb` - Complete food classification project

#### 08_introduction_to_nlp_in_tensorflow/
- `08_introduction_to_nlp_in_tensorflow.ipynb` - Natural Language Processing with TensorFlow

### pizza_steak/ Subdirectory

Contains data and resources for binary image classification (pizza vs steak).

---

**Note:** The project has been consolidated into this single `Tensorflow/` directory. The previous `Tensorflow-1/` folder has been removed to avoid duplication and save GitHub storage space.
