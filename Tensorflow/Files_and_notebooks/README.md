# TensorFlow-1 - Files & Notebooks

This folder contains an organized collection of Jupyter notebooks covering various machine learning and deep learning topics using TensorFlow. The notebooks are organized in a logical learning sequence, from fundamentals to more advanced projects.

## 📁 Folder Structure and Notebooks

### 00_fundamentals/
**Notebook:** `00_Tensorflow_fundamentals.ipynb`

This notebook covers the most fundamental concepts of tensors using TensorFlow:
- Introduction to tensors (scalars, vectors, matrices, tensors)
- Getting information from tensors (shape, rank, axis, size)
- Manipulating tensors
- Relationship between tensors and NumPy
- Using `@tf.function` to speed up Python functions
- Using GPUs with TensorFlow (or TPUs)
- Practical exercises

---

### 01_neural_network_regression/
**Notebook:** `01_neural_network_regression_with_tensorflow_video - Copy.ipynb`

Introduction to regression problems with neural networks in TensorFlow:
- What is a regression problem (predicting a numerical value)
- Creating and visualizing data
- Steps for modeling with TensorFlow:
  1. Create a model
  2. Compile the model
  3. Fit the model
- How to improve a model (more layers, more epochs, more data)
- Model evaluation
- Train/validation/test set splitting
- Visualization of data, models, and predictions

---

### 02_neural_network_classification/
**Notebook:** `02_neural_network_classification_with_tensorflow.ipynb`

Introduction to classification problems with neural networks in TensorFlow:
- What is classification (classifying something as one thing or another)
- Types of classification problems:
  - Binary classification
  - Multiclass classification
  - Multilabel classification
- Creating and visualizing classification data
- Building neural networks for classification
- Activation functions for classification (ReLU, sigmoid)
- Loss functions for classification (binary crossentropy, categorical crossentropy)

---

### 03_Intro_to_computer_vision/
**Notebook:** `03-Introduction-to-computer-vision-with-tensorflow.ipynb`

Introduction to computer vision with TensorFlow:
- Basic computer vision concepts
- Image processing with TensorFlow
- Convolutional Neural Networks (CNNs) for image classification
- Image dataset (food classification: pizza, steak, sushi, hamburger)

---

### 04_transfer_learning/
**Notebook:** `04-transfer-learning-in-tensorflow-part-1-feature-extraction.ipynb`

Part 1: Transfer Learning - Feature Extraction:
- What is transfer learning
- How to use pre-trained models for feature extraction
- Advantages of transfer learning
- Implementing feature extraction with models like EfficientNet, ResNet, etc.

---

### 05_transfer_learning_fine_tuning/
**Notebook:** `05_transfer_learning_in_tensorflow_part_2_fine_tuning.ipynb`

Part 2: Transfer Learning - Fine Tuning:
- What is fine tuning
- How to adjust layers of pre-trained models
- Difference between feature extraction and fine tuning
- Fine tuning strategies for models

---

### 06_Transfer_learning_scaling_up/
**Notebook:** `06_transfer_learning_in_tensorflow_part_3_scaling_up.ipynb`

Part 3: Transfer Learning - Scaling Up:
- How to scale transfer learning models
- Handling larger datasets
- Performance optimization
- Saved model (`101_food_classes_10_percent_saved_big_dog_model/`)

---

### 07_Project_1_Food_Vision_Big/
**Notebook:** `Project_1_Food_Vision_Big.ipynb`

Milestone Project: Food Vision Big™ - A complete food image classification project:
- Classification of 101 food classes
- Performance benchmark implementation (BF16 vs FP32)
- Hardware optimization for specific hardware (Geekom A9 Max - Ryzen 9)
- Advanced deep learning techniques for computer vision

---

### 08_introduction_to_nlp_in_tensorflow/
**Notebook:** `08_introduction_to_nlp_in_tensorflow.ipynb`

Introduction to Natural Language Processing (NLP) with TensorFlow:
- NLP fundamentals (Natural Language Processing)
- Sequence-to-sequence problems (seq2seq)
- Tweet dataset for classification (disaster vs non-disaster)
- Text preprocessing techniques
- NLP models with TensorFlow
- Optimized configuration for Ryzen 9 hardware (32 threads, 96GB RAM)

---

## 🛠️ Support Files

- `helper_functions.py` - Utility functions used across various notebooks
- `oracle.json`, `tuner0.json` - Keras Tuner configurations for hyperparameter tuning
- `trial_*/` - Keras Tuner trial results

## 📝 Requirements

To run these notebooks, you will need:
- Python 3.7+
- TensorFlow 2.x
- Jupyter Notebook/JupyterLab
- Additional libraries: numpy, pandas, matplotlib, scikit-learn

## 🚀 How to Use

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Navigate to the desired folder
4. Open the notebook in Jupyter: `jupyter notebook notebook_name.ipynb`

## 📚 Recommended Study Order

1. `00_fundamentals/` - Start here to understand TensorFlow fundamentals
2. `01_neural_network_regression/` - Learn about regression
3. `02_neural_network_classification/` - Learn about classification
4. `03_Intro_to_computer_vision/` - Introduction to computer vision
5. `04_transfer_learning/` to `06_Transfer_learning_scaling_up/` - Transfer learning series
6. `07_Project_1_Food_Vision_Big/` - Practical computer vision project
7. `08_introduction_to_nlp_in_tensorflow/` - Introduction to NLP