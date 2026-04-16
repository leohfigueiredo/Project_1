# Project_1 - TensorFlow Machine Learning Projects

This repository contains a collection of Machine Learning and Deep Learning projects and notebooks using TensorFlow.

## 📋 Overview

This project brings together practical implementations of various ML/DL concepts, including:

- **Artificial Neural Networks (ANN)**
- **Convolutional Neural Networks (CNN)**
- **Natural Language Processing (NLP)**
- **Transfer Learning**
- **TinyML**

## 📁 Project Structure

```
Project_1/
├── Tensorflow/              # Main TensorFlow notebooks and scripts
│   ├── 00_Tensorflow_fundamentals.ipynb
│   ├── 01_neural_network_regression_with_tensorflow.ipynb
│   ├── 02_neural_network_classification_with_tensorflow.ipynb
│   ├── 03-Introduction-to-computer-vision-with-tensorflow.ipynb
│   ├── CNN_Cifar_10.ipynb
│   └── TinyML/              # TinyML projects
├── Tensorflow-1/            # Additional notebooks and scripts
│   ├── Files&notebooks/     # Notebooks organized by topic
│   │   └── 08_introduction_to_nlp_in_tensorflow/
│   └── TinyML/
├── helper_functions.py      # Utility functions
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip

### Installing Dependencies

```bash
pip install -r requirements.txt
```

## 📚 Available Notebooks

### TensorFlow Fundamentals
- `00_Tensorflow_fundamentals.ipynb` - Introduction to basic TensorFlow concepts

### Regression and Classification
- `01_neural_network_regression_with_tensorflow.ipynb` - Regression with neural networks
- `02_neural_network_classification_with_tensorflow.ipynb` - Classification with neural networks

### Computer Vision
- `03-Introduction-to-computer-vision-with-tensorflow.ipynb` - Introduction to CV
- `CNN_Cifar_10.ipynb` - CNN for CIFAR-10 image classification

### Natural Language Processing (NLP)
- `08_introduction_to_nlp_in_tensorflow.ipynb` - Introduction to NLP with TensorFlow

## 🛠️ Technologies Used

- **TensorFlow** - Main Deep Learning framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - ML metrics and utilities

## 📊 Datasets Used

- **CIFAR-10** - Image classification
- **NLP Getting Started** - Tweet classification (natural disasters)
- **Food-101** - Food image classification

## 🔧 Environment Setup

### Performance Optimizations (GEKKO A9 MAX)

The project includes optimized configurations for Ryzen 9 processor:

```python
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '2',
    'TF_ENABLE_ONEDNN_OPTS': '1',  # oneDNN (2-3x faster)
    'OMP_NUM_THREADS': '32',
    'MKL_NUM_THREADS': '32',
    'TF_NUM_INTEROP_THREADS': '4',
    'TF_NUM_INTRAOP_THREADS': '32',
    'TF_XLA_FLAGS': '--tf_xla_auto_jit=2',  # XLA JIT
    'CUDA_VISIBLE_DEVICES': '-1',  # CPU only
})
```

## 📝 How to Use

1. Clone the repository:
```bash
git clone https://github.com/leohfigueiredo/Project_1.git
cd Project_1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebooks in Jupyter or VS Code

## 🤝 Contributing

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is under the MIT license.

## 👤 Author

- **Leonardo Figueiredo** - [leohfigueiredo](https://github.com/leohfigueiredo)

## 🔗 Useful Links

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [TensorFlow Hub](https://tfhub.dev/)