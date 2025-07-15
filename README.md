# Cat vs Dog Classifier 🐱🐶

A simple image classification project using Convolutional Neural Networks (CNN) to distinguish between cats and dogs.

## 📂 Dataset

Download the dataset from [Kaggle - Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats) and place it in:

```
/kaggle/input/dogs-vs-cats/
├── train/
├── test/
```

## 🧠 Model

- CNN with Conv2D, MaxPooling2D, BatchNormalization, Dropout
- Trained for 5 epochs
- Input image size: 256x256

## 🚀 Getting Started

1. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

2. **Train the model**

   ```
   python train.py
   ```

3. **Predict a new image**
   ```
   python predict.py path/to/image.jpg
   ```

## 📁 Structure

```
cat-dog-classifier-cnn/
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── model/
    └── model.h5 (generated after training)
```

## ✅ Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib
