# Simple CNN for MNIST Classification

This project contains a simple convolutional neural network (CNN) implementation using PyTorch for classifying MNIST
digits. It includes code for training the model, testing its performance on the MNIST test dataset, and using the
trained model to predict the class of custom handwritten digit images.

## Installation

To run this project, ensure you have Python 3.6+ and PyTorch installed. You can install PyTorch by following the
instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

Additionally, you will need to install the Pillow library for image processing:

```bash
pip install pillow
```

## Usage

To train and test the model on the MNIST dataset and then use it for predicting custom images, simply run the script
from the command line:

```bash
python mnist_cnn.py
```

### Training and Testing

The script will automatically download the MNIST dataset, train the SimpleNet model for a predefined number of epochs,
and then evaluate the model's performance on the test set.

### Predicting Custom Images

To use your own handwritten digit images for prediction, place your images in a directory and modify
the `predict_with_custom_image` function call in the `main` block with the path to your image file. Ensure your images
are grayscale, and have a resolution close to the 28x28 pixels used by the MNIST dataset for optimal performance.

Click the following to create and test your own images: [Whiteboard App Guide](WHITEBOARD.md)

Example:

```python
# Replace 'file_path.png' with the path to your custom image
predict_with_custom_image(model, device, 'file_path.png')
```

## Features

- Simple CNN architecture with two convolutional layers.
- Training and testing procedures on the MNIST dataset.
- Custom image prediction using a trained model.

## Model Architecture

The model defined in `SimpleNet` consists of:

- Two convolutional layers for feature extraction.
- Two fully connected layers for classification.
- ReLU activations and max-pooling.


