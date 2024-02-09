# Neural Network from Scratch in Python

This repository contains a modular implementation of a neural network from scratch in Python, capable of handling multi-classification, binary classification, and regression problems. The neural network architecture consists of customizable layers, activation functions, loss functions, optimizers, and accuracy metrics.

## Code Structure

- **Accuracy_Regression.py**: Defines the accuracy metric for regression problems.
- **Activation_Linear.py**: Implements the linear activation function.
- **Activation_ReLU.py**: Implements the Rectified Linear Unit (ReLU) activation function.
- **Layer_Dense.py**: Defines the dense (fully connected) layer of the neural network.
- **Layer_Dropout.py**: Implements dropout regularization to prevent overfitting.
- **Loss_MeanSquaredError.py**: Defines the mean squared error loss function.
- **Model.py**: Contains the Model class, which orchestrates the neural network's layers and operations.
- **Optimizer_Adam.py**: Implements the Adam optimizer for gradient descent optimization.

## Usage

1. **Dataset Preparation**:
   - Utilize the provided datasets or integrate your own dataset. The example uses the `sine_data()` function from the `nnfs.datasets` module.

2. **Model Instantiation**:
   - Instantiate the Model class using `model = Model()`.

3. **Add Layers**:
   - Add layers to the model using `model.add()` method. Choose from `Layer_Dense` and `Layer_Dropout`.
   - Example: `model.add(Layer_Dense(input_size, output_size))`.

4. **Set Loss, Optimizer, and Accuracy**:
   - Configure the loss function, optimizer, and accuracy metric using `model.set()`.
   - Example: 
     ```python
     model.set(
         loss=Loss_MeanSquaredError(),
         optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
         accuracy=Accuracy_Regression(),
     )
     ```

5. **Finalize Model**:
   - Finalize the model setup using `model.finalize()`.

6. **Training**:
   - Train the model on the dataset using `model.train()` method.
   - Example: `model.train(X_train, y_train, validation_data=(X_test, y_test), epochs=100, print_every=100)`.

## Dependencies
- This code uses the `nnfs` library for dataset generation. Install it using `pip install nnfs`.

## Example
```python
from nnfs.datasets import sine_data
from Accuracy_Regression import Accuracy_Regression
from Activation_Linear import Activation_Linear
from Activation_ReLU import Activation_ReLU
from Layer_Dense import Layer_Dense
from Layer_Dropout import Layer_Dropout
from Loss_MeanSquaredError import Loss_MeanSquaredError
from Model import Model
from Optimizer_Adam import Optimizer_Adam

# Create dataset
X_train, y_train = sine_data()
X_test, y_test = sine_data()

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Regression(),
)

# Finalize the model
model.finalize()
