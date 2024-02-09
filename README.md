# Neural Network from Scratch in Python

This repository contains a modular implementation of a neural network from scratch in Python, capable of handling multi-classification, binary classification, and regression problems. The neural network architecture consists of customizable layers, activation functions, loss functions, optimizers, and accuracy metrics.

## Code Structure

- **Accuracy_Categorical.py**: Defines the accuracy metric for multi-classification problems.
- **Accuracy_Regression.py**: Defines the accuracy metric for regression problems.
- **Activation_Linear.py**: Implements the linear activation function.
- **Activation_ReLU.py**: Implements the Rectified Linear Unit (ReLU) activation function.
- **Activation_Softmax.py**: Implements the softmax activation function.
- **Layer_Dense.py**: Defines the dense (fully connected) layer of the neural network.
- **Layer_Dropout.py**: Implements dropout regularization to prevent overfitting.
- **Loss_CategoricalCrossentropy.py**: Defines the cross-entropy loss function for classification problems.
- **Loss_MeanSquaredError.py**: Defines the mean squared error loss function.
- **Model.py**: Contains the Model class, which orchestrates the neural network's layers and operations.
- **Optimizer_Adam.py**: Implements the Adam optimizer for gradient descent optimization.

## Usage

### Multi-Classification Example

```python
from nnfs.datasets import spiral_data
from Accuracy_Categorical import Accuracy_Categorical
from Activation_ReLU import Activation_ReLU
from Activation_Softmax import Activation_Softmax
from Layer_Dense import Layer_Dense
from Layer_Dropout import Layer_Dropout
from Loss_CategoricalCrossentropy import Loss_CategoricalCrossentropy
from Model import Model
from Optimizer_Adam import Optimizer_Adam

# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

# Set loss, optimizer, and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical(),
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
```

### Binary-Classification Example

```python
from nnfs.datasets import sine_data
from Accuracy_Regression import Accuracy_Regression
from Activation_Linear import Activation_Linear
from Activation_ReLU import Activation_ReLU
from Layer_Dense import Layer_Dense
from Loss_MeanSquaredError import Loss_MeanSquaredError
from Model import Model
from Optimizer_Adam import Optimizer_Adam

# Create dataset
X, y = sine_data()
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

# Set loss, optimizer, and accuracy objects
model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Regression(),
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=100, print_every=100)
```

### Regression Example

```python
from nnfs.datasets import sine_data
from Accuracy_Regression import Accuracy_Regression
from Activation_Linear import Activation_Linear
from Activation_ReLU import Activation_ReLU
from Layer_Dense import Layer_Dense
from Loss_MeanSquaredError import Loss_MeanSquaredError
from Model import Model
from Optimizer_Adam import Optimizer_Adam

# Create dataset
X, y = sine_data()

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

# Set loss, optimizer, and accuracy objects
model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
    accuracy=Accuracy_Regression(),
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, epochs=10000, print_every=100)
```



