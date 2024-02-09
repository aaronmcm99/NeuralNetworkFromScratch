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

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Regression(),
)
# Finalize the model
model.finalize()
# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=100, print_every=100)
