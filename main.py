import numpy as np
import matplotlib.pyplot as plt


# Activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# wXOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

# Parameters
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 100000

# Random weights
weights_hidden_input = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
# Lists to save the data
mse_history = []
classification_errors = []
input_hidden_weights_history = []
output_hidden_weights_history = []

# Training
for _ in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_hidden_input)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    # Error and backward propagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Weights update
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_hidden_input += X.T.dot(d_hidden_layer) * learning_rate

    # MSE saving
    mse = np.mean(np.square(error))
    mse_history.append(mse)

    predictions = (predicted_output > 0.5).astype(int)
    classification_error = np.mean(np.abs(predictions - expected_output))
    classification_errors.append(classification_error)

    input_hidden_weights_history.append(weights_hidden_input.copy())
    output_hidden_weights_history.append(weights_hidden_output.copy())

plt.figure(figsize=(8, 6))
plt.plot(mse_history)
plt.title('MSE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()
plt.savefig('MSE.png')

plt.figure(figsize=(8, 6))
plt.plot(classification_errors)
plt.title('Classification Error Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Classification Error')
plt.show()
plt.savefig('ClassificationError.png')

plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
input_hidden_weights_history = np.array(input_hidden_weights_history)
for i in range(input_hidden_weights_history.shape[2]):
    plt.plot(input_hidden_weights_history[:, :, i])
plt.title('Input-Hidden Weights Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Weights')

plt.subplot(2, 1, 2)
output_hidden_weights_history = np.array(output_hidden_weights_history)
for i in range(output_hidden_weights_history.shape[2]):
    plt.plot(output_hidden_weights_history[:, :, i])
plt.title('Hidden-Output Weights Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Weights')

plt.tight_layout()
plt.show()
plt.savefig('Hidden-OutputWeights.png')
