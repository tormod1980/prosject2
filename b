import numpy as np

class FeedForwardNN:
    def __init__(self, layer_sizes):
        """
        Initialize the neural network with specified layer sizes.
        
        Parameters:
        layer_sizes (list): List of integers representing the number of neurons in each layer.
        """
        self.layer_sizes = layer_sizes
        self.weights = []  # List to store weights for each layer
        self.biases = []   # List to store biases for each layer
        self.initialize_weights_biases()  # Initialize weights and biases

    def initialize_weights_biases(self):
        """
        Initialize weights and biases with random values for each layer in the network.
        Weights are initialized with small random values from a normal distribution.
        Biases are initialized to zero.
        """
        for i in range(len(self.layer_sizes) - 1):
            # Initialize weights for the connection between layers i and i+1
            weight = np.random.normal(0, 0.1, (self.layer_sizes[i], self.layer_sizes[i + 1]))
            # Initialize biases for layer i+1
            bias = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, z):
        """
        Sigmoid activation function.
        
        Parameters:
        z (np.ndarray): Input array to apply the sigmoid function.
        
        Returns:
        np.ndarray: Output after applying sigmoid element-wise.
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """
        Derivative of the sigmoid function, assuming `z` is the sigmoid-activated output.
        
        Parameters:
        z (np.ndarray): Sigmoid-activated output.
        
        Returns:
        np.ndarray: Derivative of the sigmoid function applied to z.
        """
        return z * (1 - z)

    def forward(self, X):
        """
        Perform forward propagation through the network.
        
        Parameters:
        X (np.ndarray): Input data.
        
        Returns:
        np.ndarray: Output of the network after forward propagation.
        """
        self.a = [X]  # Store activations for each layer, starting with the input layer
        for weight, bias in zip(self.weights, self.biases):
            # Compute linear transformation and apply sigmoid activation
            z = self.a[-1] @ weight + bias
            a = self.sigmoid(z)
            self.a.append(a)
        return self.a[-1]  # Return the output of the final layer

    def compute_loss(self, y_true, y_pred):
        """
        Compute the Mean Squared Error (MSE) loss.
        
        Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted values from the network.
        
        Returns:
        float: Mean squared error loss.
        """
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y):
        """
        Perform backward propagation to compute gradients for weights and biases.
        
        Parameters:
        X (np.ndarray): Input data.
        y (np.ndarray): True target values.
        
        Returns:
        list: Gradients for weights.
        list: Gradients for biases.
        """
        m = y.shape[0]  # Number of samples
        delta = self.a[-1] - y  # Compute the error for the output layer
        grads_w = []
        grads_b = []

        # Calculate gradients for the output layer
        grads_w.append(self.a[-2].T @ delta / m)
        grads_b.append(np.sum(delta, axis=0, keepdims=True) / m)

        # Backpropagate through hidden layers
        for i in range(len(self.layer_sizes) - 2, 0, -1):
            # Calculate delta for the current layer
            delta = (delta @ self.weights[i].T) * self.sigmoid_derivative(self.a[i])
            grads_w.append(self.a[i - 1].T @ delta / m)
            grads_b.append(np.sum(delta, axis=0, keepdims=True) / m)

        # Reverse the gradients to match layer order
        grads_w.reverse()
        grads_b.reverse()

        return grads_w, grads_b

    def update_weights_biases(self, grads_w, grads_b, learning_rate):
        """
        Update the weights and biases using the computed gradients.
        
        Parameters:
        grads_w (list): Gradients for weights.
        grads_b (list): Gradients for biases.
        learning_rate (float): Learning rate for gradient descent.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

    def train(self, X, y, learning_rate, num_epochs):
        """
        Train the neural network using the provided data.
        
        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.
        learning_rate (float): Learning rate for optimization.
        num_epochs (int): Number of training epochs.
        """
        for epoch in range(num_epochs):
            # Forward pass
            y_pred = self.forward(X)
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            # Backward pass to compute gradients
            grads_w, grads_b = self.backward(X, y)
            # Update weights and biases
            self.update_weights_biases(grads_w, grads_b, learning_rate)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")

# Example usage with synthetic data
def f(x, a, b, c):
    return a * x ** 2 + b * x + c

X = np.random.rand(100, 1)  # 100 samples, 1 feature
y = f(X.flatten(), 1, 2, 3) + np.random.normal(0, 0.1, 100)  # Quadratic with noise

# Reshape for training
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Create and train the network
layer_sizes = [1, 10, 10, 1]  # Input layer, two hidden layers, output layer
nn = FeedForwardNN(layer_sizes)
nn.train(X, y, learning_rate=0.01, num_epochs=1000)
