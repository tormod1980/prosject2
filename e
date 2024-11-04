import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000, lambda_reg=0.0):
        # Initialize the logistic regression model with learning rate, number of epochs, and regularization parameter
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.lambda_reg = lambda_reg  # Regularization term to prevent overfitting
        self.theta = None  # Model parameters (weights), initialized during training

    def sigmoid(self, z):
        # Sigmoid activation function to output probabilities
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Fit the logistic regression model to the training data
        m, n = X.shape  # m = number of samples, n = number of features
        self.theta = np.zeros(n)  # Initialize weights to zero

        # Gradient descent optimization
        for epoch in range(self.num_epochs):
            for i in range(m):
                # Calculate the linear combination of features and weights
                z = np.dot(X[i], self.theta)
                h = self.sigmoid(z)  # Apply sigmoid function to get the predicted probability

                # Compute the gradient with regularization term
                gradient = (h - y[i]) * X[i] + (self.lambda_reg / m) * self.theta

                # Update weights using the computed gradient
                self.theta -= self.learning_rate * gradient

            # Print the loss every 100 epochs for monitoring
            if epoch % 100 == 0:
                loss = self.compute_loss(X, y)
                print(f"Epoch {epoch}: Loss = {loss}")

    def predict(self, X):
        # Predict class labels (0 or 1) based on input features X
        z = np.dot(X, self.theta)
        return (self.sigmoid(z) >= 0.5).astype(int)  # Threshold at 0.5 for binary classification

    def compute_loss(self, X, y):
        # Compute the binary cross-entropy loss with L2 regularization
        m = len(y)
        h = self.sigmoid(np.dot(X, self.theta))
        return -1/m * (np.dot(y, np.log(h + 1e-15)) + np.dot(1 - y, np.log(1 - h + 1e-15))) + \
               (self.lambda_reg / (2 * m)) * np.sum(self.theta**2)

    def accuracy(self, y_true, y_pred):
        # Calculate accuracy as the proportion of correct predictions
        return np.mean(y_true == y_pred)

# Using the previously defined code to load and prepare the dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels (0 or 1)

# Split the data into training and test sets with 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for improved model convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train logistic regression model
log_reg = LogisticRegression(learning_rate=0.01, num_epochs=1000, lambda_reg=0.1)
log_reg.fit(X_train, y_train)

# Make predictions on training and test sets
y_pred_train = log_reg.predict(X_train)
y_pred_test = log_reg.predict(X_test)

# Calculate and print accuracy on both training and test sets
train_accuracy = log_reg.accuracy(y_train, y_pred_train)
test_accuracy = log_reg.accuracy(y_test, y_pred_test)

print(f"Logistic Regression Train Accuracy: {train_accuracy:.2f}")
print(f"Logistic Regression Test Accuracy: {test_accuracy:.2f}")
