import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize the Neural Network

        Parameters:
        -----------
        layer_size: list
            List of layer sizes [input_size, hidden1_size, ..., output_size]
            Examples for MNIST: [784, 128, 64, 10]
        learning_rate: float
            Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)

        #init weight and bias
        self.weights = []
        self.biases = []

        #He initialization for weights
        for i in range(len(layer_sizes) - 1):
            # FIXED: Added He initialization multiplier
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            #bias vector
            b = np.zeros((1, layer_sizes[i+1]))

            self.weights.append(w)
            self.biases.append(b)
        
        # FIXED: Typo and added function call
        print(f"Initialization of network with architecture: {layer_sizes}")
        print(f"Number of Parameters: {self._count_parameters()}")

    # FIXED: Renamed to _count_parameters (private method)
    def _count_parameters(self):
        """Count total numbers of trainable parameters"""
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total
    
    def relu(self, Z):
        """
        ReLU activation function:

        Parameters:
        -----------
        Z: array-like
            Input values (pre-activation)

        Returns:
        -----------
        Activated values (max(0, Z))
        """
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
         """
        Derivative of ReLu function

        Parameters:
        -----------
        Z: array-like
            Input values  (pre-activation)

        Returns:
        -----------
        Gradient: 1 if Z > 0, else 0
        """
         return (Z > 0).astype(float)
    
    def softmax(self, Z):
        """
        Softmax activation function (for output layer)
        Convert logits to probalities

        Parameters:
        -----------
        Z: array-like, shape(n_samples, n_classes)
            Input logits

        Returns:
        -----------
        Probalities that sum to 1 for each sample
        """

        #subtract max for numerical stabiliuty (will prevent overflow [why does this even exist :cry:])
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss

        Parameters:
        -----------
        y_true: array-like, shape(n_samples, n_classes)
            True labels (one-hot encoded)
        y_pred: array-like, shape (n_samples, n_classes)
            Predicts probalities

        Returns:
        -----------
        loss: float
            Cross-entropy loss value
        """
        #add tiny epsilon to not log(0)
        epsilon = 1e-8
        # FIXED: Added [0] to get scalar value
        n_samples = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / n_samples
        return loss
    
    def forward(self, X):
        """
        Forward propagation through nn

        Parameters:
        -----------
        X: array-like, shape (n_samples, n_classes)
            Input data
        
        Returns:
        -----------
        output: array-like, shape (n_samples, n_classes)
            Network Prediction (probalities)
        """
        #store activation for back prop
        self.activations = [X]
        self.z_values = []

        #pass through every layer
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            #apply activation function
            if i == len(self.weights) - 1:
                a = self.softmax(z)
            else:
                #hidden layer use relu
                a = self.relu(z)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def predict(self, X):
        """
        Make predictions and return class labels

        Parameters:
        -----------
        X: array-like, shape (n_samples, n_features)
            Input data

        Returns:
        -----------
        prediction: array-like, shape(n_samples, )
            Predicted class labels (0 - 9 for MNIST)
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
    
    def accuracy(self, X, y_true):
        """
        Calculate accuracy on given data

        Parameters:
        -----------
        X: array-like, shape (n_samples, n_features)
            Input data
        y_true: array-like, shape (n_samples, n_classes)
            True labels (one-hot encoded)
        
        Returns:
        -----------
        accuracy : float
            Accuracy percentage (0-1)
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)
    
    def backward(self, y_true):
        """
        Backward propagation - compute gradients

        Parameters:
        -----------
        y_true: array-like, shape (n_samples, n_classes)
            True labels (one-hot encoded)
        """
        m = y_true.shape[0]

        #init gradient storage
        self.weight_gradients = []
        self.bias_gradients = []

        #output layer gradient (softmax + cross-entropy deravative)
        dZ = self.activations[-1] - y_true

        #backpropagate through each laywr
        for i in range(len(self.weights) - 1, -1, -1):
            #compute weight and bias gradients
            dW = np.dot(self.activations[i].T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)

            #Store gradients (insert at begin to keep order)
            self.weight_gradients.insert(0, dW)
            self.bias_gradients.insert(0, db)

            #compute gradient for prev layer (!= input layer)
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.relu_derivative(self.z_values[i-1])

    def update_parameters(self):
        """
        Update weights and biases using computed gradients
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.weight_gradients[i]
            self.biases[i] -= self.learning_rate * self.bias_gradients[i]
    
    def train_step(self, X, y):
        """
        Perform one training step (forward + backward + update)

        Parameters:
        -----------
        X: array-like, shape (n_samples, n_features)
            Input data
        y: array-like, shape (n_samples, n_classes)
            True
        
        Returns:
        -----------
        loss: float
            Loss value for this step
        """
        
        output = self.forward(X)

        loss = self.compute_loss(y, output)

        self.backward(y)

        self.update_parameters()

        return loss