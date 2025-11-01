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
    
    
# test
if __name__  == "__main__":
    #784 inputs, 2 hidden layers, 10 outputs
    nn = NeuralNetwork(layer_sizes=[784, 128, 64, 10], learning_rate=0.01)

    print("\nWeight shapes:")
    for i, (w, b) in enumerate(zip(nn.weights, nn.biases)):
        print(f"Layer {i+1}: W{i+1} = {w.shape}, b{i+1} = {b.shape}")

    # test 2
    print("\n=== Testing Activation Functions ===")
    test_input = np.array([[-2, -1, 0, 1, 2]])
    print(f"\nReLU test:")
    print(f"Input:  {test_input}")
    print(f"Output: {nn.relu(test_input)}")
    print(f"Derivative: {nn.relu_derivative(test_input)}")

    test_logits = np.array([[2.0,1.0,0.5]])
    print(f"\nSoftmax test:")
    print(f"Input:  {test_logits}")
    softmax_output = nn.softmax(test_logits)
    print(f"Output: {softmax_output}")
    print(f"Sum: {np.sum(softmax_output):.6f} (should be 1.0)")
    
    # Test Loss
    y_true = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])  # True label: 3
    y_pred = np.array([[0.05, 0.03, 0.08, 0.72, 0.04, 0.02, 0.01, 0.03, 0.01, 0.01]])
    print(f"\nLoss test:")
    print(f"True label: {np.argmax(y_true)}")
    print(f"Predicted probabilities: {y_pred}")
    print(f"Loss: {nn.compute_loss(y_true, y_pred):.4f}")