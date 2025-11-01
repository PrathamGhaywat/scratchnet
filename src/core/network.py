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
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1])
            #bias vector
            b = np.zeros((1, layer_sizes[i+1]))

            self.weights.append(w)
            self.biases.append(b)
        
        print(f"Intiialization of network with architecture: {layer_sizes}")
        print(f"Number of Parameters ")

    def count_parameters(self):
        """Count total numbers of trainable parameters"""
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total
    
# test
if __name__  == "__main__":
    #784 inputs, 2 hidden layers, 10 outputs
    nn = NeuralNetwork(layer_sizes=[784, 128, 64, 10], learning_rate=0.01)

    print("\nWeight shapes:")
    for i, (w, b) in enumerate(zip(nn.weights, nn.biases)):
        print(f"Layer {i+1}: W{i+1} = {w.shape}, b{i+1} = {b.shape}")