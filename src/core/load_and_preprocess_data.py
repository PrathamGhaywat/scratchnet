import numpy as np
from sklearn.datasets import fetch_openml

#local made funcs
# note to self: BEWARE OF CIRCULAR IMPORTS
from onehotencode import one_hot_encode

# load dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', parser='auto', as_frame=False)
X, y = mnist.data, mnist.target

# to numpy array conversion and normalization
X = np.array(X, dtype=np.float32) / 255.0 # normalize to 0, 1
y = np.array(y, dtype=np.int32) 

print(f"Data shape: {X.shape}") # (70000, 784)
print(f"Labels shape: {y.shape}") # (70000, )

# One-hot encode labels
y_one_hot = one_hot_encode(y)
print(f"One-hot labels shape: {y_one_hot.shape}") # (70000, 10)
print("-----------------------------------------------------")
