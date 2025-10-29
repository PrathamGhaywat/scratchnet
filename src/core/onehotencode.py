import numpy as np

def one_hot_encode(labels, num_classes=10):
    """Convert int labels to one-hot encoded vecs"""
    n_samples = labels.shape[0]
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), labels] = 1
    return one_hot