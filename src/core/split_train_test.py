import numpy as np
from load_and_preprocess_data import X, y_one_hot

def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets.

    Parameters
    -----------
    X: array-like, shape (n_samples, n_features) 
        Features
    y: array-like, shape (n_samples, n_classes) 
        Labels (one-hot encoded)
    test_size: float, default=0.2 that would be 20% of data
        Proportion of dataset to include into testing split 
    random_state: int, default=42
        Random seed for reproductibility
    
    Returns:
    -----------
    X_train, X_test, y_train, y_test : arrays
        Split datasets
    """

    #rand seed
    np.random.seed(random_state)

    #num sampl
    n_samples = X.shape[0]
    
    #calc num test sampl
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    #gen rand indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # split
    train_indices = indices[:n_train]
    test_indices = indices [n_train:]

    #split data using indices
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training labels shape: {y_train.shape[0]}")
print(f"Test labels shape: {y_test.shape}")
print("-----------------------------------------------------")
