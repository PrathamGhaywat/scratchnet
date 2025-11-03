import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('core')

from network import NeuralNetwork
from load_and_preprocess_data import X, y_one_hot
from split_train_test import X_train, X_test, y_train, y_test

def train_network(nn, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
    """
    Train the Neural Network

    Parameters:
    -----------
    nn: NeuralNetwork
        The network to train
    X_train, y_train: The training data
    X_test, y_test: Test data
    epochs: int
        Number of omplete passes through training data
    batch_size: int
        Number of samples per mini-batch
    """
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    print(f"\n{'='*60}")
    print(f"Starting training: {epochs} epochs, batch size {batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        #shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_losses = []

        #mini batch training
        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            loss = nn.train_step(X_batch, y_batch)
            epoch_losses.append(loss)

        
        avg_train_loss = np.mean(epoch_losses)
        train_acc = nn.accuracy(X_train, y_train)

        test_pred = nn.forward(X_test)
        test_loss = nn.compute_loss(y_test, test_pred)
        test_acc = nn.accuracy(X_test, y_test)

        #store metrics
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Train acc: {train_acc:.4f} | "
              f"Test loss: {test_loss:.4f} | "
              f"Test acc: {test_acc:.4f}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,  
        'test_accuracies': test_accuracies     
    }

def plot_training_results(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #losses
    ax1.plot(history['train_losses'], label='Train Loss', marker='o')
    ax1.plot(history['test_losses'], label='Test Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')  
    ax1.set_title('Training and Test Loss')
    ax1.legend()  
    ax1.grid(True)

    #accuracies
    ax2.plot(history['train_accuracies'], label='Train accuracy', marker='o')
    ax2.plot(history['test_accuracies'], label='Test accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    print("\nSaved training plot'")
    plt.show()

def visualize_predictions(nn, X_test, y_test, n_samples=10):
    """Visualize some predictions"""
    predictions = nn.predict(X_test[:n_samples])
    true_labels = np.argmax(y_test[:n_samples], axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i in range(n_samples):
        image = X_test[i].reshape(28, 28)
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"True: {true_labels[i]}\nPred: {predictions[i]}",
                          color='green' if predictions[i] == true_labels[i] else 'red')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Pred plot saved")
    plt.show()


if __name__ == "__main__":
    print("Creating nn")
    nn = NeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        learning_rate=0.1
    )

    history = train_network(
        nn,
        X_train, y_train,
        X_test, y_test,
        epochs=20,
        batch_size=128
    )

    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Final Train accuracy: {history['train_accuracies'][-1]:.4f}")
    print(f"Final Test accuracy: {history['test_accuracies'][-1]:.4f}")
    print(f"{'='*60}\n")

    plot_training_results(history)

    visualize_predictions(nn, X_test, y_test, n_samples=10)