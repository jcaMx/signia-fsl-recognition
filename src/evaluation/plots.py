import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_history(history, title='Model Accuracy', save_path=None):
    """
    Plots training and validation accuracy and loss from Keras training history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{title} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{title} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap='Blues', save_path=None):
    """
    Plots the confusion matrix as a heatmap.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
