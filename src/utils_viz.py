"""
Visualization utilities for MLflow logging
"""
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import io
import base64


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create and log confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar(im, ax=ax)
    
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    
    # Add values to cells
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    
    mlflow.log_figure(fig, f"confusion_matrix_{model_name}.png")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Create and log ROC curve"""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        mlflow.log_figure(fig, f"roc_curve_{model_name}.png")
        plt.close()
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")


def plot_metrics_comparison(metrics_dict):
    """Create and log metrics comparison bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(metrics_dict.keys())
    metrics = ['accuracy', 'f1', 'auc'] if 'auc' in metrics_dict[models[0]] else ['accuracy', 'f1']
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[model].get(metric, 0) for model in models]
        ax.bar(x + i*width, values, width, label=metric.upper())
    
    ax.set_ylabel('Score')
    ax.set_title('Model Metrics Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 1.0])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    mlflow.log_figure(fig, "metrics_comparison.png")
    plt.close()


def plot_training_history(loss_history, val_loss_history, model_name):
    """Create and log training history plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, 'b-', label='Training Loss', linewidth=2)
    if val_loss_history:
        ax.plot(epochs, val_loss_history, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training History - {model_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    mlflow.log_figure(fig, f"training_history_{model_name}.png")
    plt.close()
