"""
Metrics and evaluation utilities.
"""

import torch
from typing import Tuple


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy between predictions and targets.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    correct = (predictions == targets).float()
    return correct.mean().item()


def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy as a float between 0 and 1
    """
    if predictions.dim() == 1:
        return accuracy(predictions, targets)
    
    _, top_k_preds = predictions.topk(k, dim=-1)
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=-1).float()
    return correct.mean().item()


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Compute multiple metrics at once.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
    
    Returns:
        Dictionary containing various metrics
    """
    metrics = {
        'accuracy': accuracy(predictions, targets),
        'top_5_accuracy': top_k_accuracy(predictions, targets, k=5)
    }
    return metrics