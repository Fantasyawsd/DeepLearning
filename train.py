"""
Training script for deep learning models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MAE, BERT, BertForMaskedLM, BertForSequenceClassification, SwinTransformer
from utils import Config, get_logger


def get_model(config: Config):
    """Get model based on configuration."""
    model_name = config.get('model_name', '').lower()
    
    if model_name == 'mae':
        return MAE(config.to_dict())
    elif model_name == 'bert':
        task_type = config.get('task_type', 'masked_lm')
        if task_type == 'masked_lm':
            return BertForMaskedLM(config.to_dict())
        elif task_type == 'sequence_classification':
            return BertForSequenceClassification(config.to_dict())
        else:
            return BERT(config.to_dict())
    elif model_name == 'swin_transformer':
        return SwinTransformer(config.to_dict())
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_optimizer(model: nn.Module, config: Config):
    """Get optimizer based on configuration."""
    optimizer_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('learning_rate', 1e-3)
    weight_decay = config.get('weight_decay', 0.01)
    
    if optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, config: Config):
    """Get learning rate scheduler."""
    scheduler_type = config.get('scheduler', 'cosine')
    
    if scheduler_type == 'cosine':
        max_epochs = config.get('max_epochs', 100)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_type == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'linear':
        max_steps = config.get('max_steps', 1000000)
        warmup_steps = config.get('warmup_steps', 10000)
        return optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    else:
        return None


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer, scheduler, device, epoch: int, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        else:
            batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(**batch if isinstance(batch, dict) else {'x': batch})
        
        # Calculate loss
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        else:
            # For models without built-in loss calculation
            loss = nn.CrossEntropyLoss()(outputs, batch['labels'] if isinstance(batch, dict) else batch[1])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / num_batches
    logger.info(f'Epoch {epoch} completed. Average Loss: {avg_loss:.6f}')
    return avg_loss


def validate_epoch(model: nn.Module, dataloader: DataLoader, device, epoch: int, logger):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = batch.to(device)
            
            # Forward pass
            outputs = model(**batch if isinstance(batch, dict) else {'x': batch})
            
            # Calculate loss
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                loss = nn.CrossEntropyLoss()(outputs, batch['labels'] if isinstance(batch, dict) else batch[1])
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    logger.info(f'Validation Epoch {epoch}. Average Loss: {avg_loss:.6f}')
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train deep learning models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_file(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = get_logger('train', log_file=str(output_dir / 'train.log'))
    logger.info(f'Using device: {device}')
    logger.info(f'Configuration: {config}')
    
    # Create model
    model = get_model(config)
    model.to(device)
    logger.info(f'Model created: {model.model_name}')
    model.summary()
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Create dummy dataloaders (replace with actual data loading)
    # This is a placeholder - in practice, you would load actual datasets
    batch_size = config.get('batch_size', 32)
    train_dataloader = None  # Replace with actual train dataloader
    val_dataloader = None    # Replace with actual validation dataloader
    
    if train_dataloader is None:
        logger.warning("No training dataloader provided. This is a demo script.")
        logger.info("To use this script with real data, implement proper dataloaders.")
        return
    
    # Training loop
    max_epochs = config.get('max_epochs', 100)
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, max_epochs):
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, epoch, logger)
        
        # Validate
        if val_dataloader:
            val_loss = validate_epoch(model, val_dataloader, device, epoch, logger)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = output_dir / 'best_model.pth'
                model.save_checkpoint(str(best_model_path), epoch, optimizer.state_dict())
                logger.info(f'New best model saved: {best_model_path}')
        
        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            model.save_checkpoint(str(checkpoint_path), epoch, optimizer.state_dict())
            logger.info(f'Checkpoint saved: {checkpoint_path}')
    
    logger.info('Training completed!')


if __name__ == '__main__':
    main()