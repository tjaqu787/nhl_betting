import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================================
# 11. TRAINING UTILITIES
# ============================================================================

class WalkForwardValidator:
    """
    Walk-forward validation with strict temporal ordering.
    No future leakage.
    """
    def __init__(self, train_window_months=6, val_window_months=1):
        self.train_window_months = train_window_months
        self.val_window_months = val_window_months
    
    def split(self, data, timestamps):
        """
        Args:
            data: dataset with temporal ordering
            timestamps: array of timestamps
        Yields:
            (train_indices, val_indices)
        """
        sorted_indices = np.argsort(timestamps)
        total_days = (timestamps[-1] - timestamps[0]).days
        
        train_days = self.train_window_months * 30
        val_days = self.val_window_months * 30
        
        current_start = 0
        while current_start + train_days + val_days < total_days:
            train_end = current_start + train_days
            val_end = train_end + val_days
            
            train_mask = (timestamps >= timestamps[0] + pd.Timedelta(days=current_start)) & \
                        (timestamps < timestamps[0] + pd.Timedelta(days=train_end))
            val_mask = (timestamps >= timestamps[0] + pd.Timedelta(days=train_end)) & \
                      (timestamps < timestamps[0] + pd.Timedelta(days=val_end))
            
            train_indices = sorted_indices[train_mask]
            val_indices = sorted_indices[val_mask]
            
            yield train_indices, val_indices
            
            # Move forward by val_window
            current_start += val_days


def create_training_step(model, optimizer, loss_fn):
    """
    Custom training step with gradient clipping.
    """
    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(targets, predictions)
            
            # Add regularization losses
            if model.losses:
                loss += tf.add_n(model.losses)
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Clip gradients to prevent explosion
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss, predictions
    
    return train_step
