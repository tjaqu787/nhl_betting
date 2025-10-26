import numpy as np
import pandas as pd
import tensorflow as tf
import json
from datetime import timedelta

# ============================================================================
# 1. TRAINING LOOP WITH WALK-FORWARD VALIDATION
# ============================================================================

class TrainingOrchestrator:
    """
    Manages complete training with walk-forward validation and backtesting.
    """
    def __init__(self, model, optimizer, loss_fn, backtester, 
                 checkpoint_dir='./checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.backtester = backtester
        self.checkpoint_dir = checkpoint_dir
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.backtest_results = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_sharpe = -float('inf')
    
    def train_epoch(self, train_dataset, epoch):
        """Single training epoch."""
        epoch_losses = []
        
        for batch_idx, (inputs, targets) in enumerate(train_dataset):
            # Forward pass
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss = self.loss_fn(targets, predictions)
                
                # Add regularization
                if self.model.losses:
                    loss += tf.add_n(self.model.losses)
            
            # Backward pass
            gradients = tape.gradient(loss, self.model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
            
            epoch_losses.append(loss.numpy())
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.numpy():.4f}")
        
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        print(f"Epoch {epoch} - Avg Train Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, val_dataset):
        """Validation pass."""
        val_losses = []
        all_predictions = []
        all_targets = []
        
        for inputs, targets in val_dataset:
            predictions = self.model(inputs, training=False)
            loss = self.loss_fn(targets, predictions)
            
            val_losses.append(loss.numpy())
            all_predictions.append(predictions)
            all_targets.append(targets)
        
        avg_val_loss = np.mean(val_losses)
        self.val_losses.append(avg_val_loss)
        
        # Compute additional metrics
        win_probs = np.concatenate([p['win_prob'].numpy() for p in all_predictions])
        true_wins = np.concatenate([t['win'].numpy() for t in all_targets])
        
        # Brier score
        brier = np.mean((win_probs - true_wins) ** 2)
        
        # Log loss
        log_loss = -np.mean(
            true_wins * np.log(win_probs + 1e-8) +
            (1 - true_wins) * np.log(1 - win_probs + 1e-8)
        )
        
        print(f"  Validation - Loss: {avg_val_loss:.4f}, Brier: {brier:.4f}, LogLoss: {log_loss:.4f}")
        
        return avg_val_loss, all_predictions, all_targets
    
    def backtest_predictions(self, predictions, targets, metadata):
        """
        Run backtest on predictions.
        """
        self.backtester.reset()
        
        for i, (pred, target, meta) in enumerate(zip(predictions, targets, metadata)):
            win_prob = pred['win_prob'].numpy()[0, 0]
            market_odds = target['odds'].numpy()[0, 0]
            actual_outcome = target['win'].numpy()[0, 0]
            game_id = meta['game_id']
            
            self.backtester.place_bet(
                game_id, win_prob, market_odds, actual_outcome
            )
        
        metrics = self.backtester.compute_metrics()
        self.backtest_results.append(metrics)
        
        print(f"\n  Backtest Results:")
        print(f"    Total Profit: ${metrics.get('total_profit', 0):.2f}")
        print(f"    Total Return: {metrics.get('total_return', 0)*100:.2f}%")
        print(f"    Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"    Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"    Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
        print(f"    Num Bets: {metrics.get('num_bets', 0)}")
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint."""
        checkpoint_path = f"{self.checkpoint_dir}/model_epoch_{epoch}"
        self.model.save_weights(checkpoint_path)
        
        # Save metrics
        with open(f"{checkpoint_path}_metrics.json", 'w') as f:
            json.dump({
                'epoch': epoch,
                'train_loss': float(self.train_losses[-1]),
                'val_loss': float(self.val_losses[-1]),
                'backtest': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in metrics.items()}
            }, f, indent=2)
        
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def train_walk_forward(self, preprocessor, game_logs, 
                          train_window_months=6, val_window_months=1,
                          epochs_per_window=5):
        """
        Complete walk-forward training pipeline.
        """
        print("="*80)
        print("WALK-FORWARD TRAINING")
        print("="*80)
        
        # Sort games by date
        game_logs = game_logs.sort_values('game_date').reset_index(drop=True)
        game_logs['game_date'] = pd.to_datetime(game_logs['game_date'])
        
        # Split into windows
        start_date = game_logs['game_date'].min()
        end_date = game_logs['game_date'].max()
        
        current_date = start_date
        window_idx = 0
        
        while current_date + timedelta(days=train_window_months*30 + val_window_months*30) < end_date:
            train_end = current_date + timedelta(days=train_window_months*30)
            val_end = train_end + timedelta(days=val_window_months*30)
            
            print(f"\n{'='*80}")
            print(f"Window {window_idx}: {current_date.date()} to {val_end.date()}")
            print(f"  Train: {current_date.date()} to {train_end.date()}")
            print(f"  Val: {train_end.date()} to {val_end.date()}")
            print(f"{'='*80}\n")
            
            # Filter data
            train_mask = (game_logs['game_date'] >= current_date) & \
                        (game_logs['game_date'] < train_end)
            val_mask = (game_logs['game_date'] >= train_end) & \
                      (game_logs['game_date'] < val_end)
            
            train_games = game_logs[train_mask]
            val_games = game_logs[val_mask]
            
            print(f"Train games: {len(train_games)}, Val games: {len(val_games)}")
            
            if len(train_games) == 0 or len(val_games) == 0:
                print("Insufficient data, skipping window")
                current_date = val_end
                window_idx += 1
                continue
            
            # Process games (mock for now - replace with real preprocessing)
            # In production: train_data = [preprocessor.process_game(row) for row in train_games]
            
            # Train for multiple epochs on this window
            for epoch in range(epochs_per_window):
                print(f"\n--- Epoch {epoch+1}/{epochs_per_window} ---")
                
                # Create TF datasets (mock for illustration)
                # train_dataset = create_tf_dataset(train_data)
                # val_dataset = create_tf_dataset(val_data)
                
                # self.train_epoch(train_dataset, epoch)
                # val_loss, predictions, targets = self.validate(val_dataset)
                
                # Placeholder
                print("  [Training...]")
                val_loss = 0.5 - window_idx * 0.01  # Mock improvement
            
            # Backtest on validation set
            print("\n--- Backtesting validation period ---")
            # metrics = self.backtest_predictions(predictions, targets, metadata)
            metrics = {'sharpe_ratio': 0.8 + window_idx * 0.1}  # Mock
            
            # Save if best
            if metrics['sharpe_ratio'] > self.best_sharpe:
                self.best_sharpe = metrics['sharpe_ratio']
                self.save_checkpoint(window_idx, metrics)
                print(f"  ✓ New best Sharpe: {self.best_sharpe:.3f}")
            
            # Move to next window
            current_date = val_end
            window_idx += 1
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"  Best Sharpe Ratio: {self.best_sharpe:.3f}")
        print(f"  Total Windows: {window_idx}")
        print(f"{'='*80}\n")


# ============================================================================
# 2. TRAINING UTILITIES
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
