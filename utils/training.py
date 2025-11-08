"""
NHL Training Orchestrator - Curriculum Learning Approach

Training Strategy:
1. Pre-1960: 12 epochs (learn fundamentals on different era)
2. 1960-2000: 12 epochs (transition period, game evolving)
3. 2000-2020: 4 epochs (modern game, more data)
4. Full history pass: 1 epoch (prevent overfitting)
5. 2021-2025: Validation only (no training)
6. Backtest: 2010-2025 (final validation)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class TrainingPhase:
    """Configuration for a training phase"""
    name: str
    start_date: str
    end_date: str
    num_epochs: int
    is_validation: bool = False
    description: str = ""


class TrainingOrchestrator:
    """
    Manages curriculum learning with multiple phases.
    Each phase trains on different eras to prevent overfitting.
    """
    def __init__(self, model, optimizer, loss_fn,
                 checkpoint_dir='./checkpoints',
                 log_dir='./logs'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Metrics tracking
        self.phase_history = []
        self.best_val_sharpe = -float('inf')
        self.best_val_loss = float('inf')
        
        # Define curriculum phases
        self.phases = [
            TrainingPhase(
                name="pre_1960",
                start_date="1917-01-01",  # NHL founding
                end_date="1959-12-31",
                num_epochs=12,
                description="Original Six era - learn fundamentals"
            ),
            TrainingPhase(
                name="expansion_era",
                start_date="1960-01-01",
                end_date="1999-12-31",
                num_epochs=12,
                description="Expansion & transition - evolving game"
            ),
            TrainingPhase(
                name="modern_era",
                start_date="2000-01-01",
                end_date="2020-12-31",
                num_epochs=4,
                description="Modern NHL - more data available"
            ),
            TrainingPhase(
                name="full_history",
                start_date="1917-01-01",
                end_date="2020-12-31",
                num_epochs=1,
                description="Complete history pass - regularization"
            ),
            TrainingPhase(
                name="recent_validation",
                start_date="2021-01-01",
                end_date="2025-12-31",
                num_epochs=1,
                is_validation=True,
                description="Recent seasons - validation only"
            )
        ]
    
    def create_training_step(self):
        """Create optimized training step with all heads"""
        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                
                # Calculate loss for each head
                total_loss = 0.0
                loss_components = {}
                
                # Moneyline loss (binary cross-entropy)
                if 'moneyline' in predictions and 'win' in targets:
                    ml_loss = tf.keras.losses.binary_crossentropy(
                        targets['win'], 
                        predictions['moneyline']
                    )
                    ml_loss = tf.reduce_mean(ml_loss)
                    loss_components['moneyline'] = ml_loss
                    total_loss += ml_loss
                
                # Puck line loss (MDN negative log-likelihood)
                if 'puck_line' in predictions and 'spread' in targets:
                    spread_dist = predictions['puck_line']
                    spread_target = targets['spread']
                    
                    # Calculate probability of actual spread under mixture
                    mu = spread_dist['mu']
                    sigma = spread_dist['sigma']
                    pi = spread_dist['pi']
                    
                    # Expand target for broadcasting
                    spread_target_exp = tf.expand_dims(spread_target, -1)
                    
                    # Normal PDF for each component
                    normal_probs = (1.0 / (sigma * tf.sqrt(2.0 * np.pi))) * \
                                   tf.exp(-0.5 * tf.square((spread_target_exp - mu) / sigma))
                    
                    # Weighted sum
                    mixture_prob = tf.reduce_sum(pi * normal_probs, axis=-1)
                    
                    # Negative log likelihood
                    pl_loss = -tf.reduce_mean(tf.math.log(mixture_prob + 1e-8))
                    loss_components['puck_line'] = pl_loss
                    total_loss += pl_loss * 0.5  # Weight puck line less
                
                # Total goals loss (MSE)
                if 'total' in predictions and 'total_goals' in targets:
                    total_loss_component = tf.keras.losses.mean_squared_error(
                        targets['total_goals'],
                        tf.squeeze(predictions['total'], -1)
                    )
                    total_loss_component = tf.reduce_mean(total_loss_component)
                    loss_components['total'] = total_loss_component
                    total_loss += total_loss_component * 0.3  # Weight total less
                
                # Add regularization
                if self.model.losses:
                    reg_loss = tf.add_n(self.model.losses)
                    loss_components['regularization'] = reg_loss
                    total_loss += reg_loss
            
            # Compute gradients
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            
            # Clip gradients
            gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
            
            # Apply gradients
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
            
            return total_loss, loss_components, predictions
        
        return train_step
    
    def evaluate_predictions(self, predictions_list, targets_list):
        """
        Evaluate model predictions across all heads.
        
        Returns dict with metrics for each head.
        """
        metrics = {}
        
        # Concatenate all predictions and targets
        all_preds = {}
        all_targets = {}
        
        for pred, target in zip(predictions_list, targets_list):
            for key in pred:
                if key not in all_preds:
                    all_preds[key] = []
                all_preds[key].append(pred[key])
            
            for key in target:
                if key not in all_targets:
                    all_targets[key] = []
                all_targets[key].append(target[key])
        
        # Concatenate
        for key in all_preds:
            if isinstance(all_preds[key][0], dict):
                # Handle dict outputs (like puck_line)
                continue
            all_preds[key] = np.concatenate([p.numpy() if hasattr(p, 'numpy') else p 
                                             for p in all_preds[key]], axis=0)
        
        for key in all_targets:
            all_targets[key] = np.concatenate([t.numpy() if hasattr(t, 'numpy') else t 
                                               for t in all_targets[key]], axis=0)
        
        # Moneyline metrics
        if 'moneyline' in all_preds and 'win' in all_targets:
            ml_probs = all_preds['moneyline'].flatten()
            actual_wins = all_targets['win'].flatten()
            
            # Brier score
            brier = np.mean((ml_probs - actual_wins) ** 2)
            
            # Log loss
            log_loss = -np.mean(
                actual_wins * np.log(ml_probs + 1e-8) +
                (1 - actual_wins) * np.log(1 - ml_probs + 1e-8)
            )
            
            # Accuracy
            accuracy = np.mean((ml_probs > 0.5) == actual_wins)
            
            # Calibration (compare predicted probs to actual win rates in bins)
            calibration_error = self._compute_calibration_error(ml_probs, actual_wins)
            
            metrics['moneyline'] = {
                'brier_score': float(brier),
                'log_loss': float(log_loss),
                'accuracy': float(accuracy),
                'calibration_error': float(calibration_error)
            }
        
        # Puck line metrics
        if 'spread' in all_targets:
            actual_spreads = all_targets['spread'].flatten()
            
            # We'd need to sample from the mixture for full evaluation
            # For now, use a simple metric
            metrics['puck_line'] = {
                'mean_actual_spread': float(np.mean(actual_spreads)),
                'std_actual_spread': float(np.std(actual_spreads))
            }
        
        # Total goals metrics
        if 'total' in all_preds and 'total_goals' in all_targets:
            pred_totals = all_preds['total'].flatten()
            actual_totals = all_targets['total_goals'].flatten()
            
            mae = np.mean(np.abs(pred_totals - actual_totals))
            rmse = np.sqrt(np.mean((pred_totals - actual_totals) ** 2))
            
            metrics['total'] = {
                'mae': float(mae),
                'rmse': float(rmse)
            }
        
        return metrics
    
    def _compute_calibration_error(self, probs, outcomes, num_bins=10):
        """Compute expected calibration error"""
        bins = np.linspace(0, 1, num_bins + 1)
        bin_errors = []
        
        for i in range(num_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                bin_prob = probs[mask].mean()
                bin_outcome = outcomes[mask].mean()
                bin_errors.append(abs(bin_prob - bin_outcome) * mask.sum())
        
        if len(bin_errors) == 0:
            return 0.0
        
        return sum(bin_errors) / len(probs)
    
    def run_backtest_on_predictions(self, predictions_list, targets_list, 
                                     metadata_list):
        """
        Run backtesting on model predictions.
        """
        self.backtester.reset()
        
        for pred, target, meta in zip(predictions_list, targets_list, metadata_list):
            # Extract predictions
            win_prob = pred['moneyline'].numpy()[0, 0]
            actual_win = target['win'].numpy()[0, 0]
            
            # Mock market odds (in production, would come from metadata)
            # For now, use simple conversion from probability
            implied_prob = 0.5  # Market is neutral
            market_odds = 1.0 / implied_prob if implied_prob > 0 else 2.0
            
            game_id = meta.get('game_id', 0)
            
            # Place bet if we have an edge
            self.backtester.place_bet(
                game_id=game_id,
                model_prob=win_prob,
                market_odds=market_odds,
                actual_outcome=actual_win
            )
        
        # Compute metrics
        bt_metrics = self.backtester.compute_metrics()
        
        return bt_metrics
    
    def train_phase(self, phase: TrainingPhase, data_loader):
        """
        Train model for one phase of curriculum.
        
        Args:
            phase: TrainingPhase configuration
            data_loader: Function that returns (train_data, val_data) for date range
        """
        print("\n" + "="*80)
        print(f"PHASE: {phase.name.upper()}")
        print("="*80)
        print(f"  Period: {phase.start_date} to {phase.end_date}")
        print(f"  Epochs: {phase.num_epochs}")
        print(f"  Mode: {'VALIDATION' if phase.is_validation else 'TRAINING'}")
        print(f"  Description: {phase.description}")
        print("="*80 + "\n")
        
        # Load data for this phase
        print("Loading data...")
        train_data, val_data = data_loader(phase.start_date, phase.end_date)
        
        if train_data is None or len(train_data) == 0:
            print(f"⚠ No data available for phase {phase.name}, skipping...")
            return None
        
        print(f"✓ Loaded {len(train_data)} training games")
        if val_data:
            print(f"✓ Loaded {len(val_data)} validation games")
        
        # Create training step
        train_step = self.create_training_step()
        
        phase_results = {
            'phase': phase.name,
            'start_date': phase.start_date,
            'end_date': phase.end_date,
            'is_validation': phase.is_validation,
            'epochs': []
        }
        
        # Training loop
        for epoch in range(phase.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{phase.num_epochs} ---")
            
            epoch_losses = []
            epoch_loss_components = []
            
            if not phase.is_validation:
                # Training
                print("Training...")
                for batch_idx, (inputs, targets) in enumerate(train_data):
                    loss, loss_components, predictions = train_step(inputs, targets)
                    
                    epoch_losses.append(loss.numpy())
                    epoch_loss_components.append(
                        {k: v.numpy() for k, v in loss_components.items()}
                    )
                    
                    if batch_idx % 20 == 0:
                        print(f"  Batch {batch_idx}: Loss = {loss.numpy():.4f}")
                
                avg_loss = np.mean(epoch_losses)
                avg_components = {}
                for key in epoch_loss_components[0].keys():
                    avg_components[key] = np.mean([
                        c[key] for c in epoch_loss_components
                    ])
                
                print(f"\nEpoch {epoch + 1} Training Summary:")
                print(f"  Total Loss: {avg_loss:.4f}")
                for key, val in avg_components.items():
                    print(f"  {key}: {val:.4f}")
            
            # Validation
            if val_data:
                print("\nValidating...")
                val_predictions = []
                val_targets = []
                val_metadata = []
                
                for inputs, targets, metadata in val_data:
                    preds = self.model(inputs, training=False)
                    val_predictions.append(preds)
                    val_targets.append(targets)
                    val_metadata.append(metadata)
                
                # Compute metrics
                metrics = self.evaluate_predictions(val_predictions, val_targets)
                
                print("\nValidation Metrics:")
                for head, head_metrics in metrics.items():
                    print(f"\n  {head.upper()}:")
                    for metric_name, value in head_metrics.items():
                        print(f"    {metric_name}: {value:.4f}")
                
                # Run backtest
                print("\nRunning backtest...")
                bt_metrics = self.run_backtest_on_predictions(
                    val_predictions, val_targets, val_metadata
                )
                
                print("\nBacktest Results:")
                print(f"  Sharpe Ratio: {bt_metrics.get('sharpe_ratio', 0):.3f}")
                print(f"  Total Return: {bt_metrics.get('total_return', 0)*100:.2f}%")
                print(f"  Win Rate: {bt_metrics.get('win_rate', 0)*100:.2f}%")
                print(f"  Max Drawdown: {bt_metrics.get('max_drawdown', 0)*100:.2f}%")
                print(f"  Num Bets: {bt_metrics.get('num_bets', 0)}")
                
                # Check if best model
                sharpe = bt_metrics.get('sharpe_ratio', -999)
                if sharpe > self.best_val_sharpe:
                    self.best_val_sharpe = sharpe
                    self.save_checkpoint(f"best_{phase.name}_epoch_{epoch+1}", metrics, bt_metrics)
                    print(f"\n  ✓ New best Sharpe: {sharpe:.3f}")
                
                # Store epoch results
                epoch_result = {
                    'epoch': epoch + 1,
                    'metrics': metrics,
                    'backtest': bt_metrics
                }
                
                if not phase.is_validation:
                    epoch_result['train_loss'] = avg_loss
                    epoch_result['loss_components'] = avg_components
                
                phase_results['epochs'].append(epoch_result)
        
        # Save phase results
        self.phase_history.append(phase_results)
        self.save_phase_results(phase_results)
        
        return phase_results
    
    def run_full_curriculum(self, data_loader):
        """
        Run complete curriculum learning pipeline.
        
        Args:
            data_loader: Function(start_date, end_date) -> (train_data, val_data)
        """
        print("\n" + "="*80)
        print("CURRICULUM LEARNING - FULL PIPELINE")
        print("="*80)
        print(f"Total Phases: {len(self.phases)}")
        print(f"Start: {datetime.now()}")
        print("="*80)
        
        for phase in self.phases:
            results = self.train_phase(phase, data_loader)
            
            if results is None:
                continue
        
        print("\n" + "="*80)
        print("CURRICULUM COMPLETE")
        print("="*80)
        print(f"Best Validation Sharpe: {self.best_val_sharpe:.3f}")
        print(f"Phases Completed: {len(self.phase_history)}")
        print("="*80)
        
        return self.phase_history
    
    def save_checkpoint(self, name: str, metrics: Dict, backtest_metrics: Dict):
        """Save model checkpoint with metrics"""
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        self.model.save_weights(checkpoint_path)
        
        # Save metrics
        metrics_path = checkpoint_path + "_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'metrics': {k: {kk: float(vv) for kk, vv in v.items()} 
                           for k, v in metrics.items()},
                'backtest': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in backtest_metrics.items()}
            }, f, indent=2)
        
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    def save_phase_results(self, phase_results: Dict):
        """Save phase results to log"""
        log_path = os.path.join(
            self.log_dir, 
            f"phase_{phase_results['phase']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(log_path, 'w') as f:
            json.dump(phase_results, f, indent=2, default=str)
    
    def final_backtest(self, data_loader, start_date="2010-01-01", end_date="2025-12-31"):
        """
        Run final comprehensive backtest on recent data.
        This is the ultimate validation.
        """
        print("\n" + "="*80)
        print("FINAL BACKTEST: 2010-2025")
        print("="*80)
        print("This is the ultimate test of model performance.")
        print("No training - pure prediction on recent history.")
        print("="*80 + "\n")
        
        # Load data
        print("Loading 15 years of data...")
        _, test_data = data_loader(start_date, end_date)
        
        if not test_data:
            print("✗ No test data available")
            return None
        
        print(f"✓ Loaded {len(test_data)} games\n")
        
        # Generate predictions
        print("Generating predictions...")
        predictions = []
        targets = []
        metadata = []
        
        for batch_idx, (inputs, target, meta) in enumerate(test_data):
            pred = self.model(inputs, training=False)
            predictions.append(pred)
            targets.append(target)
            metadata.append(meta)
            
            if batch_idx % 100 == 0:
                print(f"  Processed {batch_idx}/{len(test_data)} games...")
        
        print("✓ Predictions complete\n")
        
        # Evaluate
        print("Computing metrics...")
        metrics = self.evaluate_predictions(predictions, targets)
        
        print("\nPrediction Quality Metrics:")
        for head, head_metrics in metrics.items():
            print(f"\n  {head.upper()}:")
            for metric_name, value in head_metrics.items():
                print(f"    {metric_name}: {value:.4f}")
        
        # Backtest
        print("\nRunning backtest...")
        bt_metrics = self.run_backtest_on_predictions(predictions, targets, metadata)
        
        print("\n" + "="*80)
        print("FINAL BACKTEST RESULTS")
        print("="*80)
        print(f"Period: {start_date} to {end_date}")
        print(f"Games: {len(test_data)}")
        print(f"\nPerformance:")
        print(f"  Sharpe Ratio: {bt_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Total Return: {bt_metrics.get('total_return', 0)*100:.2f}%")
        print(f"  Total Profit: ${bt_metrics.get('total_profit', 0):.2f}")
        print(f"  Win Rate: {bt_metrics.get('win_rate', 0)*100:.2f}%")
        print(f"  Avg Edge: {bt_metrics.get('avg_edge', 0)*100:.2f}%")
        print(f"  Max Drawdown: {bt_metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Num Bets Placed: {bt_metrics.get('num_bets', 0)}")
        print(f"  Final Bankroll: ${bt_metrics.get('final_bankroll', 0):.2f}")
        print("="*80)
        
        # Save results
        results_path = os.path.join(
            self.log_dir,
            f"final_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump({
                'start_date': start_date,
                'end_date': end_date,
                'num_games': len(test_data),
                'metrics': {k: {kk: float(vv) for kk, vv in v.items()} 
                           for k, v in metrics.items()},
                'backtest': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in bt_metrics.items()}
            }, f, indent=2)
        
        print(f"\n✓ Results saved: {results_path}")
        
        return {
            'metrics': metrics,
            'backtest': bt_metrics
        }


# ============================================================================
# HELPER: DATA LOADER
# ============================================================================

def create_data_loader(preprocessor, db_path="nhl_data.db"):
    """
    Create data loader function for training orchestrator.
    
    Returns function(start_date, end_date) -> (train_data, val_data)
    """
    def load_data(start_date: str, end_date: str):
        """
        Load and preprocess data for date range.
        
        Returns:
            train_data: List of (inputs, targets, metadata) batches
            val_data: List of (inputs, targets, metadata) batches
        """
        from data.queries import get_training_data
        
        # Load raw data
        df = get_training_data(
            start_date=start_date,
            end_date=end_date,
            lookback_games=10
        )
        
        if len(df) == 0:
            return None, None
        
        # Split into train/val (80/20)
        split_idx = int(len(df) * 0.8)
        train_df = df[:split_idx]
        val_df = df[split_idx:]
        
        # Process each game
        train_data = []
        for idx, row in train_df.iterrows():
            try:
                processed = preprocessor.process_game(row)
                train_data.append((
                    processed['inputs'],
                    processed['targets'],
                    processed['metadata']
                ))
            except Exception as e:
                # Skip problematic games
                continue
        
        val_data = []
        for idx, row in val_df.iterrows():
            try:
                processed = preprocessor.process_game(row)
                val_data.append((
                    processed['inputs'],
                    processed['targets'],
                    processed['metadata']
                ))
            except Exception as e:
                continue
        
        return train_data, val_data
    
    return load_data