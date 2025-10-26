"""
Experiment Framework: Hyperparameter Tuning, Ablation Studies, A/B Testing
Includes: Bayesian optimization, experiment tracking, model comparison
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Any
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
import optuna
from optuna.samplers import TPESampler


# ============================================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Model architecture
    num_players: int = 500
    embed_dim: int = 128
    temporal_dim: int = 64
    num_gnn_layers: int = 2
    num_transformer_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    warmup_steps: int = 1000
    max_epochs: int = 50
    early_stopping_patience: int = 10
    
    # Loss weights
    lambda_win: float = 1.0
    lambda_spread: float = 0.5
    lambda_kelly: float = 0.1
    lambda_calibration: float = 0.2
    
    # Betting strategy
    initial_bankroll: float = 10000.0
    max_kelly_fraction: float = 0.02
    min_edge_threshold: float = 0.02
    vig: float = 0.05
    
    # Data
    train_window_months: int = 6
    val_window_months: int = 1
    temporal_seq_len: int = 10
    
    # Features
    use_temporal_embeddings: bool = True
    use_gnn: bool = True
    use_cross_attention: bool = True
    use_market_signals: bool = True
    use_game_state: bool = False  # For live betting
    
    # Meta-learning
    use_maml: bool = False
    maml_inner_steps: int = 5
    maml_inner_lr: float = 0.01
    
    # Causal inference
    use_double_ml: bool = False
    
    # Experiment metadata
    experiment_name: str = "baseline"
    notes: str = ""
    random_seed: int = 42
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# ============================================================================
# 2. EXPERIMENT TRACKER
# ============================================================================

class ExperimentTracker:
    """
    Track experiments: configs, metrics, models, predictions.
    Think of this as a lightweight MLflow alternative.
    """
    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        self.current_experiment = None
        self.current_run_dir = None
    
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start new experiment run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.experiment_name}_{timestamp}"
        run_dir = os.path.join(self.base_dir, run_name)
        
        os.makedirs(run_dir, exist_ok=True)
        
        # Save config
        config.save(os.path.join(run_dir, "config.json"))
        
        self.current_experiment = run_name
        self.current_run_dir = run_dir
        
        print(f"Started experiment: {run_name}")
        print(f"  Directory: {run_dir}")
        
        return run_dir
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for current step."""
        metrics_file = os.path.join(self.current_run_dir, "metrics.jsonl")
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps({
                'step': step,
                'timestamp': datetime.now().isoformat(),
                **metrics
            }) + '\n')
    
    def log_backtest_results(self, results: Dict, window_idx: int):
        """Log backtest results."""
        backtest_file = os.path.join(self.current_run_dir, "backtests.jsonl")
        
        with open(backtest_file, 'a') as f:
            f.write(json.dumps({
                'window': window_idx,
                'timestamp': datetime.now().isoformat(),
                **results
            }) + '\n')
    
    def save_model(self, model, name: str = "final"):
        """Save model checkpoint."""
        model_dir = os.path.join(self.current_run_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, name)
        model.save_weights(model_path)
        
        print(f"  Model saved: {model_path}")
    
    def save_predictions(self, predictions: pd.DataFrame, name: str = "predictions"):
        """Save predictions for analysis."""
        pred_file = os.path.join(self.current_run_dir, f"{name}.csv")
        predictions.to_csv(pred_file, index=False)
    
    def get_summary(self) -> Dict:
        """Get experiment summary."""
        # Load metrics
        metrics_file = os.path.join(self.current_run_dir, "metrics.jsonl")
        if not os.path.exists(metrics_file):
            return {}
        
        metrics = []
        with open(metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
        
        # Load backtests
        backtest_file = os.path.join(self.current_run_dir, "backtests.jsonl")
        backtests = []
        if os.path.exists(backtest_file):
            with open(backtest_file, 'r') as f:
                for line in f:
                    backtests.append(json.loads(line))
        
        return {
            'metrics': metrics,
            'backtests': backtests
        }


# ============================================================================
# 3. HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ============================================================================

class HyperparameterOptimizer:
    """
    Bayesian optimization for hyperparameters using Optuna.
    Optimizes for Sharpe ratio on validation backtest.
    """
    def __init__(self, 
                 base_config: ExperimentConfig,
                 n_trials: int = 50,
                 timeout: int = None):
        self.base_config = base_config
        self.n_trials = n_trials
        self.timeout = timeout
        
        self.study = None
        self.best_config = None
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function: train model with suggested hyperparams,
        return validation Sharpe ratio.
        """
        # Suggest hyperparameters
        config = ExperimentConfig(
            # Architecture
            embed_dim=trial.suggest_categorical('embed_dim', [64, 128, 256]),
            temporal_dim=trial.suggest_categorical('temporal_dim', [32, 64, 128]),
            num_gnn_layers=trial.suggest_int('num_gnn_layers', 1, 3),
            num_transformer_layers=trial.suggest_int('num_transformer_layers', 1, 3),
            num_heads=trial.suggest_categorical('num_heads', [4, 8]),
            dropout=trial.suggest_float('dropout', 0.0, 0.3),
            
            # Training
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            batch_size=trial.suggest_categorical('batch_size', [64, 128, 256]),
            
            # Loss weights
            lambda_win=1.0,  # Fixed
            lambda_spread=trial.suggest_float('lambda_spread', 0.1, 1.0),
            lambda_kelly=trial.suggest_float('lambda_kelly', 0.0, 0.3),
            lambda_calibration=trial.suggest_float('lambda_calibration', 0.0, 0.5),
            
            # Betting
            max_kelly_fraction=trial.suggest_float('max_kelly_fraction', 0.01, 0.05),
            min_edge_threshold=trial.suggest_float('min_edge_threshold', 0.01, 0.05),
            
            # Copy other params from base
            **{k: v for k, v in self.base_config.to_dict().items() 
               if k not in ['embed_dim', 'temporal_dim', 'num_gnn_layers', 
                           'num_transformer_layers', 'num_heads', 'dropout',
                           'learning_rate', 'weight_decay', 'batch_size',
                           'lambda_spread', 'lambda_kelly', 'lambda_calibration',
                           'max_kelly_fraction', 'min_edge_threshold']}
        )
        
        config.experiment_name = f"optuna_trial_{trial.number}"
        
        # Train model with this config
        sharpe_ratio = self._train_and_evaluate(config, trial)
        
        return sharpe_ratio
    
    def _train_and_evaluate(self, config: ExperimentConfig, 
                           trial: optuna.Trial) -> float:
        """
        Train model and return validation Sharpe ratio.
        This is a placeholder - implement full training loop.
        """
        from models.model import SportsBettingModel, CompositeLoss, BettingBacktester
        
        # Set random seed
        tf.random.set_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Build model
        model = SportsBettingModel(
            num_players=config.num_players,
            embed_dim=config.embed_dim,
            temporal_dim=config.temporal_dim,
            num_gnn_layers=config.num_gnn_layers,
            num_transformer_layers=config.num_transformer_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss
        loss_fn = CompositeLoss(
            lambda_win=config.lambda_win,
            lambda_spread=config.lambda_spread,
            lambda_kelly=config.lambda_kelly,
            lambda_calibration=config.lambda_calibration
        )
        
        # TODO: Implement training loop
        # For now, return mock Sharpe ratio
        # In production: train on data, backtest, return real Sharpe
        
        # Mock: penalize very large models
        param_penalty = model.count_params() / 1e6
        mock_sharpe = 1.0 - param_penalty * 0.1 + np.random.randn() * 0.2
        
        # Report intermediate value for pruning
        trial.report(mock_sharpe, step=1)
        
        # Optuna can prune unpromising trials
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mock_sharpe
    
    def optimize(self) -> ExperimentConfig:
        """Run optimization."""
        print("="*80)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        print(f"  Trials: {self.n_trials}")
        print(f"  Objective: Maximize Sharpe Ratio")
        print("")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.base_config.random_seed),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best config
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"  Best Sharpe Ratio: {best_value:.4f}")
        print(f"  Best Parameters:")
        for k, v in best_params.items():
            print(f"    {k}: {v}")
        
        # Create best config
        self.best_config = ExperimentConfig(**{
            **self.base_config.to_dict(),
            **best_params
        })
        
        return self.best_config
    
    def plot_optimization_history(self, save_path: str = None):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            if save_path:
                plt.savefig(save_path)
                print(f"  Plot saved: {save_path}")
            else:
                plt.show()
        except ImportError:
            print("matplotlib not available for plotting")
    
    def plot_param_importances(self, save_path: str = None):
        """Plot parameter importances."""
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
            if save_path:
                plt.savefig(save_path)
                print(f"  Plot saved: {save_path}")
            else:
                plt.show()
        except ImportError:
            print("matplotlib not available for plotting")


# ============================================================================
# 4. ABLATION STUDIES
# ============================================================================

class AblationStudy:
    """
    Systematically disable components to measure their impact.
    """
    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.results = {}
    
    def run_all_ablations(self) -> Dict[str, Dict]:
        """Run complete ablation study."""
        print("="*80)
        print("ABLATION STUDY")
        print("="*80)
        
        ablations = [
            ('baseline', {}),
            ('no_temporal', {'use_temporal_embeddings': False}),
            ('no_gnn', {'use_gnn': False}),
            ('no_cross_attention', {'use_cross_attention': False}),
            ('no_market_signals', {'use_market_signals': False}),
            ('simple_pooling', {'use_gnn': False, 'use_cross_attention': False}),
            ('no_kelly_loss', {'lambda_kelly': 0.0}),
            ('no_calibration', {'lambda_calibration': 0.0}),
        ]
        
        for name, modifications in ablations:
            print(f"\n--- Running: {name} ---")
            
            # Create modified config
            config = ExperimentConfig(**{
                **self.base_config.to_dict(),
                **modifications,
                'experiment_name': f'ablation_{name}'
            })
            
            # Train and evaluate
            metrics = self._train_and_evaluate(config)
            self.results[name] = metrics
            
            print(f"  Sharpe: {metrics['sharpe']:.3f}")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
            print(f"  Return: {metrics['return']:.2%}")
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _train_and_evaluate(self, config: ExperimentConfig) -> Dict:
        """Train model with config and return metrics."""
        # TODO: Implement full training
        # For now, return mock metrics with some variation
        
        base_sharpe = 1.0
        
        # Penalize based on what's disabled
        if not config.use_temporal_embeddings:
            base_sharpe -= 0.15
        if not config.use_gnn:
            base_sharpe -= 0.2
        if not config.use_cross_attention:
            base_sharpe -= 0.1
        if not config.use_market_signals:
            base_sharpe -= 0.25
        if config.lambda_kelly == 0.0:
            base_sharpe -= 0.05
        
        return {
            'sharpe': base_sharpe + np.random.randn() * 0.05,
            'win_rate': 0.54 + (base_sharpe - 1.0) * 0.02,
            'return': base_sharpe * 0.15,
            'max_drawdown': 0.15 - base_sharpe * 0.03
        }
    
    def _print_summary(self):
        """Print ablation results table."""
        print("\n" + "="*80)
        print("ABLATION RESULTS SUMMARY")
        print("="*80)
        
        df = pd.DataFrame(self.results).T
        df = df.sort_values('sharpe', ascending=False)
        
        print(df.to_string())
        print("\nKey Insights:")
        
        baseline_sharpe = self.results['baseline']['sharpe']
        
        for name, metrics in self.results.items():
            if name == 'baseline':
                continue
            
            impact = baseline_sharpe - metrics['sharpe']
            if impact > 0.1:
                print(f"  ⚠ {name}: -{impact:.3f} Sharpe (HIGH IMPACT)")
            elif impact > 0.05:
                print(f"  • {name}: -{impact:.3f} Sharpe (medium impact)")


# ============================================================================
# 5. MODEL COMPARISON FRAMEWORK
# ============================================================================

class ModelComparison:
    """
    Compare multiple trained models on same test set.
    """
    def __init__(self, test_data):
        self.test_data = test_data
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def add_model(self, name: str, model, config: ExperimentConfig):
        """Add model to comparison."""
        self.models[name] = {
            'model': model,
            'config': config
        }
    
    def evaluate_all(self):
        """Evaluate all models on test data."""
        print("="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        for name, model_info in self.models.items():
            print(f"\n--- Evaluating: {name} ---")
            
            model = model_info['model']
            
            # Get predictions
            predictions = self._get_predictions(model)
            self.predictions[name] = predictions
            
            # Compute metrics
            metrics = self._compute_metrics(predictions)
            self.metrics[name] = metrics
            
            print(f"  Sharpe: {metrics['sharpe']:.3f}")
            print(f"  Brier: {metrics['brier']:.4f}")
            print(f"  Log Loss: {metrics['log_loss']:.4f}")
        
        # Statistical tests
        self._run_statistical_tests()
        
        # Print summary
        self._print_comparison_table()
    
    def _get_predictions(self, model) -> np.ndarray:
        """Get model predictions on test data."""
        # TODO: Implement actual prediction
        return np.random.rand(len(self.test_data))
    
    def _compute_metrics(self, predictions: np.ndarray) -> Dict:
        """Compute evaluation metrics."""
        # TODO: Implement with real targets
        return {
            'sharpe': np.random.randn() * 0.2 + 1.0,
            'brier': np.random.rand() * 0.1 + 0.2,
            'log_loss': np.random.rand() * 0.2 + 0.5,
            'return': np.random.randn() * 0.05 + 0.15
        }
    
    def _run_statistical_tests(self):
        """Run statistical significance tests between models."""
        from scipy import stats
        
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)
        
        model_names = list(self.models.keys())
        
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                # Pairwise t-test on returns
                # TODO: Use actual per-bet returns
                returns1 = np.random.randn(100) * 0.1
                returns2 = np.random.randn(100) * 0.1
                
                t_stat, p_value = stats.ttest_ind(returns1, returns2)
                
                if p_value < 0.05:
                    print(f"  {name1} vs {name2}: p={p_value:.4f} (SIGNIFICANT)")
                else:
                    print(f"  {name1} vs {name2}: p={p_value:.4f}")
    
    def _print_comparison_table(self):
        """Print comparison table."""
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        
        df = pd.DataFrame(self.metrics).T
        df = df.sort_values('sharpe', ascending=False)
        
        print(df.to_string())
        
        best_model = df.index[0]
        print(f"\n✓ Best Model: {best_model}")
        print(f"  Sharpe: {df.loc[best_model, 'sharpe']:.3f}")


# ============================================================================
# 6. QUICK EXPERIMENTS RUNNER
# ============================================================================

def run_quick_experiments():
    """
    Run predefined set of important experiments.
    """
    print("="*80)
    print("QUICK EXPERIMENTS SUITE")
    print("="*80)
    
    tracker = ExperimentTracker()
    
    experiments = [
        ExperimentConfig(
            experiment_name="baseline",
            notes="Simple baseline with all components"
        ),
        ExperimentConfig(
            experiment_name="large_model",
            embed_dim=256,
            num_gnn_layers=3,
            num_transformer_layers=3,
            notes="Larger capacity model"
        ),
        ExperimentConfig(
            experiment_name="aggressive_kelly",
            max_kelly_fraction=0.05,
            min_edge_threshold=0.01,
            lambda_kelly=0.3,
            notes="More aggressive betting strategy"
        ),
        ExperimentConfig(
            experiment_name="market_focused",
            lambda_win=0.5,
            lambda_spread=0.3,
            lambda_kelly=0.5,
            use_market_signals=True,
            notes="Heavy emphasis on market signals and Kelly"
        ),
    ]
    
    results = {}
    
    for config in experiments:
        print(f"\n{'='*80}")
        print(f"Experiment: {config.experiment_name}")
        print(f"{'='*80}")
        
        run_dir = tracker.start_experiment(config)
        
        # TODO: Actual training
        print("  [Training...]")
        
        # Mock results
        sharpe = 1.0 + np.random.randn() * 0.2
        results[config.experiment_name] = {
            'sharpe': sharpe,
            'config': config
        }
        
        tracker.log_backtest_results({'sharpe': sharpe}, 0)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENTS SUMMARY")
    print("="*80)
    
    for name, result in sorted(results.items(), 
                               key=lambda x: x[1]['sharpe'], 
                               reverse=True):
        print(f"  {name}: Sharpe = {result['sharpe']:.3f}")


# ============================================================================
# 7. EXAMPLE USAGE
# ============================================================================

def main():
    """Complete example of experiment framework."""
    print("="*80)
    print("EXPERIMENT FRAMEWORK - COMPLETE DEMO")
    print("="*80)
    
    # 1. Create base configuration
    base_config = ExperimentConfig(
        experiment_name="demo",
        embed_dim=128,
        num_gnn_layers=2
    )
    
    print("\n✓ Base configuration created")
    
    # 2. Initialize tracker
    tracker = ExperimentTracker()
    run_dir = tracker.start_experiment(base_config)
    
    print("✓ Experiment tracking initialized")
    
    # 3. Run hyperparameter optimization (optional)
    print("\n" + "="*80)
    print("Running hyperparameter optimization...")
    print("="*80)
    
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        n_trials=10  # Use more in production
    )
    
    # best_config = optimizer.optimize()
    # optimizer.plot_optimization_history("optuna_history.png")
    
    print("✓ Hyperparameter optimization ready")
    
    # 4. Run ablation study
    print("\n" + "="*80)
    print("Running ablation study...")
    print("="*80)
    
    ablation = AblationStudy(base_config)
    # ablation_results = ablation.run_all_ablations()
    
    print("✓ Ablation study ready")
    
    # 5. Quick experiments
    print("\n" + "="*80)
    print("Running quick experiments...")
    print("="*80)
    
    # run_quick_experiments()
    
    print("✓ Quick experiments ready")
    
    print("\n" + "="*80)
    print("EXPERIMENT FRAMEWORK COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Connect to real data pipeline")
    print("  2. Run full hyperparameter search (50-100 trials)")
    print("  3. Execute ablation study")
    print("  4. Compare best models on hold-out test set")
    print("  5. Deploy winning model to production")
    print("="*80)


if __name__ == "__main__":
    main()