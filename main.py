import os
import sys
from datetime import datetime

MODE = 1  # 1=Train, 2=Backtest, 3=Inference

def main():
    print("\n" + "="*80)
    print("NHL BETTING MODEL")
    print("="*80)
    print(f"Mode: {MODE} ({'TRAIN' if MODE == 1 else 'BACKTEST' if MODE == 2 else 'INFERENCE'})")
    print(f"Timestamp: {datetime.now()}")
    print("="*80 + "\n")
    
    if MODE == 1:
        run_training()
    elif MODE == 2:
        run_backtesting()
    elif MODE == 3:
        run_inference()
    else:
        print(f"ERROR: Invalid MODE={MODE}. Must be 1 (Train), 2 (Backtest), or 3 (Inference)")
        sys.exit(1)


def run_training():
    """Mode 1: Train the model"""
    print("="*80)
    print("MODE 1: TRAINING")
    print("="*80 + "\n")
    
    try:
        from data.data_preprocessor import SportsDataPreprocessor
        from utils.training import TrainingOrchestrator
        from models.model import build_model_and_train
        import pandas as pd
        
        print("Step 1: Initialize components")
        print("-" * 40)
        
        # Build model
        model, optimizer, loss_fn, train_step_fn, backtester = build_model_and_train()
        print("✓ Model built")
        
        # Initialize preprocessor
        preprocessor = SportsDataPreprocessor()
        print("✓ Preprocessor initialized")
        
        # Initialize training orchestrator
        orchestrator = TrainingOrchestrator(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_step_fn=train_step_fn,
            backtester=backtester
        )
        print("✓ Training orchestrator ready")
        
        print("\nStep 2: Load data")
        print("-" * 40)
        print("TODO: Load from database - data/nhl_data.db")
        print("For now using mock data...\n")
        
        # Mock data for testing
        game_logs = pd.DataFrame({
            'game_id': range(100),
            'game_date': pd.date_range('2024-01-01', periods=100),
            'home_team': ['TOR'] * 100,
            'away_team': ['MTL'] * 100,
            'home_score': [3] * 100,
            'away_score': [2] * 100,
        })
        
        preprocessor.fit_player_vocab(game_logs)
        print(f"✓ Loaded {len(game_logs)} games")
        
        print("\nStep 3: Train model")
        print("-" * 40)
        print("Starting walk-forward training...")
        print("(Set epochs_per_window=1 for quick debug)\n")
        
        orchestrator.train_walk_forward(
            preprocessor=preprocessor,
            game_logs=game_logs,
            train_window_months=6,
            val_window_months=1,
            epochs_per_window=1  # Set to 1 for quick debug
        )
        
        print("✓ Training complete (currently stubbed)")
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR in training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_backtesting():
    """Mode 2: Run backtests"""
    print("="*80)
    print("MODE 2: BACKTESTING")
    print("="*80 + "\n")
    
    try:
        from utils.backtesting import Backtester
        from models.model import build_model_and_train
        import pandas as pd
        import numpy as np
        
        print("Step 1: Load trained model")
        print("-" * 40)
        
        model, optimizer, loss_fn, train_step_fn, backtester = build_model_and_train()
        
        # TODO: Load trained weights
        # model.load_weights("checkpoints/best_model")
        print("✓ Model loaded (using fresh weights for now)")
        
        print("\nStep 2: Prepare backtest data")
        print("-" * 40)
        print("TODO: Load validation/test data from database\n")
        
        # Mock predictions for testing
        predictions_df = pd.DataFrame({
            'game_id': range(50),
            'game_date': pd.date_range('2024-06-01', periods=50),
            'home_team': ['TOR'] * 50,
            'away_team': ['MTL'] * 50,
            'predicted_home_win_prob': np.random.uniform(0.4, 0.6, 50),
            'predicted_spread': np.random.uniform(-5, 5, 50),
            'market_home_win_prob': np.random.uniform(0.4, 0.6, 50),
            'market_spread': np.random.uniform(-5, 5, 50),
            'actual_home_win': np.random.randint(0, 2, 50),
            'actual_spread': np.random.uniform(-10, 10, 50),
        })
        
        print(f"✓ Loaded {len(predictions_df)} predictions")
        
        print("\nStep 3: Run backtest")
        print("-" * 40)
        
        results = backtester.run_backtest(
            predictions=predictions_df,
            kelly_fraction=0.25,
            min_edge_threshold=0.02
        )
        
        print("\nBacktest Results:")
        print("-" * 40)
        print(f"Total Bets: {results.get('total_bets', 0)}")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"ROI: {results.get('roi', 0):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        
        print("\n" + "="*80)
        print("BACKTESTING COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR in backtesting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_inference():
    """Mode 3: Run inference on today's games"""
    print("="*80)
    print("MODE 3: INFERENCE")
    print("="*80 + "\n")
    
    try:
        from models.inference import ExperimentTracker
        from models.model import build_model_and_train
        from data.data_downloader import NHLDataScraper
        import pandas as pd
        
        print("Step 1: Load trained model")
        print("-" * 40)
        
        model, optimizer, loss_fn, train_step_fn, backtester = build_model_and_train()
        
        # TODO: Load trained weights
        # model.load_weights("checkpoints/best_model")
        print("✓ Model loaded (using fresh weights for now)")
        
        print("\nStep 2: Fetch today's games")
        print("-" * 40)
        
        scraper = NHLDataScraper()
        # TODO: Fetch today's schedule
        # today_games = scraper.fetch_schedule(date=datetime.now().strftime("%Y-%m-%d"))
        
        print("TODO: Fetch today's NHL schedule from API")
        print("For now using mock data...\n")
        
        # Mock today's games
        today_games = pd.DataFrame({
            'game_id': [2024020100, 2024020101],
            'home_team': ['TOR', 'BOS'],
            'away_team': ['MTL', 'NYR'],
            'game_time': ['19:00', '19:30'],
        })
        
        print(f"✓ Found {len(today_games)} games today")
        
        print("\nStep 3: Generate predictions")
        print("-" * 40)
        
        for idx, game in today_games.iterrows():
            print(f"\nGame {idx + 1}: {game['away_team']} @ {game['home_team']}")
            print(f"  Time: {game['game_time']}")
            
            # TODO: Prepare features and run inference
            # features = preprocessor.prepare_game_features(game)
            # prediction = model.predict(features)
            
            print(f"  TODO: Generate prediction")
            print(f"  Predicted home win prob: 0.XX")
            print(f"  Predicted spread: ±X.X")
            print(f"  Edge vs market: +X.X%")
            print(f"  Bet recommendation: [PASS/BET HOME/BET AWAY]")
        
        print("\n" + "="*80)
        print("INFERENCE COMPLETE")
        print("="*80)
        
        scraper.close()
        
    except Exception as e:
        print(f"\n✗ ERROR in inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()