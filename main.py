from models.model import EliteSportsBettingModel, CompositeLoss
from data.data_preprocessor import SportsDataPreprocessor, TrainingOrchestrator
from models.inference import ExperimentConfig, ExperimentTracker

# 1. Create configuration
config = ExperimentConfig(
    num_players=500,
    embed_dim=128,
    num_gnn_layers=2,
    learning_rate=3e-4
)

# 2. Build model
model = EliteSportsBettingModel(
    num_players=config.num_players,
    embed_dim=config.embed_dim,
    temporal_dim=config.temporal_dim,
    num_gnn_layers=config.num_gnn_layers,
    num_transformer_layers=config.num_transformer_layers,
    num_heads=config.num_heads,
    dropout=config.dropout
)

# 3. Setup training
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay
)

loss_fn = CompositeLoss(
    lambda_win=1.0,
    lambda_spread=0.5,
    lambda_kelly=0.1,
    lambda_calibration=0.2
)

# 4. Train with walk-forward validation
preprocessor = SportsDataPreprocessor()
orchestrator = TrainingOrchestrator(model, optimizer, loss_fn, backtester)

# orchestrator.train_walk_