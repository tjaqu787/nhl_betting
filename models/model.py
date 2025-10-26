"""
Elite Sports Betting Model - TensorFlow 2.x Implementation
Includes: GNN, Cross-Attention, Market Modeling, Meta-Learning, Causal Inference
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Tuple, Optional
import tensorflow_probability as tfp

tfd = tfp.distributions
from utils.training import create_training_step
from utils.backtesting import BettingBacktester

# ============================================================================
# 1. PLAYER EMBEDDING WITH TEMPORAL DYNAMICS
# ============================================================================

class TemporalPlayerEncoder(layers.Layer):
    """
    Encodes players with static + temporal embeddings.
    Uses GRU for temporal state evolution.
    """
    def __init__(self, num_players: int, embed_dim: int, temporal_dim: int, 
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_players = num_players
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        
        # Static embeddings (long-term skill)
        self.player_embeddings = layers.Embedding(
            num_players, embed_dim, 
            embeddings_regularizer=keras.regularizers.l2(1e-5),
            name='player_static_embedding'
        )
        
        # Temporal state tracker (form, fatigue)
        self.temporal_gru = layers.GRU(
            temporal_dim, 
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=dropout,
            name='temporal_state_gru'
        )
        
        # Feature encoder
        self.feature_encoder = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(temporal_dim)
        ], name='feature_encoder')
        
        # Fusion layer
        self.fusion = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout)
        ], name='embedding_fusion')
        
    def call(self, player_ids, features, temporal_history=None, training=False):
        """
        Args:
            player_ids: [batch, num_players] - player IDs
            features: [batch, num_players, feat_dim] - current features
            temporal_history: [batch, num_players, seq_len, feat_dim] - historical features
        Returns:
            [batch, num_players, embed_dim]
        """
        # Static embedding
        static_embed = self.player_embeddings(player_ids)  # [B, P, embed_dim]
        
        # Temporal encoding
        if temporal_history is not None:
            batch_size, num_players, seq_len, feat_dim = tf.shape(temporal_history)[0], \
                tf.shape(temporal_history)[1], tf.shape(temporal_history)[2], tf.shape(temporal_history)[3]
            
            # Reshape for GRU: [B*P, seq_len, feat_dim]
            temporal_flat = tf.reshape(temporal_history, [-1, seq_len, feat_dim])
            encoded_features = self.feature_encoder(temporal_flat, training=training)
            
            # Apply GRU
            temporal_state = self.temporal_gru(encoded_features, training=training)
            temporal_state = tf.reshape(temporal_state, 
                                       [batch_size, num_players, self.temporal_dim])
        else:
            # Use current features only
            encoded_features = self.feature_encoder(features, training=training)
            temporal_state = encoded_features
        
        # Fuse static + temporal
        combined = tf.concat([static_embed, temporal_state], axis=-1)
        player_vectors = self.fusion(combined, training=training)
        
        return player_vectors


# ============================================================================
# 2. GRAPH ATTENTION NETWORK FOR LINEUP SYNERGY
# ============================================================================

class GraphAttentionLayer(layers.Layer):
    """
    Multi-head GAT for modeling within-lineup interactions.
    Captures chemistry, role complementarity, spacing.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, 
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = layers.Dense(3 * embed_dim, name='qkv_projection')
        self.edge_encoder = layers.Dense(num_heads, activation='relu', name='edge_encoder')
        self.out_proj = layers.Dense(embed_dim, name='output_projection')
        
        self.dropout = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, x, edge_features=None, mask=None, training=False):
        """
        Args:
            x: [batch, num_nodes, embed_dim] - player embeddings
            edge_features: [batch, num_nodes, num_nodes, edge_dim] - co-play stats
            mask: [batch, num_nodes] - player availability mask
        Returns:
            [batch, num_nodes, embed_dim]
        """
        batch_size = tf.shape(x)[0]
        num_nodes = tf.shape(x)[1]
        
        # Multi-head QKV projection
        qkv = self.qkv_proj(x)  # [B, N, 3*embed_dim]
        qkv = tf.reshape(qkv, [batch_size, num_nodes, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(self.head_dim))
        # [B, H, N, N]
        
        # Add edge features (co-play time, on/off differentials)
        if edge_features is not None:
            edge_bias = self.edge_encoder(edge_features)  # [B, N, N, H]
            edge_bias = tf.transpose(edge_bias, [0, 3, 1, 2])  # [B, H, N, N]
            scores = scores + edge_bias
        
        # Apply mask
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = mask[:, None, None, :]  # [B, 1, 1, N]
            scores = scores + (1.0 - mask) * -1e9
        
        # Softmax and dropout
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        
        # Apply attention to values
        out = tf.matmul(attn_weights, v)  # [B, H, N, head_dim]
        out = tf.transpose(out, [0, 2, 1, 3])  # [B, N, H, head_dim]
        out = tf.reshape(out, [batch_size, num_nodes, self.embed_dim])
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out, training=training)
        
        # Residual connection + layer norm
        return self.layer_norm(x + out)


class LineupGNN(layers.Layer):
    """Stack of GAT layers for deep lineup interaction modeling."""
    def __init__(self, embed_dim: int, num_layers: int = 2, 
                 num_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.gat_layers = [
            GraphAttentionLayer(embed_dim, num_heads, dropout, name=f'gat_{i}')
            for i in range(num_layers)
        ]
        
        # FFN after each GAT
        self.ffn_layers = [
            keras.Sequential([
                layers.Dense(embed_dim * 4, activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
                layers.LayerNormalization(epsilon=1e-6)
            ], name=f'ffn_{i}')
            for i in range(num_layers)
        ]
    
    def call(self, x, edge_features=None, mask=None, training=False):
        for gat, ffn in zip(self.gat_layers, self.ffn_layers):
            x = gat(x, edge_features, mask, training)
            x = ffn(x, training=training) + x  # Residual
        return x


# ============================================================================
# 3. CROSS-ATTENTION TRANSFORMER FOR MATCHUP MODELING
# ============================================================================

class CrossAttentionMatchup(layers.Layer):
    """
    Models explicit home vs away matchup effects.
    Each team's players attend to opposing team's players.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = layers.Dense(embed_dim, name='query_proj')
        self.kv_proj = layers.Dense(2 * embed_dim, name='kv_proj')
        self.out_proj = layers.Dense(embed_dim, name='out_proj')
        
        self.dropout = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, query, key_value, query_mask=None, kv_mask=None, training=False):
        """
        Args:
            query: [batch, num_query, embed_dim] - team A players
            key_value: [batch, num_kv, embed_dim] - team B players
            query_mask, kv_mask: availability masks
        Returns:
            [batch, num_query, embed_dim]
        """
        batch_size = tf.shape(query)[0]
        num_query = tf.shape(query)[1]
        num_kv = tf.shape(key_value)[1]
        
        # Project
        q = self.q_proj(query)  # [B, Q, embed_dim]
        kv = self.kv_proj(key_value)  # [B, KV, 2*embed_dim]
        
        # Reshape for multi-head
        q = tf.reshape(q, [batch_size, num_query, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])  # [B, H, Q, head_dim]
        
        kv = tf.reshape(kv, [batch_size, num_kv, 2, self.num_heads, self.head_dim])
        kv = tf.transpose(kv, [2, 0, 3, 1, 4])  # [2, B, H, KV, head_dim]
        k, v = kv[0], kv[1]
        
        # Attention scores
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(self.head_dim))
        
        # Apply kv_mask
        if kv_mask is not None:
            kv_mask = tf.cast(kv_mask, tf.float32)
            kv_mask = kv_mask[:, None, None, :]  # [B, 1, 1, KV]
            scores = scores + (1.0 - kv_mask) * -1e9
        
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        
        # Apply to values
        out = tf.matmul(attn_weights, v)  # [B, H, Q, head_dim]
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch_size, num_query, self.embed_dim])
        
        out = self.out_proj(out)
        out = self.dropout(out, training=training)
        
        return self.layer_norm(query + out)


# ============================================================================
# 4. ATTENTION POOLING (EMPHASIZE STAR PLAYERS)
# ============================================================================

class AttentionPooling(layers.Layer):
    """
    Weighted pooling that learns to emphasize important players.
    Better than mean pooling for capturing star effects.
    """
    def __init__(self, embed_dim: int, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.attention_weights = layers.Dense(1, name='attention_scorer')
        self.temperature = temperature
        
    def call(self, x, mask=None, training=False):
        """
        Args:
            x: [batch, num_players, embed_dim]
            mask: [batch, num_players]
        Returns:
            [batch, embed_dim]
        """
        # Compute attention scores
        scores = self.attention_weights(x) / self.temperature  # [B, P, 1]
        
        if mask is not None:
            mask = tf.cast(mask, tf.float32)[:, :, None]
            scores = scores + (1.0 - mask) * -1e9
        
        attn = tf.nn.softmax(scores, axis=1)  # [B, P, 1]
        pooled = tf.reduce_sum(x * attn, axis=1)  # [B, embed_dim]
        
        return pooled


# ============================================================================
# 5. MARKET DISAGREEMENT ANALYZER (200 IQ Enhancement #2 - CORRECTED)
# ============================================================================

class MarketDisagreementAnalyzer(layers.Layer):
    """
    CRITICAL FIX: Don't use bookmaker lines as inputs!
    Instead, analyze market STRUCTURE to detect inefficiencies:
    - Line movement patterns (sharp vs public money)
    - Cross-book disagreement (arbitrage opportunities)
    - Historical market errors on similar games
    
    The model makes predictions INDEPENDENTLY, then we compare to market.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        # Encode market METADATA (not the actual lines!)
        self.metadata_encoder = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim // 2)
        ], name='market_metadata_encoder')
        
    def call(self, market_metadata, training=False):
        """
        Args:
            market_metadata: [batch, metadata_dim] - NOT actual spreads/odds!
            Contains only:
              - Line movement velocity (how fast line moved)
              - Cross-book disagreement (variance across books)
              - Public betting % (is this sharp or square money?)
              - Time until game (urgency)
              - Historical market accuracy on similar games
              - Sharp money indicators (reverse line movement)
        Returns:
            [batch, embed_dim // 2] - market structure features
        """
        encoded = self.metadata_encoder(market_metadata, training=training)
        return encoded
    
    def compute_edge(self, model_prediction, market_line):
        """
        Compare model prediction to market AFTER prediction is made.
        This happens outside the model, during bet decision.
        
        Args:
            model_prediction: Your independent model's win probability
            market_line: Bookmaker's implied probability (from odds)
        Returns:
            edge: How much you disagree (positive = bet opportunity)
        """
        # Simple edge calculation
        edge = model_prediction - market_line
        return edge


# ============================================================================
# 6. META-LEARNING FOR COLD START (200 IQ Enhancement #3)
# ============================================================================

class MAMLPlayerInitializer(keras.Model):
    """
    Model-Agnostic Meta-Learning for few-shot player adaptation.
    Learns good initialization for new/rookie players.
    """
    def __init__(self, embed_dim: int, num_adaptation_steps: int = 5, 
                 inner_lr: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_adaptation_steps = num_adaptation_steps
        self.inner_lr = inner_lr
        
        # Meta-initialization network
        self.meta_init = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(embed_dim)
        ], name='meta_initializer')
        
    def adapt(self, support_features, support_labels, query_features):
        """
        Few-shot adaptation for a new player.
        Args:
            support_features: [num_support, feat_dim] - scouting/college stats
            support_labels: [num_support, target_dim] - early performance
            query_features: [num_query, feat_dim] - features to predict
        Returns:
            adapted_embeddings: [num_query, embed_dim]
        """
        # Initialize from meta-learned prior
        init_weights = self.meta_init(support_features)
        
        # Inner loop: gradient descent on support set
        adapted_weights = init_weights
        for _ in range(self.num_adaptation_steps):
            with tf.GradientTape() as tape:
                tape.watch(adapted_weights)
                predictions = self._predict_with_weights(
                    support_features, adapted_weights
                )
                loss = tf.reduce_mean(tf.square(predictions - support_labels))
            
            grads = tape.gradient(loss, adapted_weights)
            adapted_weights = adapted_weights - self.inner_lr * grads
        
        # Apply to query
        return self._predict_with_weights(query_features, adapted_weights)
    
    def _predict_with_weights(self, x, weights):
        # Simple linear prediction for illustration
        return tf.matmul(x, weights, transpose_b=True)


# ============================================================================
# 7. GAME STATE EVOLUTION FOR LIVE BETTING (200 IQ Enhancement #1)
# ============================================================================

class GameStateTransformer(layers.Layer):
    """
    Models sequential game state evolution for live betting.
    Processes sequences of: [score, time, possessions, momentum, lineups]
    """
    def __init__(self, state_dim: int, num_layers: int = 3, 
                 num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.state_encoder = layers.Dense(state_dim, name='state_encoder')
        
        # Transformer encoder for temporal dynamics
        self.transformer_layers = [
            layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=state_dim // num_heads,
                dropout=dropout,
                name=f'mha_{i}'
            )
            for i in range(num_layers)
        ]
        
        self.ffn_layers = [
            keras.Sequential([
                layers.Dense(state_dim * 4, activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(state_dim),
                layers.LayerNormalization()
            ], name=f'ffn_{i}')
            for i in range(num_layers)
        ]
        
        self.norm_layers = [
            layers.LayerNormalization(name=f'norm_{i}')
            for i in range(num_layers)
        ]
        
        # Positional encoding
        self.pos_encoding = self.add_weight(
            shape=(1, 500, state_dim),  # max 500 time steps
            initializer='glorot_uniform',
            trainable=True,
            name='positional_encoding'
        )
    
    def call(self, game_states, training=False):
        """
        Args:
            game_states: [batch, seq_len, raw_state_dim]
        Returns:
            [batch, seq_len, state_dim] - evolved state representations
        """
        seq_len = tf.shape(game_states)[1]
        
        # Encode states
        x = self.state_encoder(game_states)  # [B, T, state_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer layers
        for mha, ffn, norm in zip(self.transformer_layers, 
                                   self.ffn_layers, 
                                   self.norm_layers):
            # Self-attention
            attn_out = mha(x, x, training=training)
            x = norm(x + attn_out)
            
            # FFN
            ffn_out = ffn(x, training=training)
            x = x + ffn_out
        
        return x


# ============================================================================
# 8. CAUSAL PLAYER EFFECT ESTIMATOR (200 IQ Enhancement #4)
# ============================================================================

class DoublMLPlayerEffects(layers.Layer):
    """
    Double Machine Learning for causal player effect estimation.
    Debiases on-off court differentials using Neyman orthogonalization.
    """
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        
        # Nuisance function: E[Y | X, W] - outcome model
        self.outcome_model = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ], name='outcome_nuisance')
        
        # Nuisance function: E[D | X, W] - treatment model (propensity)
        self.treatment_model = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ], name='treatment_nuisance')
        
        # Final effect estimator (on residualized quantities)
        self.effect_estimator = layers.Dense(1, name='causal_effect')
    
    def fit_nuisance(self, confounders, treatment, outcome, training=True):
        """
        First stage: fit nuisance functions.
        Args:
            confounders: [N, confounder_dim] - lineup, opponent, context
            treatment: [N, 1] - player on court (binary)
            outcome: [N, 1] - point differential
        """
        # Fit outcome model
        y_pred = self.outcome_model(confounders, training=training)
        outcome_loss = tf.reduce_mean(tf.square(outcome - y_pred))
        
        # Fit treatment model
        d_pred = self.treatment_model(confounders, training=training)
        treatment_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(treatment, d_pred)
        )
        
        return outcome_loss + treatment_loss
    
    def estimate_effect(self, confounders, treatment, outcome, player_features):
        """
        Second stage: estimate causal effect on residualized quantities.
        Args:
            player_features: [N, feat_dim] - player-specific features
        Returns:
            [N, 1] - estimated causal effect
        """
        # Compute residuals (orthogonalization)
        y_pred = self.outcome_model(confounders, training=False)
        d_pred = self.treatment_model(confounders, training=False)
        
        y_resid = outcome - y_pred
        d_resid = treatment - d_pred
        
        # Estimate effect on residuals
        effect = self.effect_estimator(player_features)
        
        # Orthogonal score
        score = y_resid - effect * d_resid
        
        return effect, score


# ============================================================================
# 9. MAIN MODEL: PUTTING IT ALL TOGETHER
# ============================================================================

class SportsBettingModel(keras.Model):
    """
    Complete architecture with all 200 IQ enhancements.
    """
    def __init__(self, 
                 num_players: int,
                 embed_dim: int = 128,
                 temporal_dim: int = 64,
                 num_gnn_layers: int = 2,
                 num_transformer_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        
        # Component 1: Player embeddings
        self.player_encoder = TemporalPlayerEncoder(
            num_players, embed_dim, temporal_dim, dropout
        )
        
        # Component 2: Lineup GNN
        self.lineup_gnn_home = LineupGNN(
            embed_dim, num_gnn_layers, num_heads, dropout
        )
        self.lineup_gnn_away = LineupGNN(
            embed_dim, num_gnn_layers, num_heads, dropout
        )
        
        # Component 3: Cross-attention for matchups
        self.cross_attn_home = CrossAttentionMatchup(
            embed_dim, num_heads, dropout
        )
        self.cross_attn_away = CrossAttentionMatchup(
            embed_dim, num_heads, dropout
        )
        
        # Component 4: Attention pooling
        self.pool_home = AttentionPooling(embed_dim)
        self.pool_away = AttentionPooling(embed_dim)
        
        # Component 5: Market metadata analyzer (200 IQ #2 - CORRECTED)
        self.market_analyzer = MarketDisagreementAnalyzer(embed_dim, dropout)
        
        # Component 6: Game state transformer (200 IQ #1)
        self.game_state_model = GameStateTransformer(
            embed_dim, num_transformer_layers, num_heads, dropout
        )
        
        # Component 7: Causal estimator (200 IQ #4)
        self.causal_estimator = DoublMLPlayerEffects(embed_dim)
        
        # Final prediction heads
        self.context_fusion = keras.Sequential([
            layers.Dense(embed_dim * 2, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ], name='context_fusion')
        
        # Win probability head
        self.win_head = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(64, activation='gelu'),
            layers.Dense(1, activation='sigmoid')
        ], name='win_probability')
        
        # Spread distribution head (MDN with 2 components)
        self.spread_mu_head = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(2)  # 2 mixture components
        ], name='spread_mean')
        
        self.spread_sigma_head = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(2, activation='softplus')  # positive sigma
        ], name='spread_std')
        
        self.spread_pi_head = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(2, activation='softmax')  # mixture weights
        ], name='spread_mixture_weights')
    
    def call(self, inputs, training=False):
        """
        Args:
            inputs: dict with keys:
                - home_player_ids: [batch, num_home]
                - away_player_ids: [batch, num_away]
                - home_features: [batch, num_home, feat_dim]
                - away_features: [batch, num_away, feat_dim]
                - home_temporal_history: [batch, num_home, seq_len, feat_dim]
                - away_temporal_history: [batch, num_away, seq_len, feat_dim]
                - home_edge_features: [batch, num_home, num_home, edge_dim]
                - away_edge_features: [batch, num_away, num_away, edge_dim]
                - market_features: [batch, market_dim]
                - context_features: [batch, context_dim]
                - game_states: [batch, game_seq_len, state_dim] (optional, for live)
        Returns:
            dict with 'win_prob', 'spread_dist', 'causal_effects'
        """
        # Encode players with temporal dynamics
        home_players = self.player_encoder(
            inputs['home_player_ids'],
            inputs['home_features'],
            inputs.get('home_temporal_history'),
            training=training
        )
        
        away_players = self.player_encoder(
            inputs['away_player_ids'],
            inputs['away_features'],
            inputs.get('away_temporal_history'),
            training=training
        )
        
        # GNN: within-lineup synergy
        home_players = self.lineup_gnn_home(
            home_players,
            inputs.get('home_edge_features'),
            training=training
        )
        
        away_players = self.lineup_gnn_away(
            away_players,
            inputs.get('away_edge_features'),
            training=training
        )
        
        # Cross-attention: matchup effects
        home_players = self.cross_attn_home(
            home_players, away_players, training=training
        )
        away_players = self.cross_attn_away(
            away_players, home_players, training=training
        )
        
        # Pool to team representations
        home_team = self.pool_home(home_players, training=training)
        away_team = self.pool_away(away_players, training=training)
        
        # Encode market metadata (NOT actual lines!)
        market_metadata = self.market_analyzer(
            inputs['market_metadata'], training=training
        )
        
        # Combine all signals (market metadata only provides context about market structure)
        combined = tf.concat([
            home_team, 
            away_team, 
            inputs['context_features'],
            market_metadata  # Only structural info, not anchoring lines
        ], axis=-1)
        
        fused = self.context_fusion(combined, training=training)
        
        # Predictions
        win_prob = self.win_head(fused, training=training)
        
        # MDN for spread
        spread_mu = self.spread_mu_head(fused, training=training)
        spread_sigma = self.spread_sigma_head(fused, training=training) + 1e-6
        spread_pi = self.spread_pi_head(fused, training=training)
        
        spread_dist = {
            'mu': spread_mu,
            'sigma': spread_sigma,
            'pi': spread_pi
        }
        
        outputs = {
            'win_prob': win_prob,
            'spread_dist': spread_dist,
            'home_team_embedding': home_team,
            'away_team_embedding': away_team
        }
        
        # Live betting: game state evolution (200 IQ #1)
        if 'game_states' in inputs and inputs['game_states'] is not None:
            game_state_evolved = self.game_state_model(
                inputs['game_states'], training=training
            )
            # Use final state for live prediction adjustment
            current_state = game_state_evolved[:, -1, :]
            
            # Adjust predictions based on live game state
            live_adjustment = layers.Dense(1, activation='tanh')(current_state)
            outputs['win_prob_live'] = tf.clip_by_value(
                win_prob + 0.1 * live_adjustment, 0.0, 1.0
            )
            outputs['game_state_embedding'] = current_state
        
        return outputs
    
    def compute_causal_effects(self, 
                               player_features,
                               confounders,
                               treatment,
                               outcome,
                               training=False):
        """
        Estimate causal player effects using Double ML (200 IQ #4).
        Use this in a separate training loop or evaluation.
        """
        return self.causal_estimator.estimate_effect(
            confounders, treatment, outcome, player_features
        )


# ============================================================================
# 10. LOSS FUNCTIONS
# ============================================================================

def mixture_density_loss(y_true, mu, sigma, pi):
    """
    Negative log-likelihood for Gaussian Mixture Model.
    Handles heavy tails in point differentials.
    """
    # y_true: [batch, 1]
    # mu, sigma, pi: [batch, num_components]
    
    y_true = tf.expand_dims(y_true, -1)  # [batch, 1, 1]
    mu = tf.expand_dims(mu, 1)  # [batch, 1, num_components]
    sigma = tf.expand_dims(sigma, 1)
    pi = tf.expand_dims(pi, 1)
    
    # Gaussian PDF for each component
    normal_dist = tfd.Normal(loc=mu, scale=sigma)
    component_log_probs = normal_dist.log_prob(y_true)  # [batch, 1, K]
    
    # Weight by mixture coefficients
    log_pi = tf.math.log(pi + 1e-8)
    log_weighted = component_log_probs + log_pi
    
    # Log-sum-exp for numerical stability
    log_likelihood = tf.reduce_logsumexp(log_weighted, axis=-1)
    
    return -tf.reduce_mean(log_likelihood)


def kelly_utility_loss(y_true, win_prob_pred, odds, bankroll_fraction=0.02):
    """
    Kelly criterion-based utility loss.
    Optimizes for expected log bankroll growth.
    
    Args:
        y_true: [batch, 1] - actual outcome (0 or 1)
        win_prob_pred: [batch, 1] - predicted win probability
        odds: [batch, 1] - decimal odds from bookmaker
        bankroll_fraction: max fraction to bet (cap Kelly for safety)
    """
    # Kelly fraction: f = (p * odds - 1) / (odds - 1)
    edge = win_prob_pred * odds - 1.0
    kelly_frac = edge / (odds - 1.0)
    kelly_frac = tf.clip_by_value(kelly_frac, 0.0, bankroll_fraction)
    
    # Expected log return
    # If win: log(1 + f * (odds - 1))
    # If lose: log(1 - f)
    win_return = tf.math.log(1.0 + kelly_frac * (odds - 1.0) + 1e-8)
    lose_return = tf.math.log(1.0 - kelly_frac + 1e-8)
    
    expected_log_return = y_true * win_return + (1 - y_true) * lose_return
    
    # Maximize expected log return = minimize negative
    return -tf.reduce_mean(expected_log_return)


def calibration_loss(y_true, y_pred, num_bins=10):
    """
    Expected Calibration Error (ECE).
    Ensures predicted probabilities match empirical frequencies.
    """
    # Bin predictions
    bin_boundaries = tf.linspace(0.0, 1.0, num_bins + 1)
    
    ece = 0.0
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find predictions in this bin
        in_bin = tf.logical_and(
            y_pred >= bin_lower,
            y_pred < bin_upper
        )
        
        bin_count = tf.reduce_sum(tf.cast(in_bin, tf.float32))
        
        if bin_count > 0:
            # Average prediction in bin
            avg_pred = tf.reduce_sum(
                tf.where(in_bin, y_pred, 0.0)
            ) / bin_count
            
            # Average true outcome in bin
            avg_true = tf.reduce_sum(
                tf.where(in_bin, y_true, 0.0)
            ) / bin_count
            
            # Weighted calibration error
            ece += (bin_count / tf.cast(tf.shape(y_pred)[0], tf.float32)) * \
                   tf.abs(avg_pred - avg_true)
    
    return ece


class CompositeLoss(keras.losses.Loss):
    """
    Multi-task loss combining all objectives.
    """
    def __init__(self, 
                 lambda_win=1.0,
                 lambda_spread=0.5,
                 lambda_kelly=0.1,
                 lambda_calibration=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.lambda_win = lambda_win
        self.lambda_spread = lambda_spread
        self.lambda_kelly = lambda_kelly
        self.lambda_calibration = lambda_calibration
    
    def call(self, y_true, y_pred):
        """
        Args:
            y_true: dict with 'win', 'spread', 'odds'
            y_pred: dict with 'win_prob', 'spread_dist'
        """
        # Win probability loss
        win_loss = keras.losses.binary_crossentropy(
            y_true['win'], y_pred['win_prob']
        )
        
        # Spread MDN loss
        spread_loss = mixture_density_loss(
            y_true['spread'],
            y_pred['spread_dist']['mu'],
            y_pred['spread_dist']['sigma'],
            y_pred['spread_dist']['pi']
        )
        
        # Kelly utility loss (if odds available)
        if 'odds' in y_true:
            kelly_loss = kelly_utility_loss(
                y_true['win'],
                y_pred['win_prob'],
                y_true['odds']
            )
        else:
            kelly_loss = 0.0
        
        # Calibration loss
        calib_loss = calibration_loss(y_true['win'], y_pred['win_prob'])
        
        total_loss = (
            self.lambda_win * win_loss +
            self.lambda_spread * spread_loss +
            self.lambda_kelly * kelly_loss +
            self.lambda_calibration * calib_loss
        )
        
        return total_loss



# ============================================================================
# 13. COUNTERFACTUAL LINEUP ANALYZER
# ============================================================================

class CounterfactualLineupAnalyzer:
    """
    Answer questions like: "What if Player X sits?"
    """
    def __init__(self, model):
        self.model = model
    
    def predict_lineup_swap(self, 
                           base_inputs,
                           player_out_idx,
                           player_in_id,
                           player_in_features,
                           team='home'):
        """
        Swap a player and predict new outcome.
        """
        inputs = base_inputs.copy()
        
        if team == 'home':
            # Replace player at index
            new_ids = inputs['home_player_ids'].copy()
            new_ids[0, player_out_idx] = player_in_id
            inputs['home_player_ids'] = new_ids
            
            new_features = inputs['home_features'].copy()
            new_features[0, player_out_idx, :] = player_in_features
            inputs['home_features'] = new_features
        else:
            new_ids = inputs['away_player_ids'].copy()
            new_ids[0, player_out_idx] = player_in_id
            inputs['away_player_ids'] = new_ids
            
            new_features = inputs['away_features'].copy()
            new_features[0, player_out_idx, :] = player_in_features
            inputs['away_features'] = new_features
        
        # Predict with swapped lineup
        predictions = self.model(inputs, training=False)
        
        return predictions
    
    def compute_player_value(self, base_inputs, player_idx, team='home'):
        """
        Estimate player's value by replacing with average player.
        """
        # Get prediction with current lineup
        base_pred = self.model(base_inputs, training=False)
        base_win_prob = base_pred['win_prob'][0, 0].numpy()
        
        # Replace with average player (use position-specific average)
        avg_player_id = 0  # Represents "average player"
        avg_features = np.zeros_like(
            base_inputs[f'{team}_features'][0, player_idx, :]
        )
        
        swap_pred = self.predict_lineup_swap(
            base_inputs,
            player_idx,
            avg_player_id,
            avg_features,
            team=team
        )
        swap_win_prob = swap_pred['win_prob'][0, 0].numpy()
        
        # Value = change in win probability
        player_value = base_win_prob - swap_win_prob
        
        return player_value


# ============================================================================
# 14. EXAMPLE USAGE & TRAINING SCRIPT
# ============================================================================

def build_model_and_train():
    """
    Complete training pipeline example.
    """
    # Hyperparameters
    NUM_PLAYERS = 500  # Total players in league
    EMBED_DIM = 128
    TEMPORAL_DIM = 64
    NUM_GNN_LAYERS = 2
    NUM_TRANSFORMER_LAYERS = 2
    NUM_HEADS = 4
    DROPOUT = 0.1
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    
    # Build model
    model = SportsBettingModel(
        num_players=NUM_PLAYERS,
        embed_dim=EMBED_DIM,
        temporal_dim=TEMPORAL_DIM,
        num_gnn_layers=NUM_GNN_LAYERS,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    )
    
    # Optimizer with warmup
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=10000,
        alpha=0.1
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )
    
    # Loss function
    loss_fn = CompositeLoss(
        lambda_win=1.0,
        lambda_spread=0.5,
        lambda_kelly=0.1,
        lambda_calibration=0.2
    )

    # Create training step
    train_step_fn = create_training_step(model, optimizer, loss_fn)
    
    # Backtester
    backtester = BettingBacktester(
        initial_bankroll=10000.0,
        max_kelly_fraction=0.02,
        min_edge_threshold=0.02
    )
    
    print("Model built successfully!")
    
    return model, optimizer, loss_fn, train_step_fn, backtester



# ============================================================================
# READY TO USE - EXAMPLE INSTANTIATION
# ============================================================================

if __name__ == "__main__":
    # Build model
    model, optimizer, loss_fn, train_step_fn, backtester = build_model_and_train()