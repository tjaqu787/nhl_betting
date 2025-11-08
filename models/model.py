"""
NHL Betting Model - Clean Dual Path Architecture
Per-Player Embeddings → GNN Path → Outputs
                      ↘ Bayes Path → Outputs

Key principle: Share embeddings, split processing paths
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np
from typing import Dict, List, Optional, Tuple
import sqlite3

tfd = tfp.distributions


# ============================================================================
# SHARED: PER-PLAYER EMBEDDING LAYER
# ============================================================================

class SkaterEmbeddingLayer(layers.Layer):
    """
    Skater embeddings capturing:
    - Base skill (shooting, skating, IQ)
    - Playing style (physical, finesse, defensive)
    - Positional tendencies (C, L, R, D)
    - Current form
    
    These embeddings are SHARED between GNN and Bayesian paths.
    """
    def __init__(self, num_skaters: int, embed_dim: int, 
                 temporal_dim: int = 64, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        
        # Base skater embeddings (skill + style)
        self.skater_embed = layers.Embedding(
            num_skaters,
            embed_dim,
            embeddings_regularizer=keras.regularizers.l2(1e-5),
            name='skater_base_embeddings'
        )
        
        # Position embeddings (C, L, R, D)
        self.position_embed = layers.Embedding(
            4,  # Center, Left Wing, Right Wing, Defense
            embed_dim // 4,
            name='position_embeddings'
        )
        
        # Current form encoder (from recent game features)
        self.form_encoder = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(temporal_dim)
        ], name='skater_form_encoder')
        
        # Temporal tracker (recent game sequence)
        self.temporal_gru = layers.GRU(
            temporal_dim,
            return_sequences=False,
            dropout=dropout,
            name='skater_temporal_tracker'
        )
        
        # Fusion: base + position + form → final embedding
        self.fusion = keras.Sequential([
            layers.Dense(embed_dim * 2, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ], name='skater_embedding_fusion')
    
    def call(self, skater_ids, positions, current_features, 
             temporal_history=None, training=False):
        """
        Args:
            skater_ids: [batch, num_skaters]
            positions: [batch, num_skaters] (0=C, 1=L, 2=R, 3=D)
            current_features: [batch, num_skaters, feat_dim]
            temporal_history: [batch, num_skaters, seq_len, feat_dim]
        
        Returns:
            [batch, num_skaters, embed_dim] - skater embeddings
        """
        batch_size = tf.shape(skater_ids)[0]
        num_skaters = tf.shape(skater_ids)[1]
        
        # Base embeddings
        base = self.skater_embed(skater_ids)  # [B, S, E]
        
        # Position embeddings
        pos = self.position_embed(positions)  # [B, S, E//4]
        pos_padded = tf.pad(pos, [[0, 0], [0, 0], [0, self.embed_dim - self.embed_dim // 4]])
        
        # Current form
        form = self.form_encoder(current_features, training=training)  # [B, S, T]
        
        # Temporal sequence (if available)
        if temporal_history is not None:
            seq_len = tf.shape(temporal_history)[2]
            feat_dim = tf.shape(temporal_history)[3]
            
            # Flatten: [B*S, seq, feat]
            temporal_flat = tf.reshape(temporal_history, [-1, seq_len, feat_dim])
            temporal_state = self.temporal_gru(temporal_flat, training=training)
            temporal_state = tf.reshape(temporal_state, [batch_size, num_skaters, self.temporal_dim])
        else:
            temporal_state = form
        
        # Combine everything
        combined = tf.concat([
            base + pos_padded,  # Base skill + position
            temporal_state      # Recent form
        ], axis=-1)
        
        # Final embedding
        embeddings = self.fusion(combined, training=training)
        
        return embeddings


class GoalieEmbeddingLayer(layers.Layer):
    """
    Specialized goalie embeddings.
    
    Key difference from skaters:
    - All goalies share a COMMON BASE embedding (goalies are more homogeneous)
    - Small individual adjustments per goalie
    - Focus on recent form (save %, GAA, workload)
    
    This handles the small sample size problem - we have ~80 goalies vs ~800 skaters.
    """
    def __init__(self, num_goalies: int, embed_dim: int,
                 temporal_dim: int = 64, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        
        # SHARED base embedding for all goalies
        # "Being an NHL goalie" has common characteristics
        self.shared_goalie_base = self.add_weight(
            shape=(1, embed_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='shared_goalie_base'
        )
        
        # Small individual adjustments (much smaller than skater embeddings)
        self.goalie_adjustments = layers.Embedding(
            num_goalies,
            embed_dim // 4,  # Only 1/4 the size of skater embeddings
            embeddings_regularizer=keras.regularizers.l2(1e-5),
            name='goalie_individual_adjustments'
        )
        
        # Goalie-specific features (save %, GAA, recent workload)
        self.goalie_feature_encoder = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(temporal_dim)
        ], name='goalie_features')
        
        # Recent form tracker
        self.form_gru = layers.GRU(
            temporal_dim,
            return_sequences=False,
            dropout=dropout,
            name='goalie_form_tracker'
        )
        
        # Fusion
        self.fusion = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout)
        ], name='goalie_fusion')
    
    def call(self, goalie_ids, current_features, 
             temporal_history=None, training=False):
        """
        Args:
            goalie_ids: [batch, 1] - single goalie per team
            current_features: [batch, 1, feat_dim] - save %, GAA, etc.
            temporal_history: [batch, 1, seq_len, feat_dim] - recent games
        
        Returns:
            [batch, 1, embed_dim] - goalie embeddings
        """
        batch_size = tf.shape(goalie_ids)[0]
        
        # Start with shared base (broadcast to batch)
        shared_base = tf.tile(self.shared_goalie_base, [batch_size, 1])
        shared_base = tf.expand_dims(shared_base, 1)  # [B, 1, E]
        
        # Add small individual adjustments
        adjustments = self.goalie_adjustments(goalie_ids)  # [B, 1, E//4]
        adjustments_padded = tf.pad(adjustments,
            [[0, 0], [0, 0], [0, self.embed_dim - self.embed_dim // 4]])
        
        # Encode current features
        encoded_features = self.goalie_feature_encoder(
            current_features, training=training
        )  # [B, 1, T]
        
        # Temporal form (if available)
        if temporal_history is not None:
            seq_len = tf.shape(temporal_history)[2]
            feat_dim = tf.shape(temporal_history)[3]
            
            # Reshape: [B, seq, feat]
            temporal_flat = tf.reshape(temporal_history, [-1, seq_len, feat_dim])
            form_state = self.form_gru(temporal_flat, training=training)
            form_state = tf.reshape(form_state, [batch_size, 1, self.temporal_dim])
        else:
            form_state = encoded_features
        
        # Combine: shared base + individual adjustments + form
        combined = tf.concat([
            shared_base + adjustments_padded,  # Common goalie traits + individual tweaks
            form_state                          # Recent performance
        ], axis=-1)
        
        goalie_embedding = self.fusion(combined, training=training)
        
        return goalie_embedding


# ============================================================================
# PATH 1: GNN PROCESSING
# ============================================================================

class RefPlayerModulator(layers.Layer):
    """
    Modulates individual player embeddings based on referee tendencies.
    
    Example: Physical defenseman with strict ref → reduced effectiveness
    Example: Sniper with ref who calls lots of penalties → more PP opportunities
    """
    def __init__(self, num_refs: int, embed_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        # Ref tendency vectors (what playstyle does this ref penalize/favor?)
        self.ref_tendencies = layers.Embedding(
            num_refs,
            embed_dim,
            embeddings_regularizer=keras.regularizers.l2(1e-5),
            name='ref_tendencies'
        )
        
        # Player-ref compatibility network
        # "How compatible is this player's style with this ref's tendencies?"
        self.compatibility_net = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(32, activation='gelu'),
            layers.Dense(1, activation='tanh')  # -1 (bad matchup) to +1 (good matchup)
        ], name='player_ref_compatibility')
        
        # Modulation network (how much to adjust player embedding)
        self.modulation_net = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ], name='ref_modulation')
    
    def call(self, player_embeddings, ref_ids, player_styles=None, training=False):
        """
        Args:
            player_embeddings: [batch, num_players, embed_dim]
            ref_ids: [batch, num_refs] typically 2 refs
            player_styles: [batch, num_players, style_dim] optional
                e.g., [physicality, finesse, defensive_focus, pp_specialist, ...]
        
        Returns:
            modulated_embeddings: [batch, num_players, embed_dim]
            compatibility_scores: [batch, num_players, 1]
        """
        batch_size = tf.shape(player_embeddings)[0]
        num_players = tf.shape(player_embeddings)[1]
        
        # Get ref tendency vectors (average if multiple refs)
        ref_embeds = self.ref_tendencies(ref_ids)  # [B, num_refs, E]
        ref_tendency = tf.reduce_mean(ref_embeds, axis=1, keepdims=True)  # [B, 1, E]
        ref_tendency = tf.tile(ref_tendency, [1, num_players, 1])  # [B, num_players, E]
        
        # Compute player-ref compatibility
        combined = tf.concat([player_embeddings, ref_tendency], axis=-1)
        compatibility = self.compatibility_net(combined, training=training)  # [B, P, 1]
        
        # Generate modulation (how to adjust each player's embedding)
        modulation = self.modulation_net(combined, training=training)  # [B, P, E]
        
        # Apply modulation (scaled by compatibility)
        # Positive compatibility = boost, negative = nerf
        modulated_embeddings = player_embeddings + (compatibility * modulation * 0.1)
        
        return modulated_embeddings, compatibility


class GNNPath(keras.Model):
    """
    GNN path: Models team as a network.
    Captures passing patterns, line chemistry, zone entries.
    
    Good for: Team-level play style, network effects, structural advantages
    """
    def __init__(self, embed_dim: int, num_refs: int, num_gnn_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        
        # Message passing layers
        self.gnn_layers = []
        for i in range(num_gnn_layers):
            self.gnn_layers.append({
                'attention': layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embed_dim // num_heads,
                    dropout=dropout,
                    name=f'gnn_layer_{i}_attn'
                ),
                'ffn': keras.Sequential([
                    layers.Dense(embed_dim * 4, activation='gelu'),
                    layers.Dropout(dropout),
                    layers.Dense(embed_dim)
                ], name=f'gnn_layer_{i}_ffn'),
                'norm1': layers.LayerNormalization(name=f'gnn_layer_{i}_norm1'),
                'norm2': layers.LayerNormalization(name=f'gnn_layer_{i}_norm2')
            })
        
        # Edge feature encoder (passing freq, chemistry, positioning)
        self.edge_encoder = keras.Sequential([
            layers.Dense(num_heads, activation='relu'),
            layers.Dense(num_heads)
        ], name='edge_features')
        
        # Graph-level pooling
        self.graph_pool = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout)
        ], name='graph_pooling')
        
        # NEW: Ref modulator
        self.ref_modulator = RefPlayerModulator(num_refs, embed_dim, dropout)
    
    def call(self, player_embeddings, ref_ids=None, edge_features=None, 
             attention_mask=None, training=False):
        """
        Args:
            player_embeddings: [batch, num_players, embed_dim]
            ref_ids: [batch, num_refs] NEW
            edge_features: [batch, num_players, num_players, edge_dim]
            attention_mask: [batch, num_players, num_players]
        
        Returns:
            team_embedding: [batch, embed_dim]
            player_embeddings: [batch, num_players, embed_dim]
            ref_compatibility: [batch, num_players, 1] or None
        """
        # NEW: Apply ref modulation BEFORE message passing
        if ref_ids is not None:
            x, ref_compatibility = self.ref_modulator(
                player_embeddings, ref_ids, training=training
            )
        else:
            x = player_embeddings
            ref_compatibility = None
        
        # Message passing through GNN layers
        for layer in self.gnn_layers:
            # Multi-head attention (message passing)
            attn_out = layer['attention'](
                query=x,
                value=x,
                attention_mask=attention_mask,
                training=training
            )
            x = layer['norm1'](x + attn_out)
            
            # Feed-forward
            ffn_out = layer['ffn'](x, training=training)
            x = layer['norm2'](x + ffn_out)
        
        # Aggregate to team level (graph pooling)
        team_embedding = tf.reduce_mean(x, axis=1)  # Simple mean pooling
        team_embedding = self.graph_pool(team_embedding, training=training)
        
        return team_embedding, x, ref_compatibility  # Return compatibility for analysis


# ============================================================================
# PATH 2: BAYESIAN PROCESSING
# ============================================================================

class BayesianPath(keras.Model):
    """
    Bayesian path: Hierarchical matchup modeling.
    Captures line-vs-line effects with proper uncertainty.
    
    Hierarchy:
      League archetypes → Team systems → Specific line matchups
    
    Good for: Small sample matchups, tactical advantages, line chemistry
    """
    def __init__(self, embed_dim: int, num_refs: int, num_archetypes: int = 10,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        
        # League-level archetypes (learned priors)
        # e.g., "Checking line vs Skill line", "Physical vs Finesse"
        self.archetypes = self.add_weight(
            shape=(num_archetypes, embed_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='league_archetypes'
        )
        
        # Line aggregation (players → line embedding)
        self.line_aggregator = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout)
        ], name='line_aggregator')
        
        # Archetype classifier (which archetype does this line resemble?)
        self.archetype_classifier = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(num_archetypes, activation='softmax')
        ], name='archetype_classifier')
        
        # Bayesian inference network
        self.inference_net = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim * 2)  # mu and log_sigma
        ], name='bayesian_inference')
        
        # Matchup encoder
        self.matchup_encoder = keras.Sequential([
            layers.Dense(embed_dim * 2, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ], name='matchup_encoder')

        self.ref_embeddings = layers.Embedding(
            num_refs,
            embed_dim // 4,  # Smaller than team embeddings
            embeddings_regularizer=keras.regularizers.l2(1e-5),
            name='ref_matchup_embeddings'
        )
        
        self.team_ref_interaction = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout)
        ], name='team_ref_interaction')
        
        # "How does Home Team + Away Team + Ref combo historically perform?"
        self.bias_estimator = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(32, activation='gelu'),
            layers.Dense(1, activation='tanh')  # -1 to +1 bias
        ], name='ref_bias_estimator')
    
    
    def call(self, home_player_embeddings, away_player_embeddings,
                ref_ids=None, ref_team_history=None, 
                home_line_indices=None, away_line_indices=None,
                training=False, return_uncertainty=True):
        """
        Args:
            ref_ids: [batch, num_refs] typically 2 refs
            ref_team_history: [batch, 6] features like:
                - home_team_win_rate_with_these_refs
                - away_team_win_rate_with_these_refs
                - home_team_pim_rate_with_these_refs
                - away_team_pim_rate_with_these_refs
                - ref_home_bias_general
                - games_officiated_together
        """
        # Aggregate to line level (either specific line or full team)
        if home_line_indices is not None:
            home_line = tf.gather(home_player_embeddings, home_line_indices, batch_dims=1)
            home_line = self.line_aggregator(home_line, training=training)
            home_line = tf.reduce_mean(home_line, axis=1)
        else:
            # Use full team
            home_line = tf.reduce_mean(home_player_embeddings, axis=1)
            home_line = self.line_aggregator(home_line, training=training)
        
        if away_line_indices is not None:
            away_line = tf.gather(away_player_embeddings, away_line_indices, batch_dims=1)
            away_line = self.line_aggregator(away_line, training=training)
            away_line = tf.reduce_mean(away_line, axis=1)
        else:
            away_line = tf.reduce_mean(away_player_embeddings, axis=1)
            away_line = self.line_aggregator(away_line, training=training)
        if ref_ids is not None:
            ref_embeds = self.ref_embeddings(ref_ids)  # [B, num_refs, E//4]
            ref_context = tf.reduce_mean(ref_embeds, axis=1)  # [B, E//4]
            
            # Combine team embeddings with ref
            home_ref_interaction = self.team_ref_interaction(
                tf.concat([home_line, ref_context], axis=-1),
                training=training
            )
            away_ref_interaction = self.team_ref_interaction(
                tf.concat([away_line, ref_context], axis=-1),
                training=training
            )
            
            # Estimate historical bias
            if ref_team_history is not None:
                ref_bias = self.bias_estimator(ref_team_history, training=training)
            else:
                ref_bias = tf.zeros((tf.shape(home_line)[0], 1))
        else:
            home_ref_interaction = home_line
            away_ref_interaction = away_line
            ref_bias = tf.zeros((tf.shape(home_line)[0], 1))
        
        # Use ref-adjusted line embeddings for matchup context
        matchup_context = tf.concat([
            home_ref_interaction, 
            away_ref_interaction,
            ref_bias
        ], axis=-1)
        
        # Determine which league archetype this matchup resembles
        archetype_probs = self.archetype_classifier(matchup_context, training=training)
        
        # Get archetype prior (weighted average of archetypes)
        archetype_prior = tf.matmul(archetype_probs, self.archetypes)  # [B, E]
        
        # Bayesian inference: combine prior + observed matchup
        combined = tf.concat([
            matchup_context,
            archetype_prior
        ], axis=-1)
        
        # Output distribution parameters
        params = self.inference_net(combined, training=training)
        mu, log_sigma = tf.split(params, 2, axis=-1)
        sigma = tf.nn.softplus(log_sigma) + 1e-6
        
        if training:
            # Sample during training (reparameterization trick)
            epsilon = tf.random.normal(tf.shape(mu))
            matchup_embedding = mu + sigma * epsilon
        else:
            # Use mean at inference
            matchup_embedding = mu
        
        # Encode final matchup
        matchup_embedding = self.matchup_encoder(
            tf.concat([matchup_embedding, matchup_context], axis=-1),
            training=training
        )
        
        if return_uncertainty:
            uncertainty = tf.reduce_mean(sigma, axis=-1, keepdims=True)
            return matchup_embedding, uncertainty, archetype_probs, ref_bias
        else:
            return matchup_embedding


# ============================================================================
# MODULAR OUTPUT HEADS
# ============================================================================

class OutputHead(layers.Layer):
    """Base class for output heads"""
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, features, training=False):
        raise NotImplementedError


class MoneylineHead(OutputHead):
    """Win probability"""
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(name='moneyline', **kwargs)
        self.predictor = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(64, activation='gelu'),
            layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, features, training=False):
        return self.predictor(features, training=training)


class PuckLineHead(OutputHead):
    """Spread (mixture density network)"""
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(name='puck_line', **kwargs)
        self.mu = layers.Dense(2, name='spread_mu')
        self.sigma = layers.Dense(2, activation='softplus', name='spread_sigma')
        self.pi = layers.Dense(2, activation='softmax', name='spread_pi')
    
    def call(self, features, training=False):
        return {
            'mu': self.mu(features),
            'sigma': self.sigma(features) + 1e-6,
            'pi': self.pi(features)
        }


class TotalGoalsHead(OutputHead):
    """Over/under total"""
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(name='total', **kwargs)
        self.predictor = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(1, activation='softplus')
        ])
    
    def call(self, features, training=False):
        return self.predictor(features, training=training)


# ============================================================================
# COMPLETE DUAL-PATH MODEL
# ============================================================================

class NHLDualPathModel(keras.Model):
    """
    Complete dual-path model:
    
    Player Embeddings (shared) → GNN Path → Outputs
                               ↘ Bayes Path → Outputs
    """
    def __init__(self, 
                 num_skaters: int,
                 num_goalies: int,
                 num_refs: int,
                 embed_dim: int = 128,
                 temporal_dim: int = 64,
                 num_gnn_layers: int = 3,
                 output_heads: List[str] = ['moneyline', 'puck_line', 'total'],
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        
        # Separate embeddings for skaters and goalies
        self.skater_embedder = SkaterEmbeddingLayer(
            num_skaters, embed_dim, temporal_dim, dropout
        )
        
        self.goalie_embedder = GoalieEmbeddingLayer(
            num_goalies, embed_dim, temporal_dim, dropout
        )
        
        # Path 1: GNN (with ref player modulation)
        self.gnn_path = GNNPath(
            embed_dim, num_refs, num_gnn_layers, num_heads=4, dropout=dropout
        )
        
        # Path 2: Bayesian (with ref matchup context)
        self.bayes_path = BayesianPath(
            embed_dim, num_refs, num_archetypes=10, dropout=dropout
        )
        
        # Context encoder (game situation, rest, etc.)
        self.context_encoder = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout)
        ], name='context_encoder')
        
        # Path fusion
        self.fusion = keras.Sequential([
            layers.Dense(embed_dim * 2, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ], name='dual_path_fusion')
        
        # Initialize output heads
        self.output_heads = {}
        for head_name in output_heads:
            if head_name == 'moneyline':
                self.output_heads[head_name] = MoneylineHead(embed_dim, dropout)
            elif head_name == 'puck_line':
                self.output_heads[head_name] = PuckLineHead(embed_dim, dropout)
            elif head_name == 'total':
                self.output_heads[head_name] = TotalGoalsHead(embed_dim, dropout)
    
    def call(self, inputs, training=False):
        """
        Args:
            inputs: dict with:
                # Skaters (5 per team)
                - home_skater_ids: [batch, 5]
                - home_skater_positions: [batch, 5]
                - home_skater_features: [batch, 5, feat_dim]
                - home_skater_temporal: [batch, 5, seq, feat_dim]
                
                # Goalies (1 per team)
                - home_goalie_id: [batch, 1]
                - home_goalie_position: [batch, 1]
                - home_goalie_features: [batch, 1, feat_dim]
                - home_goalie_temporal: [batch, 1, seq, feat_dim]
                
                # Same for away team
                
                # GNN features
                - home_edge_features: [batch, 6, 6, edge_dim]
                - away_edge_features: [batch, 6, 6, edge_dim]
                
                # Context
                - context_features: [batch, context_dim]
                
                # NEW: Referee features
                - ref_ids: [batch, num_refs]
                - ref_team_history: [batch, 6]
        
        Returns:
            dict with predictions from all heads + intermediates
        """
        # STEP 1: Get player embeddings (SHARED between paths)
        
        # Skaters
        home_skater_embeds = self.skater_embedder(
            inputs['home_skater_ids'],
            inputs['home_skater_positions'],
            inputs['home_skater_features'],
            inputs.get('home_skater_temporal'),
            training=training
        )  # [B, 5, E]
        
        away_skater_embeds = self.skater_embedder(
            inputs['away_skater_ids'],
            inputs['away_skater_positions'],
            inputs['away_skater_features'],
            inputs.get('away_skater_temporal'),
            training=training
        )
        
        # Goalies
        home_goalie_embed = self.goalie_embedder(
            inputs['home_goalie_id'],
            inputs['home_goalie_position'],
            inputs['home_goalie_features'],
            inputs.get('home_goalie_temporal'),
            training=training
        )  # [B, 1, E]
        
        away_goalie_embed = self.goalie_embedder(
            inputs['away_goalie_id'],
            inputs['away_goalie_position'],
            inputs['away_goalie_features'],
            inputs.get('away_goalie_temporal'),
            training=training
        )
        
        # Combine skaters + goalie
        home_players = tf.concat([home_skater_embeds, home_goalie_embed], axis=1)  # [B, 6, E]
        away_players = tf.concat([away_skater_embeds, away_goalie_embed], axis=1)
        
        # STEP 2: GNN path WITH ref modulation
        home_gnn, home_player_gnn, home_ref_compat = self.gnn_path(
            home_players,
            ref_ids=inputs.get('ref_ids'),  # NEW
            edge_features=inputs.get('home_edge_features'),
            training=training
        )
        
        away_gnn, away_player_gnn, away_ref_compat = self.gnn_path(
            away_players,
            ref_ids=inputs.get('ref_ids'),  # NEW
            edge_features=inputs.get('away_edge_features'),
            training=training
        )
        
        # STEP 3: Bayesian path WITH ref matchup context
        bayes_matchup, bayes_uncertainty, archetype_probs, ref_bias = self.bayes_path(
            home_players,
            away_players,
            ref_ids=inputs.get('ref_ids'),  # NEW
            ref_team_history=inputs.get('ref_team_history'),  # NEW
            training=training,
            return_uncertainty=True
        )
        
        # STEP 4: Encode context
        context = self.context_encoder(inputs['context_features'], training=training)
        
        # STEP 5: Fuse all information
        combined = self.fusion(tf.concat([
            home_gnn,
            away_gnn,
            bayes_matchup,
            context
        ], axis=-1), training=training)
        
        # STEP 6: Run output heads
        predictions = {}
        for head_name, head in self.output_heads.items():
            predictions[head_name] = head(combined, training=training)
        
        # Add intermediates (useful for analysis/debugging)
        predictions.update({
            'gnn_home': home_gnn,
            'gnn_away': away_gnn,
            'bayes_matchup': bayes_matchup,
            'bayes_uncertainty': bayes_uncertainty,
            'archetype_probs': archetype_probs,
            'home_player_embeddings': home_players,
            'away_player_embeddings': away_players,
            'home_ref_compatibility': home_ref_compat,
            'away_ref_compatibility': away_ref_compat,
            'ref_bias': ref_bias,
        })
        
        return predictions
    
    def add_output_head(self, head_name: str, head: OutputHead):
        """Dynamically add output head"""
        self.output_heads[head_name] = head
        print(f"✓ Added output head: {head_name}")
    
    def remove_output_head(self, head_name: str):
        """Remove output head"""
        if head_name in self.output_heads:
            del self.output_heads[head_name]
            print(f"✓ Removed output head: {head_name}")


# ============================================================================
# BUILD FUNCTION
# ============================================================================

def get_player_counts(db_path: str = "data/nhl_data.db") -> Tuple[int, int, int]:
    """Get player and referee counts from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    min_games =10
    
    cursor.execute("""
        SELECT COUNT(*) FROM (
            SELECT player_id
            FROM player_game_stats
            WHERE position != 'G'
            GROUP BY player_id
            HAVING COUNT(DISTINCT game_id) >= ?
        )
    """, (min_games,))
    result = cursor.fetchone()
    num_skaters = (result[0]) + 1
    
    # Goalies with minimum games
    cursor.execute("""
        SELECT COUNT(*) FROM (
            SELECT player_id
            FROM player_game_stats
            WHERE position = 'G'
            GROUP BY player_id
            HAVING COUNT(DISTINCT game_id) >= ?
        )
    """, (min_games,))
    result = cursor.fetchone()
    num_goalies = (result[0]) + 1
    
    # Add referee count
    try:
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT official_name
                FROM game_officials
                GROUP BY official_name
                HAVING COUNT(DISTINCT game_id) >= ?
            )
        """, (min_games,))
        num_refs = cursor.fetchone()[0] + 1
    except:
        num_refs = 100  # Default fallback
    
    conn.close()
    return num_skaters, num_goalies, num_refs


def build_model_and_train(db_path: str = "data/nhl_data.db",
                          output_heads: List[str] = ['moneyline', 'puck_line', 'total']):
    """
    Build complete NHL dual-path model.
    """
    from utils.backtesting import BettingBacktester
    
    print("="*80)
    print("NHL DUAL-PATH MODEL")
    print("="*80)
    
    # Get player counts
    num_skaters, num_goalies, num_refs = get_player_counts(db_path)
    print(f"\n✓ Skaters: {num_skaters}")
    print(f"✓ Goalies: {num_goalies}")
    print(f"✓ Referees: {num_refs}")
    
    # Build model
    model = NHLDualPathModel(
        num_skaters=num_skaters,
        num_goalies=num_goalies,
        num_refs=num_refs,
        embed_dim=128,
        temporal_dim=64,
        num_gnn_layers=3,
        output_heads=output_heads,
        dropout=0.1
    )
    
    # Optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=3e-4,
        weight_decay=1e-4
    )
    
    # Loss function (placeholder)
    def loss_fn(y_true, y_pred):
        # Combine losses from all heads
        total_loss = 0.0
        
        if 'moneyline' in y_pred:
            ml_loss = tf.keras.losses.binary_crossentropy(
                y_true['win'], y_pred['moneyline']
            )
            total_loss += ml_loss
        
        if 'total' in y_pred:
            total_loss += tf.abs(y_true['total_goals'] - y_pred['total'])
        
        return total_loss
    
    print(f"\n✓ Model built with {len(output_heads)} output heads:")
    for head in output_heads:
        print(f"    - {head}")
    
    return model, optimizer, loss_fn

