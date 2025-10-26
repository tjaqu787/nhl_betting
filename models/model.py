import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Tuple, Optional
import tensorflow_probability as tfp

tfd = tfp.distributions


# ============================================================================
# 1. GOALIE ENCODER (Separate from skaters)
# ============================================================================

class GoalieEncoder(layers.Layer):
    """
    Specialized encoder for goalies.
    All goalies share same initial embedding (no 'nemesis' effects),
    then fine-tuned based on recent performance.
    """
    def __init__(self, num_goalies: int, embed_dim: int, temporal_dim: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_goalies = num_goalies
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        
        # Shared base embedding for all goalies
        self.shared_goalie_embedding = self.add_weight(
            shape=(1, embed_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='shared_goalie_base'
        )
        
        # Individual adjustments (small capacity)
        self.goalie_adjustments = layers.Embedding(
            num_goalies,
            embed_dim // 4,  # Much smaller than skaters
            embeddings_regularizer=keras.regularizers.l2(1e-5),
            name='goalie_adjustments'
        )
        
        # Goalie-specific features (save %, GAA, recent form)
        self.goalie_feature_encoder = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(temporal_dim)
        ], name='goalie_features')
        
        # Recent form tracker (GRU)
        self.form_gru = layers.GRU(
            temporal_dim,
            return_sequences=False,
            dropout=dropout,
            name='goalie_form_gru'
        )
        
        # Fusion
        self.fusion = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout)
        ], name='goalie_fusion')
    
    def call(self, goalie_ids, goalie_features, temporal_history=None, training=False):
        """
        Args:
            goalie_ids: [batch, 1] - single goalie per team
            goalie_features: [batch, 1, feat_dim] - save %, GAA, etc.
            temporal_history: [batch, 1, seq_len, feat_dim] - recent games
        Returns:
            [batch, 1, embed_dim]
        """
        batch_size = tf.shape(goalie_ids)[0]
        
        # Start with shared embedding (broadcast to batch)
        shared_embed = tf.tile(self.shared_goalie_embedding, [batch_size, 1])
        shared_embed = tf.expand_dims(shared_embed, 1)  # [B, 1, embed_dim]
        
        # Add small individual adjustments
        adjustments = self.goalie_adjustments(goalie_ids)  # [B, 1, embed_dim//4]
        
        # Encode current features
        encoded_features = self.goalie_feature_encoder(
            goalie_features, training=training
        )  # [B, 1, temporal_dim]
        
        # Temporal form (if available)
        if temporal_history is not None:
            # Reshape: [B*1, seq_len, feat_dim]
            temporal_flat = tf.reshape(temporal_history, [-1, 
                tf.shape(temporal_history)[2], 
                tf.shape(temporal_history)[3]])
            form_state = self.form_gru(temporal_flat, training=training)
            form_state = tf.reshape(form_state, [batch_size, 1, self.temporal_dim])
        else:
            form_state = encoded_features
        
        # Combine: shared base + adjustments + form
        # Pad adjustments to match dimensions
        adjustments_padded = tf.pad(adjustments, 
            [[0, 0], [0, 0], [0, self.embed_dim - self.embed_dim // 4]])
        
        combined = tf.concat([
            shared_embed + adjustments_padded,
            form_state
        ], axis=-1)
        
        goalie_vector = self.fusion(combined, training=training)
        
        return goalie_vector


# ============================================================================
# 2. SKATER ENCODER (with position awareness)
# ============================================================================

class SkaterEncoder(layers.Layer):
    """
    Encoder for forwards and defensemen.
    Position-aware embeddings.
    """
    def __init__(self, num_skaters: int, embed_dim: int, temporal_dim: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        
        # Skater embeddings
        self.skater_embeddings = layers.Embedding(
            num_skaters,
            embed_dim,
            embeddings_regularizer=keras.regularizers.l2(1e-5),
            name='skater_embeddings'
        )
        
        # Position embeddings (C, L, R, D)
        self.position_embeddings = layers.Embedding(
            4,  # Center, Left Wing, Right Wing, Defense
            embed_dim // 4,
            name='position_embeddings'
        )
        
        # Feature encoder
        self.feature_encoder = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(temporal_dim)
        ], name='skater_features')
        
        # Temporal tracker
        self.temporal_gru = layers.GRU(
            temporal_dim,
            return_sequences=False,
            dropout=dropout,
            name='skater_form_gru'
        )
        
        # Fusion
        self.fusion = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout)
        ], name='skater_fusion')
    
    def call(self, skater_ids, skater_features, positions, 
             temporal_history=None, training=False):
        """
        Args:
            skater_ids: [batch, num_skaters]
            skater_features: [batch, num_skaters, feat_dim]
            positions: [batch, num_skaters] - position codes (0=C, 1=L, 2=R, 3=D)
            temporal_history: [batch, num_skaters, seq_len, feat_dim]
        Returns:
            [batch, num_skaters, embed_dim]
        """
        # Skater embeddings
        skater_embed = self.skater_embeddings(skater_ids)
        
        # Position embeddings
        pos_embed = self.position_embeddings(positions)
        pos_embed_padded = tf.pad(pos_embed,
            [[0, 0], [0, 0], [0, self.embed_dim - self.embed_dim // 4]])
        
        # Encode features
        if temporal_history is not None:
            batch_size = tf.shape(temporal_history)[0]
            num_skaters = tf.shape(temporal_history)[1]
            seq_len = tf.shape(temporal_history)[2]
            feat_dim = tf.shape(temporal_history)[3]
            
            temporal_flat = tf.reshape(temporal_history, [-1, seq_len, feat_dim])
            encoded = self.feature_encoder(temporal_flat, training=training)
            temporal_state = self.temporal_gru(encoded, training=training)
            temporal_state = tf.reshape(temporal_state, 
                [batch_size, num_skaters, self.temporal_dim])
        else:
            temporal_state = self.feature_encoder(skater_features, training=training)
        
        # Combine
        combined = tf.concat([
            skater_embed + pos_embed_padded,
            temporal_state
        ], axis=-1)
        
        skater_vectors = self.fusion(combined, training=training)
        
        return skater_vectors


# ============================================================================
# 3. LINE COMBINATION ENCODER (Forward lines & D-pairs)
# ============================================================================

class LineCombinationGNN(layers.Layer):
    """
    Models chemistry within line units:
    - Forward lines: 3 players
    - Defensive pairs: 2 players
    
    Uses structured attention (players only attend within their line).
    """
    def __init__(self, embed_dim: int, num_heads: int = 4,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Separate attention for forwards and defense
        self.forward_line_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name='forward_line_attention'
        )
        
        self.defense_pair_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name='defense_pair_attention'
        )
        
        # Line chemistry encoder (models time together, stats together)
        self.chemistry_encoder = layers.Dense(num_heads, activation='relu',
                                             name='line_chemistry')
        
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
        self.ffn = keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ], name='line_ffn')
    
    def call(self, skater_embeds, positions, line_chemistry=None, training=False):
        """
        Args:
            skater_embeds: [batch, num_skaters, embed_dim]
            positions: [batch, num_skaters] - 0=C, 1=L, 2=R, 3=D
            line_chemistry: [batch, num_skaters, num_skaters, chem_dim] - time together
        Returns:
            [batch, num_skaters, embed_dim]
        """
        batch_size = tf.shape(skater_embeds)[0]
        num_skaters = tf.shape(skater_embeds)[1]
        
        # Separate forwards (C, L, R) and defense (D)
        is_forward = positions < 3  # [batch, num_skaters]
        is_defense = positions == 3
        
        # Create masks for attention
        # Forwards attend to forwards, defense to defense
        forward_mask = tf.cast(is_forward[:, None, :], tf.float32)  # [B, 1, S]
        defense_mask = tf.cast(is_defense[:, None, :], tf.float32)
        
        # Forward line attention
        forward_out = self.forward_line_attn(
            query=skater_embeds,
            value=skater_embeds,
            attention_mask=forward_mask,
            training=training
        )
        
        # Defense pair attention
        defense_out = self.defense_pair_attn(
            query=skater_embeds,
            value=skater_embeds,
            attention_mask=defense_mask,
            training=training
        )
        
        # Combine based on position
        is_forward_3d = tf.cast(is_forward[:, :, None], tf.float32)
        is_defense_3d = tf.cast(is_defense[:, :, None], tf.float32)
        
        attn_out = (forward_out * is_forward_3d + 
                    defense_out * is_defense_3d)
        
        # Add chemistry bias if available
        if line_chemistry is not None:
            chem_bias = self.chemistry_encoder(line_chemistry)  # [B, S, S, H]
            # This would be added to attention scores (simplified here)
        
        # Residual + norm
        x = self.norm1(skater_embeds + attn_out)
        
        # FFN
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + ffn_out)
        
        return x


# ============================================================================
# 4. SPECIAL TEAMS ENCODER (PP/PK Units)
# ============================================================================

class SpecialTeamsEncoder(layers.Layer):
    """
    Models power play and penalty kill units.
    These are fixed units that play together in man-advantage situations.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        
        # PP unit encoder (5 players)
        self.pp_encoder = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim // 2)
        ], name='powerplay_encoder')
        
        # PK unit encoder (4 players)
        self.pk_encoder = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim // 2)
        ], name='penalty_kill_encoder')
        
        # Unit chemistry (time together on PP/PK)
        self.unit_chemistry = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=embed_dim // 4,
            dropout=dropout,
            name='special_teams_chemistry'
        )
        
        self.norm = layers.LayerNormalization()
    
    def call(self, skater_embeds, pp_indices, pk_indices, training=False):
        """
        Args:
            skater_embeds: [batch, num_skaters, embed_dim]
            pp_indices: [batch, 5] - indices of PP1 unit
            pk_indices: [batch, 4] - indices of PK1 unit
        Returns:
            pp_unit: [batch, embed_dim // 2]
            pk_unit: [batch, embed_dim // 2]
        """
        # Gather PP unit players
        batch_size = tf.shape(skater_embeds)[0]
        
        # PP unit
        pp_players = tf.gather(skater_embeds, pp_indices, batch_dims=1)  # [B, 5, E]
        pp_attended = self.unit_chemistry(pp_players, pp_players, training=training)
        pp_pooled = tf.reduce_mean(pp_attended, axis=1)  # [B, E]
        pp_encoded = self.pp_encoder(pp_pooled, training=training)
        
        # PK unit
        pk_players = tf.gather(skater_embeds, pk_indices, batch_dims=1)  # [B, 4, E]
        pk_attended = self.unit_chemistry(pk_players, pk_players, training=training)
        pk_pooled = tf.reduce_mean(pk_attended, axis=1)  # [B, E]
        pk_encoded = self.pk_encoder(pk_pooled, training=training)
        
        return pp_encoded, pk_encoded


# ============================================================================
# 5. ICE TIME WEIGHTED POOLING
# ============================================================================

class IceTimeWeightedPooling(layers.Layer):
    """
    Pool skaters weighted by their expected ice time.
    McDavid (22 min) >> 4th liner (8 min)
    """
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        
        # Learn to adjust ice time predictions
        self.ice_time_predictor = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.Dense(1, activation='softplus'),  # Positive ice time
        ], name='ice_time_predictor')
        
        # Combine with given ice time
        self.weight_fusion = layers.Dense(1, activation='sigmoid',
                                          name='ice_time_weight')
    
    def call(self, skater_embeds, ice_time_features, training=False):
        """
        Args:
            skater_embeds: [batch, num_skaters, embed_dim]
            ice_time_features: [batch, num_skaters, ice_time_dim] - avg TOI, recent TOI
        Returns:
            [batch, embed_dim]
        """
        # Predict ice time adjustment
        predicted_weights = self.ice_time_predictor(ice_time_features, training=training)
        # [B, S, 1]
        
        # Normalize to sum to 1 (softmax over players)
        weights = tf.nn.softmax(predicted_weights, axis=1)
        
        # Weighted sum
        pooled = tf.reduce_sum(skater_embeds * weights, axis=1)  # [B, E]
        
        return pooled


# ============================================================================
# 6. PHYSICAL PLAY ENCODER
# ============================================================================

class PhysicalPlayEncoder(layers.Layer):
    """
    Emphasizes physical aspects: hits, blocks, penalty minutes.
    Important for playoff hockey and certain matchups.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.physical_encoder = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim // 4)
        ], name='physical_features')
        
        # Team toughness metric
        self.toughness_scorer = keras.Sequential([
            layers.Dense(32, activation='gelu'),
            layers.Dense(1)
        ], name='toughness_score')
    
    def call(self, physical_features, training=False):
        """
        Args:
            physical_features: [batch, num_players, phys_dim]
                Contains: hits/game, blocks/game, PIM, fights, etc.
        Returns:
            team_physicality: [batch, embed_dim // 4]
        """
        # Encode each player's physical play
        encoded = self.physical_encoder(physical_features, training=training)
        # [B, P, embed_dim//4]
        
        # Aggregate to team level
        team_physical = tf.reduce_mean(encoded, axis=1)  # [B, embed_dim//4]
        
        # Compute toughness score
        toughness = self.toughness_scorer(team_physical, training=training)
        
        return team_physical, toughness


# ============================================================================
# 7. NHL BETTING MODEL (Main Architecture)
# ============================================================================

class NHLBettingModel(keras.Model):
    """
    Complete NHL-specific model.
    Handles 6-player units (5 skaters + 1 goalie).
    """
    def __init__(self,
                 num_skaters: int,
                 num_goalies: int,
                 embed_dim: int = 128,
                 temporal_dim: int = 64,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        
        # Component 1: Separate encoders for skaters and goalies
        self.skater_encoder = SkaterEncoder(
            num_skaters, embed_dim, temporal_dim, dropout
        )
        
        self.goalie_encoder = GoalieEncoder(
            num_goalies, embed_dim, temporal_dim, dropout
        )
        
        # Component 2: Line combinations
        self.line_gnn = LineCombinationGNN(embed_dim, num_heads, dropout)
        
        # Component 3: Special teams
        self.special_teams = SpecialTeamsEncoder(embed_dim, dropout)
        
        # Component 4: Ice time weighted pooling
        self.ice_time_pool = IceTimeWeightedPooling(embed_dim)
        
        # Component 5: Physical play
        self.physical_encoder = PhysicalPlayEncoder(embed_dim, dropout)
        
        # Component 6: Matchup modeling (goalie vs opposing forwards)
        self.goalie_matchup = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name='goalie_vs_forwards'
        )
        
        # Component 7: Team fusion
        self.team_fusion = keras.Sequential([
            layers.Dense(embed_dim * 2, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ], name='team_fusion')
        
        # Final prediction heads
        self.context_encoder = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout)
        ], name='context_encoder')
        
        # Win probability
        self.win_head = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(64, activation='gelu'),
            layers.Dense(1, activation='sigmoid')
        ], name='win_prob')
        
        # Total goals (over/under)
        self.total_head = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(64, activation='gelu'),
            layers.Dense(1, activation='softplus')  # Positive total
        ], name='total_goals')
        
        # Puck line (spread) - mixture model
        self.spread_mu_head = layers.Dense(2, name='spread_mu')
        self.spread_sigma_head = layers.Dense(2, activation='softplus', name='spread_sigma')
        self.spread_pi_head = layers.Dense(2, activation='softmax', name='spread_pi')
    
    def call(self, inputs, training=False):
        """
        Args:
            inputs: dict with:
                Skaters (5 per team):
                - home_skater_ids: [batch, 5]
                - home_skater_features: [batch, 5, feat_dim]
                - home_skater_positions: [batch, 5] (0=C, 1=L, 2=R, 3=D)
                - home_skater_temporal: [batch, 5, seq_len, feat_dim]
                - home_ice_time: [batch, 5, ice_dim]
                
                Goalies (1 per team):
                - home_goalie_id: [batch, 1]
                - home_goalie_features: [batch, 1, feat_dim]
                - home_goalie_temporal: [batch, 1, seq_len, feat_dim]
                
                Special teams:
                - home_pp_indices: [batch, 5] - PP1 unit
                - home_pk_indices: [batch, 4] - PK1 unit
                
                Physical:
                - home_physical_features: [batch, 5, phys_dim]
                
                Context:
                - context_features: [batch, context_dim]
                
                (Same for away team)
        Returns:
            dict with predictions
        """
        # Encode skaters
        home_skaters = self.skater_encoder(
            inputs['home_skater_ids'],
            inputs['home_skater_features'],
            inputs['home_skater_positions'],
            inputs.get('home_skater_temporal'),
            training=training
        )
        
        away_skaters = self.skater_encoder(
            inputs['away_skater_ids'],
            inputs['away_skater_features'],
            inputs['away_skater_positions'],
            inputs.get('away_skater_temporal'),
            training=training
        )
        
        # Encode goalies
        home_goalie = self.goalie_encoder(
            inputs['home_goalie_id'],
            inputs['home_goalie_features'],
            inputs.get('home_goalie_temporal'),
            training=training
        )
        
        away_goalie = self.goalie_encoder(
            inputs['away_goalie_id'],
            inputs['away_goalie_features'],
            inputs.get('away_goalie_temporal'),
            training=training
        )
        
        # Model line combinations
        home_skaters = self.line_gnn(
            home_skaters,
            inputs['home_skater_positions'],
            inputs.get('home_line_chemistry'),
            training=training
        )
        
        away_skaters = self.line_gnn(
            away_skaters,
            inputs['away_skater_positions'],
            inputs.get('away_line_chemistry'),
            training=training
        )
        
        # Special teams
        home_pp, home_pk = self.special_teams(
            home_skaters,
            inputs['home_pp_indices'],
            inputs['home_pk_indices'],
            training=training
        )
        
        away_pp, away_pk = self.special_teams(
            away_skaters,
            inputs['away_pp_indices'],
            inputs['away_pk_indices'],
            training=training
        )
        
        # Physical play
        home_physical, home_toughness = self.physical_encoder(
            inputs['home_physical_features'], training=training
        )
        
        away_physical, away_toughness = self.physical_encoder(
            inputs['away_physical_features'], training=training
        )
        
        # Ice time weighted pooling
        home_skaters_pooled = self.ice_time_pool(
            home_skaters,
            inputs['home_ice_time'],
            training=training
        )
        
        away_skaters_pooled = self.ice_time_pool(
            away_skaters,
            inputs['away_ice_time'],
            training=training
        )
        
        # Goalie vs opposing forwards matchup
        home_goalie_vs_away = self.goalie_matchup(
            home_goalie,  # Query: home goalie
            away_skaters,  # Key/Value: away forwards
            training=training
        )
        home_goalie_final = tf.squeeze(home_goalie_vs_away, axis=1)  # [B, E]
        
        away_goalie_vs_home = self.goalie_matchup(
            away_goalie,
            home_skaters,
            training=training
        )
        away_goalie_final = tf.squeeze(away_goalie_vs_home, axis=1)
        
        # Fuse everything into team representations
        home_team = self.team_fusion(tf.concat([
            home_skaters_pooled,
            home_goalie_final,
            home_pp,
            home_pk,
            home_physical
        ], axis=-1), training=training)
        
        away_team = self.team_fusion(tf.concat([
            away_skaters_pooled,
            away_goalie_final,
            away_pp,
            away_pk,
            away_physical
        ], axis=-1), training=training)
        
        # Add context
        context = self.context_encoder(inputs['context_features'], training=training)
        
        # Combine for predictions
        combined = tf.concat([
            home_team,
            away_team,
            context,
            home_toughness - away_toughness  # Toughness differential
        ], axis=-1)
        
        # Predictions
        win_prob = self.win_head(combined, training=training)
        total_goals = self.total_head(combined, training=training)
        
        # Spread distribution
        spread_mu = self.spread_mu_head(combined)
        spread_sigma = self.spread_sigma_head(combined) + 1e-6
        spread_pi = self.spread_pi_head(combined)
        
        return {
            'win_prob': win_prob,
            'total_goals': total_goals,
            'spread_dist': {
                'mu': spread_mu,
                'sigma': spread_sigma,
                'pi': spread_pi
            },
            'home_team_embedding': home_team,
            'away_team_embedding': away_team,
            'home_goalie_embedding': home_goalie_final,
            'away_goalie_embedding': away_goalie_final,
            'home_toughness': home_toughness,
            'away_toughness': away_toughness
        }


# ============================================================================
# 8. NHL-SPECIFIC LOSS FUNCTIONS
# ============================================================================

def nhl_composite_loss(y_true, y_pred, 
                       lambda_win=1.0,
                       lambda_total=0.5,
                       lambda_spread=0.5):
    """
    NHL-specific loss combining:
    - Win probability (moneyline)
    - Total goals (over/under)
    - Puck line (spread)
    """
    # Win loss
    win_loss = keras.losses.binary_crossentropy(
        y_true['win'], y_pred['win_prob']
    )
    
    # Total goals loss (MAE)
    total_loss = tf.abs(y_true['total_goals'] - y_pred['total_goals'])
    
    # Spread loss (mixture density)
    spread_nll = mixture_density_loss(
        y_true['spread'],
        y_pred['spread_dist']['mu'],
        y_pred['spread_dist']['sigma'],
        y_pred['spread_dist']['pi']
    )
    
    total = (lambda_win * win_loss +
             lambda_total * total_loss +
             lambda_spread * spread_nll)
    
    return total


def mixture_density_loss(y_true, mu, sigma, pi):
    """MDN loss for spread prediction"""
    y_true = tf.expand_dims(y_true, -1)
    mu = tf.expand_dims(mu, 1)
    sigma = tf.expand_dims(sigma, 1)
    pi = tf.expand_dims(pi, 1)
    
    normal_dist = tfd.Normal(loc=mu, scale=sigma)
    component_log_probs = normal_dist.log_prob(y_true)
    log_pi = tf.math.log(pi + 1e-8)
    log_weighted = component_log_probs + log_pi
    log_likelihood = tf.reduce_logsumexp(log_weighted, axis=-1)
    
    return -tf.reduce_mean(log_likelihood)


# ============================================================================
# 9. EXAMPLE USAGE
# ============================================================================

def build_nhl_model():
    """Build NHL-specific model"""
    
    NUM_SKATERS = 800  # ~25 skaters per team × 32 teams
    NUM_GOALIES = 80   # ~2-3 goalies per team × 32 teams
    EMBED_DIM = 128
    TEMPORAL_DIM = 64
    
    model = NHLBettingModel(
        num_skaters=NUM_SKATERS,
        num_goalies=NUM_GOALIES,
        embed_dim=EMBED_DIM,
        temporal_dim=TEMPORAL_DIM,
        num_heads=4,
        dropout=0.1
    )
    
    print("✓ NHL-specific model built!")
    print(f"  Skaters: {NUM_SKATERS}")
    print(f"  Goalies: {NUM_GOALIES} (shared initialization)")
    print(f"  Embed dim: {EMBED_DIM}")
    print("\nFeatures:")
    print("  ✓ Separate goalie encoder (shared base)")
    print("  ✓ Line combinations (forward lines + D-pairs)")
    print("  ✓ Special teams (PP/PK units)")
    print("  ✓ Ice time weighted pooling")
    print("  ✓ Physical play emphasis")
    print("  ✓ 6-player ice units (5 skaters + 1 goalie)")
    
    return model


if __name__ == "__main__":
    model = build_nhl_model()