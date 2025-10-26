"""
NHL Betting Models - Dual Path Architecture
Path 1: GNN (Team vs Team) - Network structure modeling
Path 2: Bayesian (Line vs Line) - Hierarchical matchup modeling

Modular output heads for different bet types
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np
from typing import Dict, List, Optional, Tuple
import sqlite3

tfd = tfp.distributions
tfb = tfp.bijectors


# ============================================================================
# QUERIES: Get player counts dynamically
# ============================================================================

def get_player_counts(db_path: str = "nhl_data.db") -> Tuple[int, int]:
    """
    Get actual number of skaters and goalies from database.
    
    Returns: 
        (num_skaters, num_goalies)
        
    Note: Add 1 to each for padding/unknown player at index 0
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count distinct skaters (non-goalies)
    cursor.execute("""
        SELECT COUNT(DISTINCT player_id)
        FROM player_game_stats
        WHERE position != 'G'
    """)
    num_skaters = cursor.fetchone()[0] + 1  # +1 for unknown/padding
    
    # Count distinct goalies
    cursor.execute("""
        SELECT COUNT(DISTINCT player_id)
        FROM player_game_stats
        WHERE position = 'G'
    """)
    num_goalies = cursor.fetchone()[0] + 1  # +1 for unknown/padding
    
    conn.close()
    
    print(f"Database counts: {num_skaters} skaters, {num_goalies} goalies (including padding)")
    return num_skaters, num_goalies


# ============================================================================
# PATH 1: GNN NETWORK MODEL (Team vs Team)
# ============================================================================

class PassingNetworkGNN(layers.Layer):
    """
    Models hockey as a graph network.
    Nodes = players on ice
    Edges = passing relationships, chemistry, spatial positioning
    """
    def __init__(self, embed_dim: int, num_gnn_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_gnn_layers = num_gnn_layers
        
        # Message passing layers
        self.gnn_layers = []
        for i in range(num_gnn_layers):
            self.gnn_layers.append({
                'message': layers.Dense(embed_dim, name=f'gnn_{i}_message'),
                'attention': layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embed_dim // num_heads,
                    dropout=dropout,
                    name=f'gnn_{i}_attn'
                ),
                'update': keras.Sequential([
                    layers.Dense(embed_dim * 4, activation='gelu'),
                    layers.Dropout(dropout),
                    layers.Dense(embed_dim)
                ], name=f'gnn_{i}_update'),
                'norm': layers.LayerNormalization(name=f'gnn_{i}_norm')
            })
        
        # Edge feature encoder (passing freq, chemistry, positioning)
        self.edge_encoder = keras.Sequential([
            layers.Dense(num_heads, activation='relu'),
            layers.Dense(num_heads)
        ], name='edge_feature_encoder')
    
    def call(self, node_features, edge_features, adjacency_mask=None, training=False):
        """
        Args:
            node_features: [batch, num_players, embed_dim] - player embeddings
            edge_features: [batch, num_players, num_players, edge_dim]
                Edges contain: pass frequency, TOI together, zone entry chains
            adjacency_mask: [batch, num_players, num_players] - valid edges
        Returns:
            [batch, num_players, embed_dim] - updated node features
        """
        x = node_features
        
        # Encode edge features into attention bias
        edge_bias = self.edge_encoder(edge_features)  # [B, P, P, num_heads]
        edge_bias = tf.transpose(edge_bias, [0, 3, 1, 2])  # [B, H, P, P]
        
        # Message passing
        for layer in self.gnn_layers:
            # Attention with edge bias
            attn_out = layer['attention'](
                query=x,
                value=x,
                attention_mask=adjacency_mask,
                training=training
            )
            
            # Could incorporate edge_bias here for weighted message passing
            # For now, using standard attention
            
            # Residual
            x = layer['norm'](x + attn_out)
            
            # Update
            update = layer['update'](x, training=training)
            x = x + update
        
        return x


class ZoneEntryPatternEncoder(layers.Layer):
    """
    Models zone entry patterns and transition play.
    Captures: dump-and-chase vs controlled entry, breakout patterns
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        # Zone entry type encoder
        self.entry_encoder = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim // 4)
        ], name='zone_entry_encoder')
        
        # Transition chain encoder (defense → center → wing)
        self.transition_encoder = layers.GRU(
            embed_dim // 4,
            return_sequences=False,
            name='transition_chain'
        )
    
    def call(self, zone_entry_features, transition_sequences=None, training=False):
        """
        Args:
            zone_entry_features: [batch, entry_dim]
                Contains: controlled entry %, dump-in %, carry-in rate
            transition_sequences: [batch, seq_len, feat_dim]
                Sequences of zone exit → neutral → zone entry
        Returns:
            [batch, embed_dim // 4] - zone entry style embedding
        """
        entry_style = self.entry_encoder(zone_entry_features, training=training)
        
        if transition_sequences is not None:
            transition_pattern = self.transition_encoder(
                transition_sequences, training=training
            )
            combined = tf.concat([entry_style, transition_pattern], axis=-1)
            return combined[:, :self.embed_dim // 4]  # Truncate to match dim
        
        return entry_style


class PowerPlayStructureEncoder(layers.Layer):
    """
    Models power play formations: 1-3-1, umbrella, overload, etc.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        # PP formation encoder
        self.formation_encoder = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim // 4)
        ], name='pp_formation')
        
        # Movement patterns (how unit rotates, passes)
        self.movement_encoder = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.Dense(embed_dim // 4)
        ], name='pp_movement')
    
    def call(self, pp_formation_features, pp_movement_features, training=False):
        """
        Args:
            pp_formation_features: [batch, form_dim]
                Formation type, positioning metrics
            pp_movement_features: [batch, move_dim]
                Pass patterns, rotation frequency
        Returns:
            [batch, embed_dim // 4]
        """
        formation = self.formation_encoder(pp_formation_features, training=training)
        movement = self.movement_encoder(pp_movement_features, training=training)
        
        return formation + movement  # Combine formation and movement


class GNNTeamModel(keras.Model):
    """
    Path 1: GNN-based team modeling
    Captures network structure, passing patterns, zone entries, PP structure
    """
    def __init__(self, num_skaters: int, num_goalies: int,
                 embed_dim: int = 128, num_gnn_layers: int = 3,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        
        # Player embeddings
        self.skater_embed = layers.Embedding(
            num_skaters, embed_dim,
            embeddings_regularizer=keras.regularizers.l2(1e-5),
            name='skater_embeddings'
        )
        
        self.goalie_embed = layers.Embedding(
            num_goalies, embed_dim,
            embeddings_regularizer=keras.regularizers.l2(1e-5),
            name='goalie_embeddings'
        )
        
        # Core GNN
        self.passing_network = PassingNetworkGNN(
            embed_dim, num_gnn_layers, num_heads=4, dropout=dropout
        )
        
        # Tactical encoders
        self.zone_entry = ZoneEntryPatternEncoder(embed_dim, dropout)
        self.pp_structure = PowerPlayStructureEncoder(embed_dim, dropout)
        
        # Team aggregation (graph-level pooling)
        self.graph_pool = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout)
        ], name='graph_pooling')
    
    def call(self, inputs, training=False):
        """
        Args:
            inputs: dict with:
                - home_skater_ids: [batch, 5]
                - home_goalie_id: [batch, 1]
                - home_edge_features: [batch, 6, 6, edge_dim]
                    Edge features: passing freq, chemistry, positioning
                - home_zone_entry_features: [batch, entry_dim]
                - home_pp_formation: [batch, form_dim]
                - home_pp_movement: [batch, move_dim]
                (Same for away)
        Returns:
            dict with home_team and away_team embeddings
        """
        # Embed players
        home_skaters = self.skater_embed(inputs['home_skater_ids'])
        home_goalie = self.goalie_embed(inputs['home_goalie_id'])
        home_players = tf.concat([home_skaters, home_goalie], axis=1)  # [B, 6, E]
        
        away_skaters = self.skater_embed(inputs['away_skater_ids'])
        away_goalie = self.goalie_embed(inputs['away_goalie_id'])
        away_players = tf.concat([away_skaters, away_goalie], axis=1)
        
        # GNN: model passing network and player interactions
        home_network = self.passing_network(
            home_players,
            inputs['home_edge_features'],
            training=training
        )
        
        away_network = self.passing_network(
            away_players,
            inputs['away_edge_features'],
            training=training
        )
        
        # Zone entry patterns
        home_entries = self.zone_entry(
            inputs['home_zone_entry_features'],
            inputs.get('home_transition_sequences'),
            training=training
        )
        
        away_entries = self.zone_entry(
            inputs['away_zone_entry_features'],
            inputs.get('away_transition_sequences'),
            training=training
        )
        
        # Power play structure
        home_pp = self.pp_structure(
            inputs['home_pp_formation'],
            inputs['home_pp_movement'],
            training=training
        )
        
        away_pp = self.pp_structure(
            inputs['away_pp_formation'],
            inputs['away_pp_movement'],
            training=training
        )
        
        # Aggregate to team level (graph pooling)
        home_pooled = tf.reduce_mean(home_network, axis=1)  # [B, E]
        away_pooled = tf.reduce_mean(away_network, axis=1)
        
        # Combine with tactical features
        home_team = self.graph_pool(tf.concat([
            home_pooled, home_entries, home_pp
        ], axis=-1), training=training)
        
        away_team = self.graph_pool(tf.concat([
            away_pooled, away_entries, away_pp
        ], axis=-1), training=training)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_network': home_network,
            'away_network': away_network
        }


# ============================================================================
# PATH 2: BAYESIAN MATCHUP MODEL (Line vs Line)
# ============================================================================

class HierarchicalBayesianMatchup(keras.Model):
    """
    Path 2: Bayesian hierarchical model for line matchups
    
    Hierarchy:
    1. League-level priors: "Checking lines vs skill lines"
    2. Team style adjustment: "Boston's system vs Edmonton's system"
    3. Line-specific effects: "Bergeron line vs McDavid line"
    
    Handles small sample sizes with proper uncertainty quantification.
    """
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        
        # League-level archetypes (learned priors)
        self.archetype_embeddings = self.add_weight(
            shape=(10, embed_dim),  # 10 archetype pairs (e.g., "skill vs shutdown")
            initializer='glorot_uniform',
            trainable=True,
            name='league_archetypes'
        )
        
        # Team style encoder
        self.team_style_encoder = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ], name='team_style')
        
        # Line composition encoder
        self.line_encoder = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ], name='line_composition')
        
        # Bayesian inference network (outputs distribution parameters)
        self.inference_net = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(64, activation='gelu'),
            layers.Dense(embed_dim * 2)  # mu and log_sigma
        ], name='bayesian_inference')
        
        # Archetype classifier (which archetype pair does this matchup resemble?)
        self.archetype_classifier = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.Dense(10, activation='softmax')  # 10 archetypes
        ], name='archetype_classifier')
    
    def call(self, inputs, training=False, return_uncertainty=True):
        """
        Args:
            inputs: dict with:
                - home_line_features: [batch, line_dim]
                    Line composition: avg age, skill level, TOI
                - away_line_features: [batch, line_dim]
                - home_team_style: [batch, style_dim]
                    System: aggressive forecheck, passive trap, etc.
                - away_team_style: [batch, style_dim]
                - matchup_history: [batch, history_dim]
                    Past encounters (if any)
        Returns:
            dict with matchup embedding and uncertainty
        """
        # Encode line compositions
        home_line = self.line_encoder(inputs['home_line_features'], training=training)
        away_line = self.line_encoder(inputs['away_line_features'], training=training)
        
        # Encode team styles
        home_style = self.team_style_encoder(inputs['home_team_style'], training=training)
        away_style = self.team_style_encoder(inputs['away_team_style'], training=training)
        
        # Determine archetype (which league-level pattern does this match?)
        matchup_context = tf.concat([
            home_line, away_line, home_style, away_style
        ], axis=-1)
        
        archetype_probs = self.archetype_classifier(matchup_context, training=training)
        # [B, 10]
        
        # Get archetype prior
        archetype_prior = tf.matmul(
            archetype_probs,
            self.archetype_embeddings
        )  # [B, embed_dim]
        
        # Bayesian inference: incorporate prior + data
        combined = tf.concat([
            home_line + home_style,
            away_line + away_style,
            archetype_prior,
            inputs.get('matchup_history', tf.zeros_like(home_line))
        ], axis=-1)
        
        # Output distribution parameters
        params = self.inference_net(combined, training=training)
        mu, log_sigma = tf.split(params, 2, axis=-1)
        sigma = tf.nn.softplus(log_sigma) + 1e-6
        
        if return_uncertainty:
            # Sample from posterior (reparameterization trick)
            if training:
                epsilon = tf.random.normal(tf.shape(mu))
                matchup_embedding = mu + sigma * epsilon
            else:
                matchup_embedding = mu  # Use mean at inference
            
            return {
                'matchup_embedding': matchup_embedding,
                'mu': mu,
                'sigma': sigma,
                'archetype_probs': archetype_probs,
                'uncertainty': tf.reduce_mean(sigma, axis=-1, keepdims=True)
            }
        else:
            return {
                'matchup_embedding': mu
            }


# ============================================================================
# MODULAR OUTPUT HEADS (Easily Changeable!)
# ============================================================================

class OutputHead(layers.Layer):
    """Base class for output heads"""
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, combined_features, training=False):
        raise NotImplementedError


class MoneylineHead(OutputHead):
    """Binary win probability for moneyline bets"""
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(name='moneyline', **kwargs)
        
        self.predictor = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(64, activation='gelu'),
            layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, combined_features, training=False):
        return self.predictor(combined_features, training=training)


class PuckLineHead(OutputHead):
    """Spread prediction (typically ±1.5 goals in NHL)"""
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(name='puck_line', **kwargs)
        
        # Mixture density network for spread
        self.mu_head = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(2)  # 2 components
        ])
        
        self.sigma_head = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(2, activation='softplus')
        ])
        
        self.pi_head = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(2, activation='softmax')
        ])
    
    def call(self, combined_features, training=False):
        mu = self.mu_head(combined_features, training=training)
        sigma = self.sigma_head(combined_features, training=training) + 1e-6
        pi = self.pi_head(combined_features, training=training)
        
        return {'mu': mu, 'sigma': sigma, 'pi': pi}


class TotalGoalsHead(OutputHead):
    """Over/under total goals"""
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(name='total_goals', **kwargs)
        
        self.predictor = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(64, activation='gelu'),
            layers.Dense(1, activation='softplus')  # Positive total
        ])
    
    def call(self, combined_features, training=False):
        return self.predictor(combined_features, training=training)


class PeriodWinnerHead(OutputHead):
    """First period winner (3-way: home/away/tie)"""
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(name='period_winner', **kwargs)
        
        self.predictor = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(64, activation='gelu'),
            layers.Dense(3, activation='softmax')  # home, away, tie
        ])
    
    def call(self, combined_features, training=False):
        return self.predictor(combined_features, training=training)


class BothTeamsScoreHead(OutputHead):
    """Both teams to score (yes/no)"""
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(name='both_teams_score', **kwargs)
        
        self.predictor = keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, combined_features, training=False):
        return self.predictor(combined_features, training=training)


class PlayerPropHead(OutputHead):
    """Player props (points, goals, assists)"""
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(name='player_props', **kwargs)
        
        # Distribution over discrete outcomes (0, 1, 2, 3+ points)
        self.predictor = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(64, activation='gelu'),
            layers.Dense(4, activation='softmax')  # 0, 1, 2, 3+ points
        ])
    
    def call(self, player_features, training=False):
        return self.predictor(player_features, training=training)


class MultiLegOutputHead(OutputHead):
    """
    Multi-leg bet output (parlays, same-game parlays)
    Jointly predicts multiple correlated outcomes
    """
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(name='multi_leg', **kwargs)
        
        # Shared representation
        self.shared = keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(128, activation='gelu')
        ])
        
        # Individual predictions (but learned jointly)
        self.moneyline = layers.Dense(1, activation='sigmoid', name='ml')
        self.total = layers.Dense(1, activation='softplus', name='total')
        self.puck_line = layers.Dense(1, activation='tanh', name='spread')
        
        # Correlation encoder
        self.correlation = layers.Dense(6, name='correlation_matrix')  # 3x3 correlation
    
    def call(self, combined_features, training=False):
        shared = self.shared(combined_features, training=training)
        
        ml = self.moneyline(shared)
        total = self.total(shared)
        spread = self.puck_line(shared)
        
        # Correlation between outcomes
        corr = self.correlation(shared)
        
        return {
            'moneyline': ml,
            'total': total,
            'spread': spread,
            'correlation': corr
        }


# ============================================================================
# COMBINED MODEL WITH MODULAR OUTPUTS
# ============================================================================

class NHLBettingModel(keras.Model):
    """
    Complete NHL betting model combining:
    - Path 1: GNN (network structure)
    - Path 2: Bayesian (line matchups)
    - Modular output heads (easily swappable)
    """
    def __init__(self, num_skaters: int, num_goalies: int,
                 embed_dim: int = 128,
                 output_heads: List[str] = ['moneyline', 'puck_line', 'total'],
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.output_heads_list = output_heads
        
        # Path 1: GNN
        self.gnn_model = GNNTeamModel(
            num_skaters, num_goalies, embed_dim, num_gnn_layers=3, dropout=dropout
        )
        
        # Path 2: Bayesian
        self.bayesian_model = HierarchicalBayesianMatchup(embed_dim, dropout)
        
        # Context encoder
        self.context_encoder = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout)
        ], name='context_encoder')
        
        # Fusion layer (combine GNN + Bayesian + context)
        self.fusion = keras.Sequential([
            layers.Dense(embed_dim * 2, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ], name='path_fusion')
        
        # Initialize output heads
        self.output_heads = {}
        for head_name in output_heads:
            if head_name == 'moneyline':
                self.output_heads[head_name] = MoneylineHead(embed_dim, dropout)
            elif head_name == 'puck_line':
                self.output_heads[head_name] = PuckLineHead(embed_dim, dropout)
            elif head_name == 'total':
                self.output_heads[head_name] = TotalGoalsHead(embed_dim, dropout)
            elif head_name == 'period_winner':
                self.output_heads[head_name] = PeriodWinnerHead(embed_dim, dropout)
            elif head_name == 'both_teams_score':
                self.output_heads[head_name] = BothTeamsScoreHead(embed_dim, dropout)
            elif head_name == 'multi_leg':
                self.output_heads[head_name] = MultiLegOutputHead(embed_dim, dropout)
    
    def call(self, inputs, training=False):
        """
        Args:
            inputs: dict with both GNN and Bayesian inputs
        Returns:
            dict with predictions from all output heads
        """
        # Path 1: GNN team representations
        gnn_outputs = self.gnn_model(inputs, training=training)
        
        # Path 2: Bayesian line matchup
        bayes_outputs = self.bayesian_model(inputs, training=training)
        
        # Encode context
        context = self.context_encoder(inputs['context_features'], training=training)
        
        # Fuse all paths
        combined = self.fusion(tf.concat([
            gnn_outputs['home_team'],
            gnn_outputs['away_team'],
            bayes_outputs['matchup_embedding'],
            context
        ], axis=-1), training=training)
        
        # Run all output heads
        predictions = {}
        for head_name, head in self.output_heads.items():
            predictions[head_name] = head(combined, training=training)
        
        # Add intermediate representations
        predictions.update({
            'gnn_home': gnn_outputs['home_team'],
            'gnn_away': gnn_outputs['away_team'],
            'bayesian_matchup': bayes_outputs['matchup_embedding'],
            'matchup_uncertainty': bayes_outputs.get('uncertainty'),
            'archetype_probs': bayes_outputs.get('archetype_probs')
        })
        
        return predictions
    
    def add_output_head(self, head_name: str, head: OutputHead):
        """Dynamically add new output head"""
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

def build_nhl_model_v2(db_path: str = "nhl_data.db",
                       output_heads: List[str] = ['moneyline', 'puck_line', 'total']):
    """
    Build NHL model with dynamic player counts from database.
    
    Args:
        db_path: Path to NHL database
        output_heads: List of output heads to include
    """
    print("="*80)
    print("NHL BETTING MODEL V2 - GNN + Bayesian")
    print("="*80)
    
    # Get actual player counts from database
    print("\nQuerying database for player counts...")
    num_skaters, num_goalies = get_player_counts(db_path)
    
    print(f"✓ Found {num_skaters} skaters")
    print(f"✓ Found {num_goalies} goalies")
    
    # Build model
    model = NHLBettingModelV2(
        num_skaters=num_skaters,
        num_goalies=num_goalies,
        embed_dim=128,
        output_heads=output_heads,
        dropout=0.1
    )
    
    print(f"\n✓ Model built with output heads: {output_heads}")
    print("\nArchitecture:")
    print("  Path 1 (GNN):")
    print("    - Passing network modeling")
    print("    - Zone entry patterns")
    print("    - Power play structure")
    print("  Path 2 (Bayesian):")
    print("    - Hierarchical line matchups")
    print("    - League archetypes → team style → specific lines")
    print("    - Uncertainty quantification")
    print("\nModular Output Heads:")
    for head in output_heads:
        print(f"    - {head}")
    print("="*80)
    
    return model


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Build with default heads
    model = build_nhl_model_v2(
        output_heads=['moneyline', 'puck_line', 'total']
    )
    
    # Later: add new output head dynamically
    print("\nAdding period winner head...")
    model.add_output_head('period_winner', PeriodWinnerHead(128))
    
    print("\nAdding multi-leg parlay head...")
    model.add_output_head('multi_leg', MultiLegOutputHead(128))
    
    print("\n✓ Model ready for training!")