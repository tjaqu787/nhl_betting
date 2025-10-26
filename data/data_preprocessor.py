"""
Complete Training Pipeline with Data Processing
Handles real NBA/sports data → features → training → backtesting
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from collections import defaultdict

from utils.training import TrainingOrchestrator

# ============================================================================
# 1. DATA PREPROCESSING PIPELINE
# ============================================================================

class SportsDataPreprocessor:
    """
    Converts raw game logs into model-ready features.
    Handles: player stats, lineups, temporal sequences, edges.
    """
    def __init__(self):
        self.player_id_map = {}  # name -> integer ID
        self.team_id_map = {}
        self.position_map = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4}
        
        self.player_history = defaultdict(list)  # rolling stats per player
        self.lineup_coplay = defaultdict(lambda: defaultdict(int))  # minutes together
        
    def fit_player_vocab(self, game_logs: pd.DataFrame):
        """Build player ID vocabulary from historical data."""
        unique_players = set()
        
        for col in ['home_player_1', 'home_player_2', 'home_player_3', 
                    'home_player_4', 'home_player_5',
                    'away_player_1', 'away_player_2', 'away_player_3',
                    'away_player_4', 'away_player_5']:
            if col in game_logs.columns:
                unique_players.update(game_logs[col].dropna().unique())
        
        # Assign IDs (0 reserved for unknown/average player)
        self.player_id_map = {
            player: idx + 1 
            for idx, player in enumerate(sorted(unique_players))
        }
        self.player_id_map['<UNKNOWN>'] = 0
        
        print(f"Mapped {len(self.player_id_map)} players")
        
    def compute_player_features(self, player_name: str, game_date: datetime,
                                recent_games: pd.DataFrame) -> np.ndarray:
        """
        Compute per-player features from recent history.
        
        Returns: [feat_dim] array with:
          - Per-minute stats (PTS, REB, AST, etc.)
          - Usage rate
          - Efficiency metrics
          - Fatigue proxies
          - Opponent matchup history
        """
        # Filter to player's recent games before this date
        player_games = recent_games[
            (recent_games['player_name'] == player_name) &
            (recent_games['game_date'] < game_date)
        ].sort_values('game_date', ascending=False).head(10)
        
        if len(player_games) == 0:
            # Cold start - return zeros
            return np.zeros(32)
        
        features = []
        
        # Last 10 games averages (per 36 minutes)
        features.append(player_games['points'].mean() * 36 / (player_games['minutes'].mean() + 1))
        features.append(player_games['rebounds'].mean() * 36 / (player_games['minutes'].mean() + 1))
        features.append(player_games['assists'].mean() * 36 / (player_games['minutes'].mean() + 1))
        features.append(player_games['steals'].mean() * 36 / (player_games['minutes'].mean() + 1))
        features.append(player_games['blocks'].mean() * 36 / (player_games['minutes'].mean() + 1))
        features.append(player_games['turnovers'].mean() * 36 / (player_games['minutes'].mean() + 1))
        features.append(player_games['fg_pct'].mean())
        features.append(player_games['three_pct'].mean())
        features.append(player_games['ft_pct'].mean())
        
        # Usage rate
        features.append(player_games['usage_rate'].mean())
        
        # Efficiency metrics
        features.append(player_games['plus_minus'].mean())
        features.append(player_games['per'].mean())  # Player Efficiency Rating
        
        # Recent form (last 3 games vs last 10 games)
        recent_3 = player_games.head(3)['points'].mean()
        recent_10 = player_games['points'].mean()
        features.append((recent_3 - recent_10) / (recent_10 + 1))  # Form trend
        
        # Fatigue proxies
        features.append(player_games.head(3)['minutes'].sum())  # Last 3 games minutes
        features.append(player_games.head(7)['minutes'].sum())  # Last 7 games minutes
        
        # Days rest
        if len(player_games) >= 2:
            last_game_date = player_games.iloc[0]['game_date']
            days_rest = (game_date - last_game_date).days
            features.append(min(days_rest, 7))  # Cap at 7
        else:
            features.append(3)  # Default
        
        # Home/away splits (if available)
        home_games = player_games[player_games['home_away'] == 'home']
        away_games = player_games[player_games['home_away'] == 'away']
        features.append(home_games['plus_minus'].mean() if len(home_games) > 0 else 0)
        features.append(away_games['plus_minus'].mean() if len(away_games) > 0 else 0)
        
        # Position encoding (one-hot flattened)
        position = player_games.iloc[0].get('position', 'SF')
        pos_vector = [0] * 5
        pos_vector[self.position_map.get(position, 2)] = 1
        features.extend(pos_vector)
        
        # Pad to fixed size
        features = features[:32]
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def compute_temporal_history(self, player_name: str, game_date: datetime,
                                 recent_games: pd.DataFrame, 
                                 seq_len: int = 10) -> np.ndarray:
        """
        Get sequence of features for last N games (for GRU).
        Returns: [seq_len, feat_dim]
        """
        player_games = recent_games[
            (recent_games['player_name'] == player_name) &
            (recent_games['game_date'] < game_date)
        ].sort_values('game_date', ascending=False).head(seq_len)
        
        history = []
        for _, game in player_games.iterrows():
            game_features = [
                game['points'], game['rebounds'], game['assists'],
                game['minutes'], game['plus_minus'], game['fg_pct'],
                game.get('usage_rate', 0.2), game.get('per', 15.0)
            ]
            # Pad to 32
            game_features.extend([0.0] * (32 - len(game_features)))
            history.append(game_features[:32])
        
        # Pad sequence if needed
        while len(history) < seq_len:
            history.append([0.0] * 32)
        
        return np.array(history, dtype=np.float32)
    
    def compute_edge_features(self, lineup_players: List[str],
                                 coplay_stats: Dict) -> np.ndarray:
        """
        Compute edge features for lineup graph (co-play time, chemistry).
        Returns: [num_players, num_players, edge_dim]
        """
        num_players = len(lineup_players)
        edge_features = np.zeros((num_players, num_players, 8), dtype=np.float32)
        
        for i, player_i in enumerate(lineup_players):
            for j, player_j in enumerate(lineup_players):
                if i == j:
                    continue
                
                # Co-play minutes this season
                coplay_key = tuple(sorted([player_i, player_j]))
                coplay_min = coplay_stats.get(coplay_key, {}).get('minutes', 0)
                edge_features[i, j, 0] = min(coplay_min / 1000.0, 1.0)  # Normalize
                
                # Net rating when together
                edge_features[i, j, 1] = coplay_stats.get(coplay_key, {}).get('net_rating', 0) / 20.0
                
                # On-court vs off-court differential
                edge_features[i, j, 2] = coplay_stats.get(coplay_key, {}).get('on_off_diff', 0) / 10.0
                
                # Offensive synergy
                edge_features[i, j, 3] = coplay_stats.get(coplay_key, {}).get('off_rating', 110) / 120.0
                
                # Defensive synergy
                edge_features[i, j, 4] = coplay_stats.get(coplay_key, {}).get('def_rating', 110) / 120.0
                
                # Pace when together
                edge_features[i, j, 5] = coplay_stats.get(coplay_key, {}).get('pace', 100) / 110.0
                
                # Position compatibility (guards + bigs = good spacing)
                pos_i = self._get_position_type(player_i)
                pos_j = self._get_position_type(player_j)
                edge_features[i, j, 6] = float(pos_i != pos_j)  # Different positions
                
                # Recent co-play (last 5 games)
                edge_features[i, j, 7] = coplay_stats.get(coplay_key, {}).get('recent_minutes', 0) / 200.0
        
        return edge_features
    
    def _get_position_type(self, player_name: str) -> str:
        """Guard vs Big classification."""
        # Simplified - in production, use actual position data
        return 'guard' if hash(player_name) % 2 == 0 else 'big'
    
    def extract_market_features(self, game_row: pd.Series) -> np.ndarray:
        """
        Extract bookmaker line features.
        
        Returns: [market_dim] with:
          - Opening spread, current spread, line movement
          - Opening total, current total, total movement
          - Moneyline odds (both sides)
          - Public betting percentages
          - Sharp money indicators
          - Market consensus
        """
        features = []
        
        # Spread features
        opening_spread = game_row.get('opening_spread', 0.0)
        current_spread = game_row.get('current_spread', 0.0)
        features.append(current_spread)
        features.append(opening_spread)
        features.append(current_spread - opening_spread)  # Movement
        
        # Total features
        opening_total = game_row.get('opening_total', 220.0)
        current_total = game_row.get('current_total', 220.0)
        features.append(current_total / 250.0)  # Normalize
        features.append(opening_total / 250.0)
        features.append((current_total - opening_total) / 10.0)
        
        # Moneyline odds (convert to implied probability)
        home_ml = game_row.get('home_moneyline', -110)
        away_ml = game_row.get('away_moneyline', -110)
        home_implied = self._moneyline_to_prob(home_ml)
        away_implied = self._moneyline_to_prob(away_ml)
        features.append(home_implied)
        features.append(away_implied)
        
        # Public betting percentages
        features.append(game_row.get('public_bet_pct_home', 0.5))
        features.append(game_row.get('public_money_pct_home', 0.5))
        
        # Sharp money indicators (reverse line movement)
        public_on_home = game_row.get('public_bet_pct_home', 0.5) > 0.65
        line_moved_away = (current_spread - opening_spread) < -0.5
        features.append(float(public_on_home and line_moved_away))  # RLM signal
        
        # Market consensus (average across multiple books)
        features.append(game_row.get('consensus_spread', current_spread))
        features.append(game_row.get('spread_std', 0.5))  # Disagreement
        
        # Time until game (urgency)
        features.append(game_row.get('hours_until_game', 24.0) / 48.0)
        
        # Pad to 16 features
        while len(features) < 16:
            features.append(0.0)
        
        return np.array(features[:16], dtype=np.float32)
    
    def _moneyline_to_prob(self, moneyline: float) -> float:
        """Convert American odds to implied probability."""
        if moneyline > 0:
            return 100.0 / (moneyline + 100.0)
        else:
            return abs(moneyline) / (abs(moneyline) + 100.0)
    
    def extract_context_features(self, game_row: pd.Series) -> np.ndarray:
        """
        Game context features: home/away, rest, travel, schedule, motivation.
        Returns: [24]
        """
        features = []
        
        # Home court advantage (binary + strength)
        features.append(1.0)  # Home team perspective
        home_record = game_row.get('home_record_home', 0.5)
        features.append(home_record)
        
        # Rest days for each team
        home_rest = game_row.get('home_days_rest', 2)
        away_rest = game_row.get('away_days_rest', 2)
        features.append(min(home_rest, 7) / 7.0)
        features.append(min(away_rest, 7) / 7.0)
        
        # Back-to-back flags
        features.append(float(home_rest == 0))
        features.append(float(away_rest == 0))
        
        # Travel distance for away team
        travel_miles = game_row.get('away_travel_miles', 1000)
        features.append(min(travel_miles / 3000.0, 1.0))
        
        # Schedule position (games played / total games)
        home_games_played = game_row.get('home_games_played', 41)
        away_games_played = game_row.get('away_games_played', 41)
        features.append(home_games_played / 82.0)
        features.append(away_games_played / 82.0)
        
        # Playoff positioning motivation
        home_playoff_spot = game_row.get('home_playoff_position', 8)
        away_playoff_spot = game_row.get('away_playoff_position', 8)
        features.append(self._motivation_score(home_playoff_spot, home_games_played))
        features.append(self._motivation_score(away_playoff_spot, away_games_played))
        
        # Recent performance (last 10 games)
        features.append(game_row.get('home_last_10_wins', 5) / 10.0)
        features.append(game_row.get('away_last_10_wins', 5) / 10.0)
        
        # Head-to-head this season
        features.append(game_row.get('h2h_home_wins', 0) / 4.0)
        features.append(game_row.get('h2h_away_wins', 0) / 4.0)
        
        # Season phase (early/mid/late/playoffs)
        season_phase = self._get_season_phase(home_games_played)
        phase_onehot = [0.0] * 4
        phase_onehot[season_phase] = 1.0
        features.extend(phase_onehot)
        
        # Altitude (for specific venues)
        features.append(game_row.get('altitude_feet', 0) / 5000.0)
        
        # Referee crew (if available - use historical foul rate)
        features.append(game_row.get('referee_foul_rate', 0.22))
        
        # Pad to 24
        while len(features) < 24:
            features.append(0.0)
        
        return np.array(features[:24], dtype=np.float32)
    
    def _motivation_score(self, playoff_position: int, games_played: int) -> float:
        """
        Compute motivation factor based on playoff race.
        High motivation if: fighting for playoff spot, or jockeying for seeding late season.
        """
        if games_played < 60:
            return 0.5  # Early season, moderate motivation
        
        # Late season
        if 7 <= playoff_position <= 10:
            return 1.0  # Fighting for playoff spot
        elif 1 <= playoff_position <= 3:
            return 0.9  # Fighting for home court
        elif playoff_position > 12:
            return 0.3  # Tanking incentive
        else:
            return 0.6
    
    def _get_season_phase(self, games_played: int) -> int:
        """0: early, 1: mid, 2: late, 3: playoffs"""
        if games_played < 20:
            return 0
        elif games_played < 60:
            return 1
        elif games_played <= 82:
            return 2
        else:
            return 3
    
    def process_game(self, game_row: pd.Series, 
                    player_stats_db: pd.DataFrame,
                    coplay_db: Dict) -> Dict:
        """
        Complete preprocessing for a single game.
        Returns model-ready inputs.
        """
        game_date = pd.to_datetime(game_row['game_date'])
        
        # Extract lineups
        home_lineup = [
            game_row.get(f'home_player_{i}', '<UNKNOWN>') 
            for i in range(1, 6)
        ]
        away_lineup = [
            game_row.get(f'away_player_{i}', '<UNKNOWN>')
            for i in range(1, 6)
        ]
        
        # Convert to IDs
        home_ids = np.array([
            self.player_id_map.get(p, 0) for p in home_lineup
        ])
        away_ids = np.array([
            self.player_id_map.get(p, 0) for p in away_lineup
        ])
        
        # Compute features for each player
        home_features = np.stack([
            self.compute_player_features(p, game_date, player_stats_db)
            for p in home_lineup
        ])
        away_features = np.stack([
            self.compute_player_features(p, game_date, player_stats_db)
            for p in away_lineup
        ])
        
        # Temporal histories
        home_temporal = np.stack([
            self.compute_temporal_history(p, game_date, player_stats_db)
            for p in home_lineup
        ])
        away_temporal = np.stack([
            self.compute_temporal_history(p, game_date, player_stats_db)
            for p in away_lineup
        ])
        
        # Edge features
        home_edges = self.compute_edge_features(home_lineup, coplay_db)
        away_edges = self.compute_edge_features(away_lineup, coplay_db)
        
        # Market and context
        market_features = self.extract_market_features(game_row)
        context_features = self.extract_context_features(game_row)
        
        # Targets
        home_won = float(game_row.get('home_score', 0) > game_row.get('away_score', 0))
        spread = game_row.get('home_score', 0) - game_row.get('away_score', 0)
        odds = game_row.get('home_odds_decimal', 2.0)
        
        return {
            'inputs': {
                'home_player_ids': home_ids,
                'away_player_ids': away_ids,
                'home_features': home_features,
                'away_features': away_features,
                'home_temporal_history': home_temporal,
                'away_temporal_history': away_temporal,
                'home_edge_features': home_edges,
                'away_edge_features': away_edges,
                'market_features': market_features,
                'context_features': context_features,
            },
            'targets': {
                'win': home_won,
                'spread': spread,
                'odds': odds
            },
            'metadata': {
                'game_id': game_row.get('game_id', ''),
                'game_date': game_date,
                'home_team': game_row.get('home_team', ''),
                'away_team': game_row.get('away_team', '')
            }
        }

