"""
NHL Data Preprocessing Pipeline
Handles real NHL data from database → features → training
Uses actual database schema with teams, games, player_game_stats, etc.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Import query functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.queries import (
    get_training_data,
    get_team_rolling_window,
    get_player_rolling_window,
    get_team_game_rolling_window,
    get_h2h_rolling_window,
    get_rest_days_rolling_window,
    db_path
)


# ============================================================================
# 1. NHL DATA PREPROCESSOR
# ============================================================================

class DataPreprocessor:
    """
    Converts raw NHL game data into model-ready features.
    Handles: player stats, lineups, temporal sequences, team features.
    """
    def __init__(self, db_path: str = "data/nhl_data.db"):
        self.db_path = db_path
        self.conn = None
        
        # Mappings
        self.player_id_map = {}  # player_id -> integer index
        self.team_id_map = {}    # team_abbrev -> integer index
        self.position_map = {'C': 0, 'L': 1, 'R': 2, 'D': 3, 'G': 4}  # Center, Left, Right, Defense, Goalie
        
        # Cache for player/team data
        self.player_cache = {}
        self.team_cache = {}
        
    def connect(self):
        """Open database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def fit_player_vocab(self, min_games: int = 5):
        """
        Build player ID vocabulary from database.
        Only include players with minimum number of games.
        """
        self.connect()
        
        query = """
        SELECT DISTINCT player_id, COUNT(*) as games
        FROM player_game_stats
        GROUP BY player_id
        HAVING games >= ?
        ORDER BY games DESC
        """
        
        df = pd.read_sql_query(query, self.conn, params=(min_games,))
        
        # Assign IDs (0 reserved for unknown/replacement player)
        self.player_id_map = {
            int(player_id): idx + 1 
            for idx, player_id in enumerate(df['player_id'])
        }
        self.player_id_map[0] = 0  # Unknown player
        
        print(f"✓ Mapped {len(self.player_id_map)} players (min {min_games} games)")
        
        # Also build team mapping
        team_query = "SELECT DISTINCT team_abbrev FROM teams ORDER BY team_abbrev"
        teams = pd.read_sql_query(team_query, self.conn)
        
        self.team_id_map = {
            team: idx 
            for idx, team in enumerate(teams['team_abbrev'])
        }
        
        print(f"✓ Mapped {len(self.team_id_map)} teams")
    
    def get_team_roster(self, team_abbrev: str, game_date: str, 
                       lookback_days: int = 30) -> List[int]:
        """
        Get active roster for a team around a specific date.
        Returns list of player_ids.
        """
        self.connect()
        
        # Get players who played for this team recently
        start_date = (pd.to_datetime(game_date) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        query = """
        SELECT DISTINCT pgs.player_id, COUNT(*) as games
        FROM player_game_stats pgs
        JOIN games g ON pgs.game_id = g.game_id
        WHERE pgs.team_abbrev = ?
        AND g.game_date <= ?
        AND g.game_date >= ?
        AND g.game_state IN ('OFF', 'FINAL')
        GROUP BY pgs.player_id
        ORDER BY games DESC
        LIMIT 20
        """
        
        df = pd.read_sql_query(query, self.conn, params=(team_abbrev, game_date, start_date))
        
        return df['player_id'].tolist()
    
    def compute_player_features(self, player_id: int, game_date: str,
                                lookback_games: int = 10) -> np.ndarray:
        """
        Compute per-player features from recent history.
        
        Returns: [feat_dim] array with:
          - Per-game stats (goals, assists, points, shots, +/-, TOI)
          - Efficiency metrics (shooting %, points per 60)
          - Special teams stats (PP goals, SH goals)
          - Physical stats (hits, blocks)
          - Recent form trend
          - Position encoding
        """
        # Get recent games for this player
        player_games = get_player_rolling_window(player_id, game_date, lookback_games)
        
        if len(player_games) == 0:
            # Cold start - return zeros
            return np.zeros(32)
        
        features = []
        
        # Basic stats (per game averages)
        features.append(player_games['goals'].mean())
        features.append(player_games['assists'].mean())
        features.append(player_games['points'].mean())
        features.append(player_games['shots'].mean())
        features.append(player_games['plus_minus'].mean())
        
        # Time on ice (convert MM:SS to minutes if needed)
        toi_values = []
        for toi in player_games['toi']:
            if toi and isinstance(toi, str) and ':' in toi:
                parts = toi.split(':')
                minutes = int(parts[0]) + int(parts[1]) / 60.0
                toi_values.append(minutes)
            elif toi:
                toi_values.append(float(toi))
        
        avg_toi = np.mean(toi_values) if toi_values else 15.0
        features.append(avg_toi)
        
        # Efficiency metrics
        total_shots = player_games['shots'].sum()
        shooting_pct = player_games['goals'].sum() / max(total_shots, 1)
        features.append(shooting_pct)
        
        # Points per 60 minutes
        total_toi = sum(toi_values) if toi_values else 1.0
        points_per_60 = player_games['points'].sum() / max(total_toi / 60.0, 0.1)
        features.append(points_per_60)
        
        # Special teams
        features.append(player_games['powerplay_goals'].mean())
        
        # Physical play
        features.append(player_games['hits'].mean())
        features.append(player_games['blocked'].mean())
        
        # Recent form (last 3 games vs last 10 games)
        if len(player_games) >= 3:
            recent_3_pts = player_games.head(3)['points'].mean()
            all_pts = player_games['points'].mean()
            form_trend = (recent_3_pts - all_pts) / (all_pts + 0.1)
        else:
            form_trend = 0.0
        features.append(form_trend)
        
        # Consistency (std dev of points)
        features.append(player_games['points'].std())
        
        # Games played (as proxy for fatigue/availability)
        features.append(min(len(player_games) / 10.0, 1.0))
        
        # Position encoding (one-hot)
        if len(player_games) > 0:
            position = player_games.iloc[0]['position']
            # Extract first character (C, L, R, D, G)
            pos_char = position[0] if position else 'C'
            pos_idx = self.position_map.get(pos_char, 0)
        else:
            pos_idx = 0
        
        pos_vector = [0.0] * 5
        pos_vector[pos_idx] = 1.0
        features.extend(pos_vector)
        
        # Recent shot volume (indicator of offensive role)
        features.append(player_games.head(5)['shots'].sum())
        
        # Plus/minus trend
        if len(player_games) >= 5:
            recent_pm = player_games.head(5)['plus_minus'].mean()
            overall_pm = player_games['plus_minus'].mean()
            pm_trend = recent_pm - overall_pm
        else:
            pm_trend = 0.0
        features.append(pm_trend)
        
        # Pad or truncate to fixed size
        features = features[:32]
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def compute_temporal_history(self, player_id: int, game_date: str,
                                 lookback_games: int = 10) -> np.ndarray:
        """
        Get sequence of features for last N games (for GRU/LSTM).
        Returns: [seq_len, feat_dim]
        """
        player_games = get_player_rolling_window(player_id, game_date, lookback_games)
        
        history = []
        for _, game in player_games.iterrows():
            # Convert TOI to float
            toi = game['toi']
            if toi and isinstance(toi, str) and ':' in toi:
                parts = toi.split(':')
                toi_minutes = int(parts[0]) + int(parts[1]) / 60.0
            else:
                toi_minutes = 15.0
            
            game_features = [
                game['goals'],
                game['assists'],
                game['points'],
                game['shots'],
                game['plus_minus'],
                toi_minutes,
                game['powerplay_goals'],
                game['hits'],
                game['blocked']
            ]
            
            # Pad to 32
            game_features.extend([0.0] * (32 - len(game_features)))
            history.append(game_features[:32])
        
        # Pad sequence if needed
        while len(history) < lookback_games:
            history.append([0.0] * 32)
        
        return np.array(history, dtype=np.float32)
    
    def compute_team_features(self, team_abbrev: str, game_date: str,
                             lookback_games: int = 10) -> np.ndarray:
        """
        Compute team-level features from recent performance.
        
        Returns: [team_feat_dim] with:
          - Win percentage
          - Goals for/against
          - Shot metrics
          - Special teams efficiency
          - Recent form
        """
        team_games = get_team_rolling_window(team_abbrev, game_date, lookback_games)
        
        if len(team_games) == 0:
            return np.zeros(24)
        
        features = []
        
        # Win rate
        features.append(team_games['win'].mean())
        
        # Scoring
        features.append(team_games['goals_for'].mean())
        features.append(team_games['goals_against'].mean())
        features.append(team_games['goal_differential'].mean())
        
        # Shot metrics
        features.append(team_games['shots_for'].mean())
        features.append(team_games['shots_against'].mean())
        features.append(team_games['shooting_pct'].mean())
        features.append(team_games['save_pct'].mean())
        
        # Home/away splits
        home_games = team_games[team_games['is_home'] == 1]
        away_games = team_games[team_games['is_home'] == 0]
        
        features.append(home_games['win'].mean() if len(home_games) > 0 else 0.5)
        features.append(away_games['win'].mean() if len(away_games) > 0 else 0.5)
        
        # Recent form (last 5 games)
        if len(team_games) >= 5:
            recent_5_wins = team_games.head(5)['win'].sum()
            features.append(recent_5_wins / 5.0)
        else:
            features.append(0.5)
        
        # Goal differential trend
        if len(team_games) >= 5:
            recent_gd = team_games.head(5)['goal_differential'].mean()
            overall_gd = team_games['goal_differential'].mean()
            gd_trend = recent_gd - overall_gd
        else:
            gd_trend = 0.0
        features.append(gd_trend)
        
        # Consistency (std of goals)
        features.append(team_games['goals_for'].std())
        features.append(team_games['goals_against'].std())
        
        # Get team_game_stats for more detailed metrics
        team_game_stats = get_team_game_rolling_window(team_abbrev, game_date, lookback_games)
        
        if len(team_game_stats) > 0:
            # Power play efficiency
            pp_goals = team_game_stats['powerplay_goals'].sum()
            pp_opps = team_game_stats['powerplay_opportunities'].sum()
            pp_pct = pp_goals / max(pp_opps, 1)
            features.append(pp_pct)
            
            # Faceoff win percentage
            features.append(team_game_stats['faceoff_win_pct'].mean())
            
            # Physical metrics
            features.append(team_game_stats['hits'].mean())
            features.append(team_game_stats['blocked'].mean())
            
            # Discipline (penalty minutes)
            features.append(team_game_stats['pim'].mean())
            
            # Possession metrics
            features.append(team_game_stats['giveaways'].mean())
            features.append(team_game_stats['takeaways'].mean())
        else:
            # Default values
            features.extend([0.2, 0.5, 20.0, 15.0, 8.0, 8.0, 8.0])
        
        # Pad to 24
        features = features[:24]
        while len(features) < 24:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def compute_context_features(self, game_row: pd.Series) -> np.ndarray:
        """
        Game context features: home/away, rest, head-to-head, schedule position.
        Returns: [24]
        """
        features = []
        
        # Home advantage (always 1.0 for home team perspective)
        features.append(1.0)
        
        # Rest days
        game_id = game_row['game_id']
        home_team = game_row['home_team']
        away_team = game_row['away_team']
        game_date = game_row['game_date']
        
        # Get rest days from database
        self.connect()
        rest_query = """
        SELECT rest_days 
        FROM team_rest_days 
        WHERE team = ? AND game_id = ?
        """
        
        home_rest = pd.read_sql_query(rest_query, self.conn, 
                                       params=(home_team, game_id))
        away_rest = pd.read_sql_query(rest_query, self.conn,
                                       params=(away_team, game_id))
        
        home_rest_days = home_rest['rest_days'].iloc[0] if len(home_rest) > 0 else 2
        away_rest_days = away_rest['rest_days'].iloc[0] if len(away_rest) > 0 else 2
        
        features.append(min(home_rest_days / 7.0, 1.0) if home_rest_days else 0.3)
        features.append(min(away_rest_days / 7.0, 1.0) if away_rest_days else 0.3)
        
        # Back-to-back flags
        features.append(1.0 if home_rest_days == 1 else 0.0)
        features.append(1.0 if away_rest_days == 1 else 0.0)
        
        # Head-to-head record
        h2h = get_h2h_rolling_window(home_team, away_team, game_date, lookback_games=10)
        
        if len(h2h) > 0:
            home_h2h_wins = h2h[h2h['team'] == home_team]['win'].sum()
            away_h2h_wins = h2h[h2h['team'] == away_team]['win'].sum()
            total_h2h = len(h2h)
            
            features.append(home_h2h_wins / max(total_h2h, 1))
            features.append(away_h2h_wins / max(total_h2h, 1))
        else:
            features.extend([0.5, 0.5])
        
        # Season progress (games played / 82)
        games_query = """
        SELECT COUNT(*) as games
        FROM games g
        WHERE (g.home_team = ? OR g.away_team = ?)
        AND g.game_date < ?
        AND g.game_state IN ('OFF', 'FINAL')
        """
        
        home_games = pd.read_sql_query(games_query, self.conn,
                                        params=(home_team, home_team, game_date))
        away_games = pd.read_sql_query(games_query, self.conn,
                                        params=(away_team, away_team, game_date))
        
        home_games_played = home_games['games'].iloc[0] if len(home_games) > 0 else 0
        away_games_played = away_games['games'].iloc[0] if len(away_games) > 0 else 0
        
        features.append(home_games_played / 82.0)
        features.append(away_games_played / 82.0)
        
        # Get team rolling stats for recent performance
        home_stats = get_team_rolling_window(home_team, game_date, lookback_games=10)
        away_stats = get_team_rolling_window(away_team, game_date, lookback_games=10)
        
        if len(home_stats) >= 5:
            home_last_5 = home_stats.head(5)['win'].sum() / 5.0
        else:
            home_last_5 = 0.5
        
        if len(away_stats) >= 5:
            away_last_5 = away_stats.head(5)['win'].sum() / 5.0
        else:
            away_last_5 = 0.5
        
        features.append(home_last_5)
        features.append(away_last_5)
        
        # Pad to 24
        features = features[:24]
        while len(features) < 24:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def process_game(self, game_row: pd.Series, 
                    num_players_per_team: int = 18) -> Dict:
        """
        Complete preprocessing for a single game.
        Returns model-ready inputs.
        
        Args:
            game_row: Row from games table
            num_players_per_team: Number of roster players to include
        
        Returns:
            Dictionary with inputs, targets, and metadata
        """
        game_date = game_row['game_date']
        home_team = game_row['home_team']
        away_team = game_row['away_team']
        
        # Get rosters (active players around game time)
        home_roster = self.get_team_roster(home_team, game_date)[:num_players_per_team]
        away_roster = self.get_team_roster(away_team, game_date)[:num_players_per_team]
        
        # Pad rosters if needed
        while len(home_roster) < num_players_per_team:
            home_roster.append(0)
        while len(away_roster) < num_players_per_team:
            away_roster.append(0)
        
        # Convert to model indices
        home_ids = np.array([
            self.player_id_map.get(pid, 0) for pid in home_roster
        ])
        away_ids = np.array([
            self.player_id_map.get(pid, 0) for pid in away_roster
        ])
        
        # Compute features for each player
        home_features = np.stack([
            self.compute_player_features(pid, game_date) 
            for pid in home_roster
        ])
        away_features = np.stack([
            self.compute_player_features(pid, game_date)
            for pid in away_roster
        ])
        
        # Temporal histories
        home_temporal = np.stack([
            self.compute_temporal_history(pid, game_date)
            for pid in home_roster
        ])
        away_temporal = np.stack([
            self.compute_temporal_history(pid, game_date)
            for pid in away_roster
        ])
        
        # Team-level features
        home_team_features = self.compute_team_features(home_team, game_date)
        away_team_features = self.compute_team_features(away_team, game_date)
        
        # Context features
        context_features = self.compute_context_features(game_row)
        
        # Market features (placeholder - would come from betting sites)
        market_features = np.zeros(16, dtype=np.float32)
        
        # Edge features (co-play stats - simplified for NHL)
        # In full implementation, would track line combinations
        home_edges = np.zeros((num_players_per_team, num_players_per_team, 8), 
                              dtype=np.float32)
        away_edges = np.zeros((num_players_per_team, num_players_per_team, 8),
                              dtype=np.float32)
        
        # Targets
        home_score = game_row.get('home_score', 0)
        away_score = game_row.get('away_score', 0)
        
        home_won = 1.0 if home_score > away_score else 0.0
        spread = home_score - away_score
        
        # Placeholder odds (would come from market)
        odds = 2.0
        
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
                'home_team_features': home_team_features,
                'away_team_features': away_team_features,
                'context_features': context_features,
                'market_metadata': market_features,
            },
            'targets': {
                'win': home_won,
                'spread': spread,
                'odds': odds,
                'total_goals': home_score + away_score
            },
            'metadata': {
                'game_id': game_row['game_id'],
                'game_date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score
            }
        }
    
    def prepare_training_dataset(self, 
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                lookback_games: int = 10) -> List[Dict]:
        """
        Prepare complete training dataset using database queries.
        
        Returns:
            List of processed game dictionaries
        """
        print("\n" + "="*60)
        print("PREPARING TRAINING DATASET")
        print("="*60)
        
        # Get training data using existing query function
        df = get_training_data(start_date, end_date, lookback_games)
        
        print(f"✓ Loaded {len(df)} games from database")
        print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        
        # Process each game
        processed_games = []
        
        for idx, row in df.iterrows():
            try:
                processed = self.process_game(row)
                processed_games.append(processed)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} games...")
            
            except Exception as e:
                print(f"  ✗ Error processing game {row['game_id']}: {e}")
                continue
        
        print(f"✓ Successfully processed {len(processed_games)} games")
        print("="*60 + "\n")
        
        return processed_games

# ============================================================================
# REFEREE DATA PREPROCESSING FUNCTIONS
# ============================================================================

def compute_ref_team_history(home_team: str, away_team: str, 
                             ref_names: List[str], game_date: str,
                             db_path: str = "data/nhl_data.db",
                             lookback_games: int = 50) -> np.ndarray:
    """
    Compute historical team-ref interaction stats.
    
    Returns: [6] features:
        - home_win_rate_with_refs
        - away_win_rate_with_refs
        - home_pim_rate_with_refs
        - away_pim_rate_with_refs
        - ref_home_bias_general
        - games_with_these_refs_together
    """
    import pandas as pd
    
    conn = sqlite3.connect(db_path)
    
    features = []
    
    # Home team with these refs
    home_query = """
    SELECT AVG(CASE WHEN g.home_team = ? AND g.home_score > g.away_score THEN 1
                    WHEN g.away_team = ? AND g.away_score > g.home_score THEN 1
                    ELSE 0 END) as win_rate,
           AVG(tgs.pim) as avg_pim
    FROM games g
    JOIN game_officials go ON g.game_id = go.game_id
    JOIN team_game_stats tgs ON g.game_id = tgs.game_id 
        AND (tgs.team_abbrev = ? OR tgs.team_abbrev = ?)
    WHERE go.official_name IN ({})
    AND g.game_date < ?
    AND g.game_date > date(?, '-{} days')
    """.format(','.join(['?']*len(ref_names)), lookback_games * 2)
    
    home_stats = pd.read_sql_query(home_query, conn, 
        params=[home_team, home_team, home_team, home_team] + 
               ref_names + [game_date, game_date])
    
    # Away team with these refs
    away_query = home_query  # Same query structure for away team
    away_stats = pd.read_sql_query(away_query, conn, 
        params=[away_team, away_team, away_team, away_team] + 
               ref_names + [game_date, game_date])
    
    # Ref general home bias
    bias_query = """
    SELECT 
        AVG(CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END) - 0.55 as home_bias
    FROM games g
    JOIN game_officials go ON g.game_id = go.game_id
    WHERE go.official_name IN ({})
    AND g.game_date < ?
    """.format(','.join(['?']*len(ref_names)))
    
    ref_bias = pd.read_sql_query(bias_query, conn, 
        params=ref_names + [game_date])
    
    features.append(home_stats['win_rate'].iloc[0] if len(home_stats) > 0 else 0.5)
    features.append(away_stats['win_rate'].iloc[0] if len(away_stats) > 0 else 0.5)
    features.append(home_stats['avg_pim'].iloc[0] if len(home_stats) > 0 else 8.0)
    features.append(away_stats['avg_pim'].iloc[0] if len(away_stats) > 0 else 8.0)
    features.append(ref_bias['home_bias'].iloc[0] if len(ref_bias) > 0 else 0.0)
    features.append(float(len(home_stats)))  # Sample size indicator
    
    conn.close()
    return np.array(features, dtype=np.float32)


def compute_player_ref_style(player_id: int, game_date: str,
                             db_path: str = "data/nhl_data.db", 
                             lookback_games: int = 20) -> np.ndarray:
    """
    Compute player style features relevant to ref interactions.
    
    Returns: [8] features:
        - physicality (hits + blocked per game)
        - pim_rate (penalties per game)
        - pp_usage (% of PP time)
        - pk_usage (% of PK time)
        - finesse_score (points per physical play)
        - defensive_focus (def zone starts %)
        - agitator_score (pim + hits normalized)
        - discipline (pim variance)
    """
    import pandas as pd
    
    conn = sqlite3.connect(db_path)
    
    # Get recent player games
    query = """
    SELECT hits, blocked, pim, powerplay_goals, powerplay_assists,
           shorthanded_goals, shorthanded_assists, points, shots
    FROM player_game_stats
    WHERE player_id = ? AND game_date < ?
    ORDER BY game_date DESC
    LIMIT ?
    """
    
    player_games = pd.read_sql_query(query, conn, 
        params=[player_id, game_date, lookback_games])
    
    conn.close()
    
    if len(player_games) == 0:
        return np.zeros(8, dtype=np.float32)
    
    features = []
    
    # Physicality
    features.append(player_games['hits'].mean() + player_games['blocked'].mean())
    
    # PIM rate
    features.append(player_games['pim'].mean())
    
    # PP usage (from pp_goals/assists)
    pp_rate = (player_games['powerplay_goals'] + player_games['powerplay_assists']).sum()
    features.append(pp_rate / max(len(player_games), 1))
    
    # PK usage (estimated from shorthanded stats)
    pk_rate = (player_games['shorthanded_goals'] + player_games['shorthanded_assists']).sum()
    features.append(pk_rate / max(len(player_games), 1))
    
    # Finesse score (offensive skill vs physical play)
    total_physical = player_games['hits'].sum() + player_games['blocked'].sum()
    finesse = player_games['points'].sum() / max(total_physical, 1)
    features.append(finesse)
    
    # Defensive focus (proxy: blocked shots / total shots)
    def_focus = player_games['blocked'].sum() / max(player_games['shots'].sum(), 1)
    features.append(def_focus)
    
    # Agitator score (normalized)
    agitator = (player_games['pim'].mean() + player_games['hits'].mean() * 0.1)
    features.append(min(agitator / 5.0, 1.0))
    
    # Discipline (consistency in penalty taking)
    features.append(player_games['pim'].std())
    
    return np.array(features, dtype=np.float32)



# ============================================================================
# 2. EXAMPLE USAGE
# ============================================================================

def test_preprocessor():
    """Test the preprocessor with real database"""
    
    preprocessor = DataPreprocessor()
    
    # Build vocabulary
    preprocessor.fit_player_vocab(min_games=5)
    
    # Get a sample game
    preprocessor.connect()
    query = """
    SELECT * FROM games 
    WHERE game_state IN ('OFF', 'FINAL') 
    AND game_type = 2
    AND home_score IS NOT NULL
    LIMIT 1
    """
    
    sample_game = pd.read_sql_query(query, preprocessor.conn).iloc[0]
    
    print("\nProcessing sample game:")
    print(f"  {sample_game['away_team']} @ {sample_game['home_team']}")
    print(f"  Date: {sample_game['game_date']}")
    print(f"  Score: {sample_game['away_score']} - {sample_game['home_score']}")
    
    # Process it
    processed = preprocessor.process_game(sample_game)
    
    print("\nProcessed features:")
    print(f"  Home players: {processed['inputs']['home_player_ids'].shape}")
    print(f"  Home features: {processed['inputs']['home_features'].shape}")
    print(f"  Home temporal: {processed['inputs']['home_temporal_history'].shape}")
    print(f"  Context features: {processed['inputs']['context_features'].shape}")
    print(f"  Target: Home win = {processed['targets']['win']}")
    
    preprocessor.close()
    
    print("\n✓ Preprocessor test complete!")


if __name__ == "__main__":
    test_preprocessor()