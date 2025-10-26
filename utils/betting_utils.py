import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from collections import defaultdict


# ============================================================================
# 1. PRODUCTION DEPLOYMENT UTILITIES
# ============================================================================

class LiveBettingDeployer:
    """
    Real-time prediction for live betting.
    """
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.game_state_buffer = {}  # Store evolving game states
    
    def predict_pregame(self, game_info: Dict) -> Dict:
        """
        Make pre-game prediction.
        """
        # Process inputs
        processed = self.preprocessor.process_game(
            pd.Series(game_info),
            player_stats_db=None,  # Load from database
            coplay_db={}
        )
        
        # Add batch dimension
        inputs = {
            k: tf.expand_dims(v, 0) 
            for k, v in processed['inputs'].items()
        }
        
        # Predict
        outputs = self.model(inputs, training=False)
        
        win_prob = outputs['win_prob'].numpy()[0, 0]
        spread_dist = {
            k: v.numpy()[0] 
            for k, v in outputs['spread_dist'].items()
        }
        
        return {
            'win_probability': float(win_prob),
            'expected_spread': float(np.sum(
                spread_dist['mu'] * spread_dist['pi']
            )),
            'spread_uncertainty': float(np.sum(
                spread_dist['sigma'] * spread_dist['pi']
            ))
        }
    
    def update_live(self, game_id: str, game_state: Dict) -> Dict:
        """
        Update prediction during live game.
        game_state: {quarter, time_remaining, score_home, score_away, ...}
        """
        # Buffer game states for sequence
        if game_id not in self.game_state_buffer:
            self.game_state_buffer[game_id] = []
        
        self.game_state_buffer[game_id].append(game_state)
        
        # Process sequence
        state_sequence = self._format_game_states(
            self.game_state_buffer[game_id]
        )
        
        # Add to inputs
        inputs = game_state['base_inputs'].copy()
        inputs['game_states'] = tf.expand_dims(state_sequence, 0)
        
        # Predict with live context
        outputs = self.model(inputs, training=False)
        
        win_prob_live = outputs.get('win_prob_live', outputs['win_prob'])
        
        return {
            'win_probability_live': float(win_prob_live.numpy()[0, 0]),
            'game_state_confidence': self._compute_confidence(state_sequence)
        }
    
    def _format_game_states(self, states: List[Dict]) -> np.ndarray:
        """Convert game state dicts to tensor."""
        formatted = []
        for state in states:
            vec = [
                state.get('quarter', 1),
                state.get('time_remaining', 720),
                state.get('score_diff', 0),
                state.get('possessions', 0),
                # ... more features
            ]
            formatted.append(vec)
        
        # Pad to fixed length
        while len(formatted) < 50:
            formatted.append([0] * len(formatted[0]))
        
        return np.array(formatted[:50], dtype=np.float32)
    
    def _compute_confidence(self, state_sequence: np.ndarray) -> float:
        """Confidence based on sample size and stability."""
        return min(len(state_sequence) / 20.0, 1.0)
