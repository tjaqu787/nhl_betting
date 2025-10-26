
# ============================================================================
# 12. BACKTESTING FRAMEWORK
# ============================================================================

class BettingBacktester:
    """
    Rigorous backtesting with realistic conditions.
    """
    def __init__(self, 
                 initial_bankroll=10000.0,
                 max_kelly_fraction=0.02,
                 vig=0.05,
                 min_edge_threshold=0.02):
        self.initial_bankroll = initial_bankroll
        self.max_kelly_fraction = max_kelly_fraction
        self.vig = vig
        self.min_edge_threshold = min_edge_threshold
        
        self.reset()
    
    def reset(self):
        self.bankroll_history = [self.initial_bankroll]
        self.bet_history = []
        self.current_bankroll = self.initial_bankroll
    
    def compute_edge(self, model_prob, market_odds):
        """
        Compute expected edge vs bookmaker.
        """
        implied_prob = 1.0 / market_odds
        # Adjust for vig (bookmaker takes commission)
        fair_implied_prob = implied_prob / (1 + self.vig)
        
        edge = model_prob - fair_implied_prob
        return edge
    
    def compute_kelly_bet(self, model_prob, market_odds):
        """
        Kelly criterion bet sizing.
        """
        edge = self.compute_edge(model_prob, market_odds)
        
        if edge < self.min_edge_threshold:
            return 0.0  # Don't bet
        
        # Kelly formula: f = (p * odds - 1) / (odds - 1)
        kelly_frac = (model_prob * market_odds - 1.0) / (market_odds - 1.0)
        
        # Cap at max fraction for safety
        kelly_frac = np.clip(kelly_frac, 0.0, self.max_kelly_fraction)
        
        bet_size = kelly_frac * self.current_bankroll
        return bet_size
    
    def place_bet(self, game_id, model_prob, market_odds, actual_outcome):
        """
        Simulate placing a bet and update bankroll.
        """
        bet_size = self.compute_kelly_bet(model_prob, market_odds)
        
        if bet_size == 0:
            return
        
        # Bet on home team if model_prob > implied prob
        if actual_outcome == 1:  # Home wins
            profit = bet_size * (market_odds - 1)
        else:  # Home loses
            profit = -bet_size
        
        self.current_bankroll += profit
        self.bankroll_history.append(self.current_bankroll)
        
        self.bet_history.append({
            'game_id': game_id,
            'bet_size': bet_size,
            'model_prob': model_prob,
            'market_odds': market_odds,
            'outcome': actual_outcome,
            'profit': profit,
            'bankroll': self.current_bankroll
        })
    
    def compute_metrics(self):
        """
        Compute performance metrics.
        """
        if len(self.bet_history) == 0:
            return {}
        
        df = pd.DataFrame(self.bet_history)
        
        total_profit = self.current_bankroll - self.initial_bankroll
        total_return = total_profit / self.initial_bankroll
        
        # Sharpe ratio (assuming daily bets)
        returns = np.diff(self.bankroll_history) / self.bankroll_history[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Max drawdown
        peak = np.maximum.accumulate(self.bankroll_history)
        drawdown = (peak - self.bankroll_history) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        win_rate = (df['profit'] > 0).mean()
        
        # Average edge on bets placed
        avg_edge = df.apply(
            lambda row: self.compute_edge(row['model_prob'], row['market_odds']),
            axis=1
        ).mean()
        
        return {
            'total_profit': total_profit,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_bets': len(df),
            'win_rate': win_rate,
            'avg_edge': avg_edge,
            'final_bankroll': self.current_bankroll
        }
