"""
NHL Player Rolling Statistics - Efficient SQL + Plotly Visualization
Uses SQL window functions for fast rolling averages
"""

import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_player_rolling_stats(player_id: int, db_path: str = "data/nhl_data.db") -> pd.DataFrame:
    """
    Efficient query using SQL window functions for rolling averages.
    MUCH faster than Python loops!
    
    Args:
        player_id: NHL player ID
        db_path: Path to SQLite database
    
    Returns:
        DataFrame with rolling averages for 5, 10, 20, 50 games
    """
    conn = sqlite3.connect(db_path)
    
    query = f"""
    WITH player_games AS (
      SELECT 
        pgs.player_id,
        pgs.game_id,
        g.game_date,
        pgs.goals,
        pgs.assists,
        pgs.points,
        pgs.shots,
        pgs.plus_minus,
        pgs.hits,
        pgs.blocked,
        pgs.powerplay_goals,
        pgs.pim,
        ROW_NUMBER() OVER (PARTITION BY pgs.player_id ORDER BY g.game_date) as game_num
      FROM player_game_stats pgs
      JOIN games g ON pgs.game_id = g.game_id
      WHERE pgs.player_id = {player_id}
        AND g.game_state IN ('OFF', 'FINAL')
      ORDER BY g.game_date
    )
    SELECT 
      player_id,
      game_id,
      game_date,
      game_num,
      goals,
      assists,
      points,
      shots,
      plus_minus,
      hits,
      blocked,
      powerplay_goals,
      pim,
      
      -- 5-game rolling averages (window functions are FAST!)
      AVG(goals) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
      ) as goals_5g,
      AVG(assists) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
      ) as assists_5g,
      AVG(points) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
      ) as points_5g,
      AVG(shots) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
      ) as shots_5g,
      AVG(CAST(plus_minus AS REAL)) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
      ) as plus_minus_5g,
      AVG(hits) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
      ) as hits_5g,
      
      -- 10-game rolling averages
      AVG(goals) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
      ) as goals_10g,
      AVG(assists) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
      ) as assists_10g,
      AVG(points) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
      ) as points_10g,
      AVG(shots) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
      ) as shots_10g,
      AVG(CAST(plus_minus AS REAL)) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
      ) as plus_minus_10g,
      AVG(hits) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
      ) as hits_10g,
      
      -- 20-game rolling averages
      AVG(goals) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
      ) as goals_20g,
      AVG(assists) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
      ) as assists_20g,
      AVG(points) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
      ) as points_20g,
      AVG(shots) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
      ) as shots_20g,
      AVG(CAST(plus_minus AS REAL)) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
      ) as plus_minus_20g,
      AVG(hits) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
      ) as hits_20g,
      
      -- 50-game rolling averages
      AVG(goals) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
      ) as goals_50g,
      AVG(assists) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
      ) as assists_50g,
      AVG(points) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
      ) as points_50g,
      AVG(shots) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
      ) as shots_50g,
      AVG(CAST(plus_minus AS REAL)) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
      ) as plus_minus_50g,
      AVG(hits) OVER (
        PARTITION BY player_id 
        ORDER BY game_date 
        ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
      ) as hits_50g

    FROM player_games
    ORDER BY game_date;
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def plot_player_rolling_stats(df: pd.DataFrame, metrics: list = None):
    """
    Create interactive Plotly dashboard with rolling averages.
    
    Args:
        df: DataFrame from get_player_rolling_stats()
        metrics: List of metrics to plot. Default: ['points', 'goals', 'assists', 'shots']
    """
    if metrics is None:
        metrics = ['points', 'goals', 'assists', 'shots', 'plus_minus', 'hits']
    
    # Create subplots
    fig = make_subplots(
        rows=len(metrics), 
        cols=1,
        subplot_titles=[m.replace('_', ' ').title() for m in metrics],
        vertical_spacing=0.05,
        shared_xaxes=True
    )
    
    colors = {
        'actual': 'rgba(100, 116, 139, 0.3)',
        '5g': '#ef4444',
        '10g': '#f59e0b', 
        '20g': '#10b981',
        '50g': '#3b82f6'
    }
    
    for idx, metric in enumerate(metrics, 1):
        # Actual values (faint)
        fig.add_trace(
            go.Scatter(
                x=df['game_num'],
                y=df[metric],
                mode='lines',
                name=f'{metric} (actual)' if idx == 1 else None,
                line=dict(color=colors['actual'], width=1),
                showlegend=(idx == 1),
                legendgroup='actual'
            ),
            row=idx, col=1
        )
        
        # 5-game average
        fig.add_trace(
            go.Scatter(
                x=df['game_num'],
                y=df[f'{metric}_5g'],
                mode='lines',
                name='5-game avg' if idx == 1 else None,
                line=dict(color=colors['5g'], width=2),
                showlegend=(idx == 1),
                legendgroup='5g'
            ),
            row=idx, col=1
        )
        
        # 10-game average
        fig.add_trace(
            go.Scatter(
                x=df['game_num'],
                y=df[f'{metric}_10g'],
                mode='lines',
                name='10-game avg' if idx == 1 else None,
                line=dict(color=colors['10g'], width=2),
                showlegend=(idx == 1),
                legendgroup='10g'
            ),
            row=idx, col=1
        )
        
        # 20-game average
        fig.add_trace(
            go.Scatter(
                x=df['game_num'],
                y=df[f'{metric}_20g'],
                mode='lines',
                name='20-game avg' if idx == 1 else None,
                line=dict(color=colors['20g'], width=2),
                showlegend=(idx == 1),
                legendgroup='20g'
            ),
            row=idx, col=1
        )
        
        # 50-game average
        fig.add_trace(
            go.Scatter(
                x=df['game_num'],
                y=df[f'{metric}_50g'],
                mode='lines',
                name='50-game avg' if idx == 1 else None,
                line=dict(color=colors['50g'], width=3),
                showlegend=(idx == 1),
                legendgroup='50g'
            ),
            row=idx, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=300 * len(metrics),
        title_text=f"Player {df['player_id'].iloc[0]} - Rolling Statistics ({len(df)} games)",
        title_font_size=20,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        template='plotly_dark'
    )
    
    # Update x-axis
    fig.update_xaxes(title_text="Game Number", row=len(metrics), col=1)
    
    # Update y-axes
    for idx, metric in enumerate(metrics, 1):
        fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=idx, col=1)
    
    return fig


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics for latest game"""
    latest = df.iloc[-1]
    
    print("\n" + "="*60)
    print(f"PLAYER {latest['player_id']} - LATEST STATS")
    print("="*60)
    print(f"Game: {latest['game_num']} | Date: {latest['game_date']}")
    print("-"*60)
    
    metrics = ['points', 'goals', 'assists', 'shots', 'plus_minus', 'hits']
    
    
    print("="*60 + "\n")


def main():
    """Example usage"""
    import time
    
    # Example player ID (replace with actual player from your database)
    player_id = 8477444
    
    print(f"Fetching rolling stats for player {player_id}...")
    start = time.time()
    
    # Get data (this is FAST because SQL does the heavy lifting)
    df = get_player_rolling_stats(player_id)
    
    elapsed = time.time() - start
    print(f"✓ Loaded {len(df)} games in {elapsed:.3f} seconds")
    
    # Print summary
    print_summary_stats(df)
    
    # Create interactive plot
    print("Creating interactive plot...")
    fig = plot_player_rolling_stats(df)
    
    # Save to HTML
    output_file = f"player_{player_id}_rolling_stats.html"
    fig.write_html(output_file)
    print(f"✓ Saved plot to {output_file}")
    
    # Show plot (opens in browser)
    fig.show()


if __name__ == "__main__":
    main()