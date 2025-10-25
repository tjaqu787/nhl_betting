"""
NHL Sports Betting Model - Data Query Functions
Simple query wrappers for feature engineering
"""

import sqlite3
import pandas as pd
from typing import Optional


db_path = "nhl_data.db"
    
def get_training_data( 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        lookback_games: int = 10) -> pd.DataFrame:
    """Get complete training dataset with rolling features"""
    conn = sqlite3.connect(db_path)
    query = f"""
    WITH game_features AS (
        SELECT 
            g.game_id,
            g.game_date,
            g.season,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_win,
            
            (SELECT AVG(win) FROM team_rolling_stats 
                WHERE team = g.home_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as home_win_pct_l{lookback_games},
                
            (SELECT AVG(goals_for) FROM team_rolling_stats 
                WHERE team = g.home_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as home_goals_l{lookback_games},
                
            (SELECT AVG(goals_against) FROM team_rolling_stats 
                WHERE team = g.home_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as home_goals_against_l{lookback_games},
                
            (SELECT AVG(goal_differential) FROM team_rolling_stats 
                WHERE team = g.home_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as home_goal_diff_l{lookback_games},
                
            (SELECT AVG(shooting_pct) FROM team_rolling_stats 
                WHERE team = g.home_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as home_shooting_pct_l{lookback_games},
                
            (SELECT AVG(save_pct) FROM team_rolling_stats 
                WHERE team = g.home_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as home_save_pct_l{lookback_games},
            
            (SELECT AVG(win) FROM team_rolling_stats 
                WHERE team = g.away_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as away_win_pct_l{lookback_games},
                
            (SELECT AVG(goals_for) FROM team_rolling_stats 
                WHERE team = g.away_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as away_goals_l{lookback_games},
                
            (SELECT AVG(goals_against) FROM team_rolling_stats 
                WHERE team = g.away_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as away_goals_against_l{lookback_games},
                
            (SELECT AVG(goal_differential) FROM team_rolling_stats 
                WHERE team = g.away_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as away_goal_diff_l{lookback_games},
                
            (SELECT AVG(shooting_pct) FROM team_rolling_stats 
                WHERE team = g.away_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as away_shooting_pct_l{lookback_games},
                
            (SELECT AVG(save_pct) FROM team_rolling_stats 
                WHERE team = g.away_team AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as away_save_pct_l{lookback_games},
            
            (SELECT rest_days FROM team_rest_days 
                WHERE team = g.home_team AND game_id = g.game_id) as home_rest_days,
                
            (SELECT rest_days FROM team_rest_days 
                WHERE team = g.away_team AND game_id = g.game_id) as away_rest_days,
            
            (SELECT team_a_wins * 1.0 / NULLIF(games_played, 0)
                FROM head_to_head 
                WHERE team_a = g.home_team AND team_b = g.away_team 
                AND as_of_date <= g.game_date
                ORDER BY as_of_date DESC LIMIT 1) as h2h_home_win_rate,
            
            (SELECT AVG(win) FROM team_rolling_stats 
                WHERE team = g.home_team AND is_home = 1 AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as home_at_home_win_pct,
                
            (SELECT AVG(win) FROM team_rolling_stats 
                WHERE team = g.away_team AND is_home = 0 AND game_date < g.game_date 
                ORDER BY game_date DESC LIMIT {lookback_games}) as away_on_road_win_pct
            
        FROM games g
        WHERE g.game_state IN ('OFF', 'FINAL') AND g.game_type = 2
    )
    SELECT * FROM game_features
    WHERE home_win_pct_l{lookback_games} IS NOT NULL
    """
    
    if start_date:
        query += f" AND game_date >= '{start_date}'"
    if end_date:
        query += f" AND game_date <= '{end_date}'"
        
    query += " ORDER BY game_date"

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_team_rolling_window( team: str, as_of_date: str, lookback_games: int = 10) -> pd.DataFrame:
    """Get rolling window stats for a specific team"""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT 
        team,
        game_date,
        game_id,
        opponent,
        is_home,
        win,
        goals_for,
        goals_against,
        goal_differential,
        shots_for,
        shots_against,
        shooting_pct,
        save_pct
    FROM team_rolling_stats
    WHERE team = '{team}'
    AND game_date <= '{as_of_date}'
    ORDER BY game_date DESC
    LIMIT {lookback_games}
"""
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_player_rolling_window( player_id: int, as_of_date: str, lookback_games: int = 10) -> pd.DataFrame:
    """Get rolling window stats for a specific player"""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT 
        pgs.player_id,
        g.game_date,
        pgs.game_id,
        pgs.team_abbrev,
        pgs.position,
        pgs.goals,
        pgs.assists,
        pgs.points,
        pgs.shots,
        pgs.plus_minus,
        pgs.toi,
        pgs.powerplay_goals,
        pgs.hits,
        pgs.blocked
    FROM player_game_stats pgs
    JOIN games g ON pgs.game_id = g.game_id
    WHERE pgs.player_id = {player_id}
    AND g.game_date <= '{as_of_date}'
    AND g.game_state IN ('OFF', 'FINAL')
    ORDER BY g.game_date DESC
    LIMIT {lookback_games}
"""
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_team_game_rolling_window( team: str, as_of_date: str, lookback_games: int = 10) -> pd.DataFrame:
    """Get rolling window from team_game_stats table"""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT 
        tgs.game_id,
        g.game_date,
        tgs.team_abbrev,
        tgs.is_home,
        tgs.goals,
        tgs.shots,
        tgs.pim,
        tgs.powerplay_goals,
        tgs.powerplay_opportunities,
        tgs.faceoff_win_pct,
        tgs.blocked,
        tgs.hits,
        tgs.giveaways,
        tgs.takeaways
    FROM team_game_stats tgs
    JOIN games g ON tgs.game_id = g.game_id
    WHERE tgs.team_abbrev = '{team}'
    AND g.game_date <= '{as_of_date}'
    AND g.game_state IN ('OFF', 'FINAL')
    ORDER BY g.game_date DESC
    LIMIT {lookback_games}
"""
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_games_rolling_window( start_date: str, end_date: str) -> pd.DataFrame:
    """Get all games in a date range"""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT 
        game_id,
        season,
        game_type,
        game_date,
        start_time,
        home_team,
        away_team,
        home_score,
        away_score,
        game_state,
        period,
        venue,
        home_sog,
        away_sog
    FROM games
    WHERE game_date >= '{start_date}'
    AND game_date <= '{end_date}'
    ORDER BY game_date, start_time
"""
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_all_teams_rolling_stats( as_of_date: str, lookback_games: int = 10) -> pd.DataFrame:
    """Get rolling stats for all teams as of a specific date"""
    conn = sqlite3.connect(db_path)
    query = f"""
    WITH team_recent AS (
        SELECT 
            t1.team,
            AVG(t2.win) as win_pct,
            AVG(t2.goals_for) as avg_goals_for,
            AVG(t2.goals_against) as avg_goals_against,
            AVG(t2.goal_differential) as avg_goal_diff,
            AVG(t2.shooting_pct) as avg_shooting_pct,
            AVG(t2.save_pct) as avg_save_pct,
            COUNT(*) as games_played
        FROM (SELECT DISTINCT team FROM team_rolling_stats) t1
        CROSS JOIN team_rolling_stats t2
        WHERE t1.team = t2.team
        AND t2.game_date <= '{as_of_date}'
        AND t2.game_date > date('{as_of_date}', '-{lookback_games * 2} days')
        GROUP BY t1.team
        HAVING COUNT(*) >= {lookback_games}
    )
    SELECT * FROM team_recent
    ORDER BY win_pct DESC
"""
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_h2h_rolling_window( team_a: str, team_b: str, as_of_date: str, lookback_games: int = 10) -> pd.DataFrame:
    """Get head-to-head games between two teams"""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT 
        game_date,
        team,
        opponent,
        is_home,
        win,
        goals_for,
        goals_against,
        goal_differential
    FROM team_rolling_stats
    WHERE ((team = '{team_a}' AND opponent = '{team_b}')
        OR (team = '{team_b}' AND opponent = '{team_a}'))
    AND game_date <= '{as_of_date}'
    ORDER BY game_date DESC
    LIMIT {lookback_games}
"""
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_rest_days_rolling_window( team: str, as_of_date: str, lookback_games: int = 10) -> pd.DataFrame:
    """Get rest days for team's recent games"""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT 
        trd.team,
        g.game_date,
        trd.game_id,
        trd.rest_days
    FROM team_rest_days trd
    JOIN games g ON trd.game_id = g.game_id
    WHERE trd.team = '{team}'
    AND g.game_date <= '{as_of_date}'
    ORDER BY g.game_date DESC
    LIMIT {lookback_games}
"""
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_upcoming_games() -> pd.DataFrame:
    """Get upcoming games with basic info"""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        game_id,
        game_date,
        start_time,
        home_team,
        away_team,
        venue,
        game_state
    FROM games
    WHERE game_date >= date('now')
    AND game_state NOT IN ('OFF', 'FINAL')
    ORDER BY game_date, start_time
"""

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_over_under_data( start_date: Optional[str] = None, lookback_games: int = 10) -> pd.DataFrame:
    """Get data for over/under modeling"""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT 
        g.game_date,
        g.game_id,
        g.home_team,
        g.away_team,
        g.home_score + g.away_score as total_goals,
        
        (SELECT AVG(goals_for) FROM team_rolling_stats 
            WHERE team = g.home_team AND game_date < g.game_date 
            ORDER BY game_date DESC LIMIT {lookback_games}) as home_off_l{lookback_games},
            
        (SELECT AVG(goals_for) FROM team_rolling_stats 
            WHERE team = g.away_team AND game_date < g.game_date 
            ORDER BY game_date DESC LIMIT {lookback_games}) as away_off_l{lookback_games},
        
        (SELECT AVG(goals_against) FROM team_rolling_stats 
            WHERE team = g.home_team AND game_date < g.game_date 
            ORDER BY game_date DESC LIMIT {lookback_games}) as home_def_l{lookback_games},
            
        (SELECT AVG(goals_against) FROM team_rolling_stats 
            WHERE team = g.away_team AND game_date < g.game_date 
            ORDER BY game_date DESC LIMIT {lookback_games}) as away_def_l{lookback_games},
        
        (SELECT AVG(shots_for + shots_against) FROM team_rolling_stats 
            WHERE team = g.home_team AND game_date < g.game_date 
            ORDER BY game_date DESC LIMIT {lookback_games}) as home_pace_l{lookback_games},
            
        (SELECT AVG(shots_for + shots_against) FROM team_rolling_stats 
            WHERE team = g.away_team AND game_date < g.game_date 
            ORDER BY game_date DESC LIMIT {lookback_games}) as away_pace_l{lookback_games}
        
    FROM games g
    WHERE g.game_state IN ('OFF', 'FINAL') AND g.game_type = 2
    """
    
    if start_date:
        query += f" AND g.game_date >= '{start_date}'"
    
    query += " ORDER BY g.game_date DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


