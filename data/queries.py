"""
NHL Sports Betting Model - Optimized Data Query Functions
Requires rolling_features table to be created first via create_rolling_features_table()
"""

import sqlite3
import pandas as pd
from typing import Optional
import time

db_path = "data/nhl_data.db"

def create_rolling_features_table(lookback_games: int = 10):
    """
    Create pre-computed rolling features table with indexes.
    Run this once after data updates, then queries will be 100x faster.
    """
    start_time = time.time()
    print(f"[{time.ctime(start_time)}] START: create_rolling_features_table (lookback={lookback_games})")
    
    conn = sqlite3.connect(db_path)
    
    print(f"[{time.ctime(time.time())}] Computing rolling features using window functions...")
    
    # Create the rolling features table with window functions
    conn.execute(f"""
    CREATE TABLE  IF NOT EXISTS rolling_features AS
    SELECT 
        r.game_id,
        r.game_date,
        r.team,
        r.is_home,
        
        -- Rolling stats (last {lookback_games} games, excluding current)
        AVG(r.win) OVER (
            PARTITION BY r.team
            ORDER BY r.game_date
            ROWS BETWEEN {lookback_games} PRECEDING AND 1 PRECEDING
        ) AS win_pct_l{lookback_games},
        
        AVG(r.goals_for) OVER (
            PARTITION BY r.team
            ORDER BY r.game_date
            ROWS BETWEEN {lookback_games} PRECEDING AND 1 PRECEDING
        ) AS goals_l{lookback_games},

        AVG(r.goals_against) OVER (
            PARTITION BY r.team
            ORDER BY r.game_date
            ROWS BETWEEN {lookback_games} PRECEDING AND 1 PRECEDING
        ) AS goals_against_l{lookback_games},
        
        AVG(r.goal_differential) OVER (
            PARTITION BY r.team
            ORDER BY r.game_date
            ROWS BETWEEN {lookback_games} PRECEDING AND 1 PRECEDING
        ) AS goal_diff_l{lookback_games},
        
        AVG(r.shooting_pct) OVER (
            PARTITION BY r.team
            ORDER BY r.game_date
            ROWS BETWEEN {lookback_games} PRECEDING AND 1 PRECEDING
        ) AS shooting_pct_l{lookback_games},
        
        AVG(r.save_pct) OVER (
            PARTITION BY r.team
            ORDER BY r.game_date
            ROWS BETWEEN {lookback_games} PRECEDING AND 1 PRECEDING
        ) AS save_pct_l{lookback_games},
        
        -- Pace calculation
        AVG(r.shots_for + r.shots_against) OVER (
            PARTITION BY r.team
            ORDER BY r.game_date
            ROWS BETWEEN {lookback_games} PRECEDING AND 1 PRECEDING
        ) AS pace_l{lookback_games},
        
        -- Home/away splits (use subqueries as SQLite lacks conditional window aggregates)
        (SELECT AVG(win) FROM team_rolling_stats r2 
         WHERE r2.team = r.team AND r2.is_home = 1 AND r2.game_date < r.game_date 
         ORDER BY r2.game_date DESC LIMIT {lookback_games}) as at_home_win_pct,
        
        (SELECT AVG(win) FROM team_rolling_stats r2 
         WHERE r2.team = r.team AND r2.is_home = 0 AND r2.game_date < r.game_date 
         ORDER BY r2.game_date DESC LIMIT {lookback_games}) as on_road_win_pct
        
    FROM team_rolling_stats r
    WHERE EXISTS (
        SELECT 1 FROM team_rolling_stats r_prev 
        WHERE r_prev.team = r.team AND r_prev.game_date < r.game_date
        LIMIT {lookback_games} OFFSET {lookback_games - 1}
    )
    """)
    
    print(f"[{time.ctime(time.time())}] Creating indexes...")
    
    # Critical indexes for fast lookups
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rolling_team_date ON rolling_features(team, game_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rolling_game ON rolling_features(game_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rolling_team_home ON rolling_features(team, is_home, game_date)")
    
    # Get count
    count = conn.execute("SELECT COUNT(*) FROM rolling_features").fetchone()[0]
    
    conn.commit()
    conn.close()
    
    end_time = time.time()
    print(f"[{time.ctime(end_time)}] FINISHED: create_rolling_features_table")
    print(f"  Duration: {end_time - start_time:.2f} seconds")
    print(f"  Rows created: {count:,}")
    print(f"  ✓ Indexes created for optimal query performance")
    
    return count


def refresh_rolling_features(lookback_games: int = 10):
    """
    Refresh the rolling features table.
    Call this after updating game data.
    """
    print("\n" + "="*60)
    print("REFRESHING ROLLING FEATURES TABLE")
    print("="*60)
    create_rolling_features_table(lookback_games)
    print("="*60 + "\n")


def verify_rolling_features_exist():
    """Check if rolling_features table exists, raise error if not"""
    conn = sqlite3.connect(db_path)
    table_check = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='rolling_features'
    """).fetchone()
    conn.close()
    
    if not table_check:
        raise RuntimeError(
            "\n❌ ERROR: rolling_features table not found!\n"
            "Run create_rolling_features_table() first:\n"
            "  from data.queries import create_rolling_features_table\n"
            "  create_rolling_features_table(lookback_games=10)\n"
        )


# ============================================================================
# MAIN QUERY FUNCTIONS - All require rolling_features table
# ============================================================================
    
def get_training_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_games: int = 10
) -> pd.DataFrame:
    """
    Get complete training dataset with rolling features.
    Requires rolling_features table to exist.
    """
    verify_rolling_features_exist()
    
    start_time = time.time()
    print(f"[{time.ctime(start_time)}] START: get_training_data (lookback={lookback_games}, start_date={start_date}, end_date={end_date})")
    
    conn = sqlite3.connect(db_path)
    
    query = f"""
    SELECT 
        g.game_id,
        g.game_date,
        g.season,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score,
        CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_win,
        
        -- Home team rolling features
        h.win_pct_l{lookback_games} as home_win_pct_l{lookback_games},
        h.goals_l{lookback_games} as home_goals_l{lookback_games},
        h.goals_against_l{lookback_games} as home_goals_against_l{lookback_games},
        h.goal_diff_l{lookback_games} as home_goal_diff_l{lookback_games},
        h.shooting_pct_l{lookback_games} as home_shooting_pct_l{lookback_games},
        h.save_pct_l{lookback_games} as home_save_pct_l{lookback_games},
        h.at_home_win_pct as home_at_home_win_pct,
        
        -- Away team rolling features
        a.win_pct_l{lookback_games} as away_win_pct_l{lookback_games},
        a.goals_l{lookback_games} as away_goals_l{lookback_games},
        a.goals_against_l{lookback_games} as away_goals_against_l{lookback_games},
        a.goal_diff_l{lookback_games} as away_goal_diff_l{lookback_games},
        a.shooting_pct_l{lookback_games} as away_shooting_pct_l{lookback_games},
        a.save_pct_l{lookback_games} as away_save_pct_l{lookback_games},
        a.on_road_win_pct as away_on_road_win_pct,
        
        -- Rest days
        r_home.rest_days as home_rest_days,
        r_away.rest_days as away_rest_days,
        
        -- Head to head
        (SELECT team_a_wins * 1.0 / NULLIF(games_played, 0)
         FROM head_to_head_view 
         WHERE team_a = g.home_team AND team_b = g.away_team 
         AND as_of_date <= g.game_date
         ORDER BY as_of_date DESC LIMIT 1) as h2h_home_win_rate
        
    FROM games g
    
    -- Join pre-computed rolling features
    LEFT JOIN rolling_features h 
        ON g.home_team = h.team 
        AND g.game_date = h.game_date
        AND g.game_id = h.game_id
    
    LEFT JOIN rolling_features a 
        ON g.away_team = a.team 
        AND g.game_date = a.game_date
        AND g.game_id = a.game_id
    
    -- Rest days
    LEFT JOIN team_rest_days_view r_home 
        ON g.game_id = r_home.game_id 
        AND g.home_team = r_home.team
    
    LEFT JOIN team_rest_days_view r_away 
        ON g.game_id = r_away.game_id 
        AND g.away_team = r_away.team
    
    WHERE g.game_state IN ('OFF', 'FINAL') 
    AND g.game_type = 2
    AND h.win_pct_l{lookback_games} IS NOT NULL
    AND a.win_pct_l{lookback_games} IS NOT NULL
    """
    
    if start_date:
        query += f" AND g.game_date >= '{start_date}'"
    if end_date:
        query += f" AND g.game_date <= '{end_date}'"
    
    query += " ORDER BY g.game_date"
    
    print(f"[{time.ctime(time.time())}] EXECUTING OPTIMIZED QUERY...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    end_time = time.time()
    print(f"[{time.ctime(end_time)}] FINISHED: get_training_data (Duration: {end_time - start_time:.2f} seconds. Rows: {len(df)})")
    
    return df


def get_team_rolling_window(team: str, as_of_date: str, lookback_games: int = 10) -> pd.DataFrame:
    """Get rolling window stats for a specific team"""
    start_time = time.time()
    print(f"[{time.ctime(start_time)}] START: get_team_rolling_window (Team: {team}, Date: {as_of_date}, Lookback: {lookback_games})")
    
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
    WHERE team = ?
    AND game_date <= ?
    ORDER BY game_date DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(team, as_of_date, lookback_games))
    conn.close()
    
    end_time = time.time()
    print(f"[{time.ctime(end_time)}] FINISHED: get_team_rolling_window (Duration: {end_time - start_time:.2f} seconds. Rows: {len(df)})")
    
    return df


def get_player_rolling_window(player_id: int, as_of_date: str) -> pd.DataFrame:
    """
    Get player's cumulative stats as of a specific date.
    Returns single row with season-to-date totals and averages.
    """
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        player_id,
        game_date,
        season,
        games_played,
        total_goals,
        total_assists,
        total_points,
        total_shots,
        total_plus_minus,
        total_toi_minutes,
        total_powerplay_goals,
        total_hits,
        total_blocked,
        total_pim,
        avg_goals,
        avg_assists,
        avg_points,
        avg_shots,
        avg_plus_minus,
        avg_toi
    FROM player_cumulative_stats
    WHERE player_id = ?
    AND game_date <= ?
    ORDER BY game_date DESC
    LIMIT 1
    """
    
    df = pd.read_sql_query(query, conn, params=(player_id, as_of_date))
    conn.close()
    return df


def get_team_game_rolling_window(team: str, as_of_date: str, lookback_games: int = 10) -> pd.DataFrame:
    """Get rolling window from team_game_stats table"""
    start_time = time.time()
    print(f"[{time.ctime(start_time)}] START: get_team_game_rolling_window (Team: {team}, Date: {as_of_date}, Lookback: {lookback_games})")
    
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
    WHERE tgs.team_abbrev = ?
    AND g.game_date <= ?
    AND g.game_state IN ('OFF', 'FINAL')
    ORDER BY g.game_date DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(team, as_of_date, lookback_games))
    conn.close()
    
    end_time = time.time()
    print(f"[{time.ctime(end_time)}] FINISHED: get_team_game_rolling_window (Duration: {end_time - start_time:.2f} seconds. Rows: {len(df)})")
    
    return df


def get_games_rolling_window(start_date: str, end_date: str) -> pd.DataFrame:
    """Get all games in a date range"""
    start_time = time.time()
    print(f"[{time.ctime(start_time)}] START: get_games_rolling_window (Start Date: {start_date}, End Date: {end_date})")
    
    conn = sqlite3.connect(db_path)
    query = """
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
    WHERE game_date >= ?
    AND game_date <= ?
    ORDER BY game_date, start_time
    """
    
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    conn.close()
    
    end_time = time.time()
    print(f"[{time.ctime(end_time)}] FINISHED: get_games_rolling_window (Duration: {end_time - start_time:.2f} seconds. Rows: {len(df)})")
    
    return df


def get_all_teams_rolling_stats(as_of_date: str, lookback_games: int = 10) -> pd.DataFrame:
    """Get rolling stats for all teams as of a specific date - uses rolling_features"""
    verify_rolling_features_exist()
    
    start_time = time.time()
    print(f"[{time.ctime(start_time)}] START: get_all_teams_rolling_stats (Date: {as_of_date}, Lookback: {lookback_games})")
    
    conn = sqlite3.connect(db_path)
    
    query = f"""
    SELECT 
        team,
        AVG(win_pct_l{lookback_games}) as win_pct,
        AVG(goals_l{lookback_games}) as avg_goals_for,
        AVG(goals_against_l{lookback_games}) as avg_goals_against,
        AVG(goal_diff_l{lookback_games}) as avg_goal_diff,
        AVG(shooting_pct_l{lookback_games}) as avg_shooting_pct,
        AVG(save_pct_l{lookback_games}) as avg_save_pct,
        COUNT(*) as games_played
    FROM rolling_features
    WHERE game_date <= ?
    GROUP BY team
    HAVING games_played >= ?
    ORDER BY win_pct DESC
    """
    
    df = pd.read_sql_query(query, conn, params=(as_of_date, lookback_games))
    conn.close()
    
    end_time = time.time()
    print(f"[{time.ctime(end_time)}] FINISHED: get_all_teams_rolling_stats (Duration: {end_time - start_time:.2f} seconds. Rows: {len(df)})")
    
    return df


def get_h2h_rolling_window(team_a: str, team_b: str, as_of_date: str, lookback_games: int = 10) -> pd.DataFrame:
    """Get head-to-head games between two teams"""
    start_time = time.time()
    print(f"[{time.ctime(start_time)}] START: get_h2h_rolling_window (Teams: {team_a} vs {team_b}, Date: {as_of_date}, Lookback: {lookback_games})")
    
    conn = sqlite3.connect(db_path)
    query = """
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
    WHERE ((team = ? AND opponent = ?)
        OR (team = ? AND opponent = ?))
    AND game_date <= ?
    ORDER BY game_date DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(team_a, team_b, team_b, team_a, as_of_date, lookback_games))
    conn.close()
    
    end_time = time.time()
    print(f"[{time.ctime(end_time)}] FINISHED: get_h2h_rolling_window (Duration: {end_time - start_time:.2f} seconds. Rows: {len(df)})")
    
    return df


def get_rest_days_rolling_window(team: str, as_of_date: str, lookback_games: int = 10) -> pd.DataFrame:
    """Get rest days for team's recent games"""
    start_time = time.time()
    print(f"[{time.ctime(start_time)}] START: get_rest_days_rolling_window (Team: {team}, Date: {as_of_date}, Lookback: {lookback_games})")
    
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        trd.team,
        g.game_date,
        trd.game_id,
        trd.rest_days
    FROM team_rest_days_view trd
    JOIN games g ON trd.game_id = g.game_id
    WHERE trd.team = ?
    AND g.game_date <= ?
    ORDER BY g.game_date DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(team, as_of_date, lookback_games))
    conn.close()
    
    end_time = time.time()
    print(f"[{time.ctime(end_time)}] FINISHED: get_rest_days_rolling_window (Duration: {end_time - start_time:.2f} seconds. Rows: {len(df)})")
    
    return df


def get_upcoming_games() -> pd.DataFrame:
    """Get upcoming games with basic info"""
    start_time = time.time()
    print(f"[{time.ctime(start_time)}] START: get_upcoming_games")
    
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
    
    end_time = time.time()
    print(f"[{time.ctime(end_time)}] FINISHED: get_upcoming_games (Duration: {end_time - start_time:.2f} seconds. Rows: {len(df)})")
    
    return df


def get_over_under_data(start_date: Optional[str] = None, lookback_games: int = 10) -> pd.DataFrame:
    """Get data for over/under modeling - requires rolling_features"""
    verify_rolling_features_exist()
    
    start_time = time.time()
    print(f"[{time.ctime(start_time)}] START: get_over_under_data (Lookback: {lookback_games}, Start Date: {start_date})")
    
    conn = sqlite3.connect(db_path)
    
    query = f"""
    SELECT 
        g.game_date,
        g.game_id,
        g.home_team,
        g.away_team,
        g.home_score + g.away_score as total_goals,
        
        h.goals_l{lookback_games} as home_off_l{lookback_games},
        a.goals_l{lookback_games} as away_off_l{lookback_games},
        h.goals_against_l{lookback_games} as home_def_l{lookback_games},
        a.goals_against_l{lookback_games} as away_def_l{lookback_games},
        h.pace_l{lookback_games} as home_pace_l{lookback_games},
        a.pace_l{lookback_games} as away_pace_l{lookback_games}
        
    FROM games g
    LEFT JOIN rolling_features h 
        ON g.home_team = h.team AND g.game_date = h.game_date AND g.game_id = h.game_id
    LEFT JOIN rolling_features a 
        ON g.away_team = a.team AND g.game_date = a.game_date AND g.game_id = a.game_id
    WHERE g.game_state IN ('OFF', 'FINAL') 
    AND g.game_type = 2
    AND h.goals_l{lookback_games} IS NOT NULL
    AND a.goals_l{lookback_games} IS NOT NULL
    """
    
    if start_date:
        query += f" AND g.game_date >= '{start_date}'"
    
    query += " ORDER BY g.game_date DESC"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    end_time = time.time()
    print(f"[{time.ctime(end_time)}] FINISHED: get_over_under_data (Duration: {end_time - start_time:.2f} seconds. Rows: {len(df)})")
    
    return df