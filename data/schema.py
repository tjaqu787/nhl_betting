
rosters = """
        CREATE TABLE IF NOT EXISTS rosters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_abbrev TEXT,
            player_id INTEGER,
            season TEXT,
            position_code TEXT,
            last_updated TIMESTAMP,
            UNIQUE(team_abbrev, player_id, season)
        )
    """
players = """
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            full_name TEXT,
            first_name TEXT,
            last_name TEXT,
            position TEXT,
            shoots_catches TEXT,
            height_inches INTEGER,
            weight_lbs INTEGER,
            birth_date TEXT,
            birth_city TEXT,
            birth_country TEXT,
            current_team TEXT,
            sweater_number INTEGER,
            headshot_url TEXT,
            data_json TEXT,
            updated_at TIMESTAMP
        )
    """

team_game_stats = """
        CREATE TABLE IF NOT EXISTS team_game_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            team_abbrev TEXT,
            is_home INTEGER,
            goals INTEGER,
            shots INTEGER,
            pim INTEGER,
            powerplay_goals INTEGER,
            powerplay_opportunities INTEGER,
            faceoff_win_pct REAL,
            blocked INTEGER,
            hits INTEGER,
            giveaways INTEGER,
            takeaways INTEGER,
            data_json TEXT,
            updated_at TIMESTAMP,
            UNIQUE(game_id, team_abbrev)
        )
    """
player_game_stats = """
        CREATE TABLE IF NOT EXISTS player_game_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER,
            game_id INTEGER,
            team_abbrev TEXT,
            position TEXT,
            goals INTEGER,
            assists INTEGER,
            points INTEGER,
            plus_minus INTEGER,
            pim INTEGER,
            shots INTEGER,
            toi TEXT,
            powerplay_goals INTEGER,
            powerplay_assists INTEGER,
            shorthanded_goals INTEGER,
            shorthanded_assists INTEGER,
            ot_goals INTEGER,
            faceoff_pct REAL,
            data_json TEXT,
            updated_at TIMESTAMP,
            UNIQUE(player_id, game_id)
        )
    """
games = """
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    season TEXT,
    game_type INTEGER,
    game_date TEXT,
    start_time TEXT,
    home_team TEXT,
    away_team TEXT,
    home_score INTEGER,
    away_score INTEGER,
    game_state TEXT,
    period INTEGER,
    venue TEXT,
    home_sog INTEGER,
    away_sog INTEGER,
    data_json TEXT,
    updated_at TIMESTAMP
)
"""

teams = """
CREATE TABLE IF NOT EXISTS teams (
team_abbrev TEXT PRIMARY KEY,
team_name TEXT,
franchise_id INTEGER,
conference TEXT,
division TEXT,
data_json TEXT,
updated_at TIMESTAMP
)
    """

player_cumulative_stats = """
CREATE TABLE IF NOT EXISTS player_cumulative_stats (
player_id INTEGER,
game_date TEXT,
season TEXT,
games_played INTEGER,
total_goals INTEGER,
total_assists INTEGER,
total_points INTEGER,
total_shots INTEGER,
total_plus_minus INTEGER,
total_toi_minutes REAL,
total_powerplay_goals INTEGER,
total_powerplay_assists INTEGER,
total_shorthanded_goals INTEGER,
total_shorthanded_assists INTEGER,
total_game_winning_goals INTEGER,
total_ot_goals INTEGER,
total_hits INTEGER,
total_blocked INTEGER,
total_pim INTEGER,
-- Pre-computed averages
avg_goals REAL,
avg_assists REAL,
avg_points REAL,
avg_shots REAL,
avg_plus_minus REAL,
avg_toi REAL,
avg_pim REAL,
-- Efficiency metrics
shooting_pct REAL,
points_per_60 REAL,
PRIMARY KEY (player_id, game_date)
)
"""
plays = """
CREATE TABLE IF NOT EXISTS plays (
play_id INTEGER PRIMARY KEY,
game_id INTEGER NOT NULL,
event_id INTEGER UNIQUE NOT NULL,
period_number INTEGER,
period_type TEXT,
time_in_period TEXT,
time_remaining TEXT,
type_code INTEGER,
type_desc_key TEXT,
sort_order INTEGER,
details TEXT, -- JSON blob for all the details
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
FOREIGN KEY (game_id) REFERENCES games(game_id)
);
"""

team_rolling_stats_view = """
CREATE VIEW IF NOT EXISTS team_rolling_stats_view AS
WITH game_results AS (
    SELECT 
        game_id,
        game_date,
        season,
        home_team as team,
        away_team as opponent,
        1 as is_home,
        CASE 
            WHEN home_score > away_score THEN 1 
            WHEN home_score < away_score THEN 0
            ELSE 0.5  -- Overtime/shootout loss
        END as win,
        home_score as goals_for,
        away_score as goals_against,
        home_sog as shots_for,
        away_sog as shots_against
    FROM games
    WHERE game_state IN ('OFF', 'FINAL')
    
    UNION ALL
    
    SELECT 
        game_id,
        game_date,
        season,
        away_team as team,
        home_team as opponent,
        0 as is_home,
        CASE 
            WHEN away_score > home_score THEN 1 
            WHEN away_score < home_score THEN 0
            ELSE 0.5
        END as win,
        away_score as goals_for,
        home_score as goals_against,
        away_sog as shots_for,
        home_sog as shots_against
    FROM games
    WHERE game_state IN ('OFF', 'FINAL')
)
SELECT 
    team,
    game_date,
    season,
    game_id,
    is_home,
    opponent,
    win,
    goals_for,
    goals_against,
    shots_for,
    shots_against,
    goals_for - goals_against as goal_differential,
    CASE WHEN shots_for > 0 THEN CAST(goals_for AS REAL) / shots_for ELSE 0 END as shooting_pct,
    CASE WHEN shots_against > 0 THEN 1.0 - CAST(goals_against AS REAL) / shots_against ELSE 0 END as save_pct
FROM game_results
"""

team_form_view = """
        CREATE VIEW IF NOT EXISTS team_form_view AS
        SELECT 
            t1.team,
            t1.game_date as as_of_date,
            COUNT(*) as games_played,
            SUM(t2.win) as wins,
            AVG(t2.goals_for) as avg_goals_for,
            AVG(t2.goals_against) as avg_goals_against,
            AVG(t2.goal_differential) as avg_goal_diff,
            AVG(t2.shots_for) as avg_shots_for,
            AVG(t2.shots_against) as avg_shots_against,
            AVG(t2.shooting_pct) as avg_shooting_pct,
            AVG(t2.save_pct) as avg_save_pct,
            SUM(CASE WHEN t2.is_home = 1 THEN t2.win ELSE 0 END) as home_wins,
            SUM(CASE WHEN t2.is_home = 1 THEN 1 ELSE 0 END) as home_games,
            SUM(CASE WHEN t2.is_home = 0 THEN t2.win ELSE 0 END) as away_wins,
            SUM(CASE WHEN t2.is_home = 0 THEN 1 ELSE 0 END) as away_games
        FROM team_rolling_stats t1
        JOIN team_rolling_stats t2 ON t1.team = t2.team 
            AND t2.game_date <= t1.game_date
            AND t2.game_date > date(t1.game_date, '-30 days')
        GROUP BY t1.team, t1.game_date
    """
head_to_head_view = """
        CREATE VIEW IF NOT EXISTS head_to_head_view AS
        SELECT 
            t1.team as team_a,
            t1.opponent as team_b,
            t1.game_date as as_of_date,
            COUNT(*) as games_played,
            SUM(t2.win) as team_a_wins,
            AVG(t2.goals_for) as team_a_avg_goals,
            AVG(t2.goals_against) as team_b_avg_goals
        FROM team_rolling_stats t1
        JOIN team_rolling_stats t2 ON t1.team = t2.team 
            AND t1.opponent = t2.opponent
            AND t2.game_date <= t1.game_date
            AND t2.game_date > date(t1.game_date, '-365 days')
        GROUP BY t1.team, t1.opponent, t1.game_date
    """
player_form_view = """
        CREATE VIEW IF NOT EXISTS player_form_view AS
        SELECT 
            pgs1.player_id,
            g1.game_date as as_of_date,
            COUNT(*) as games_played,
            AVG(pgs2.goals) as avg_goals,
            AVG(pgs2.assists) as avg_assists,
            AVG(pgs2.points) as avg_points,
            AVG(pgs2.shots) as avg_shots,
            AVG(pgs2.plus_minus) as avg_plus_minus,
            SUM(pgs2.powerplay_goals) as total_pp_goals,
            SUM(pgs2.game_winning_goals) as total_gwg
        FROM player_game_stats pgs1
        JOIN games g1 ON pgs1.game_id = g1.game_id
        JOIN player_game_stats pgs2 ON pgs1.player_id = pgs2.player_id
        JOIN games g2 ON pgs2.game_id = g2.game_id
        WHERE g2.game_date <= g1.game_date
            AND g2.game_date > date(g1.game_date, '-30 days')
            AND g1.game_state IN ('OFF', 'FINAL')
            AND g2.game_state IN ('OFF', 'FINAL')
        GROUP BY pgs1.player_id, g1.game_date
    """
team_rest_days_view = """
        CREATE VIEW IF NOT EXISTS team_rest_days_view AS
        WITH team_games AS (
            SELECT team, game_date, game_id,
                LAG(game_date) OVER (PARTITION BY team ORDER BY game_date) as prev_game_date
            FROM (
                SELECT home_team as team, game_date, game_id FROM games WHERE game_state IN ('OFF', 'FINAL')
                UNION ALL
                SELECT away_team as team, game_date, game_id FROM games WHERE game_state IN ('OFF', 'FINAL')
            )
        )
        SELECT 
            team,
            game_date,
            game_id,
            CASE 
                WHEN prev_game_date IS NULL THEN NULL
                ELSE CAST(julianday(game_date) - julianday(prev_game_date) AS INTEGER)
            END as rest_days
        FROM team_games
    """
team_streaks_view = """
        CREATE VIEW IF NOT EXISTS team_streaks_view AS
        WITH game_outcomes AS (
            SELECT 
                team,
                game_date,
                game_id,
                CASE 
                    WHEN win = 1 THEN 'W'
                    WHEN win = 0 THEN 'L'
                    ELSE 'O'
                END as outcome,
                ROW_NUMBER() OVER (PARTITION BY team ORDER BY game_date) as game_num
            FROM team_rolling_stats
        )
        SELECT 
            team,
            game_date,
            game_id,
            outcome,
            game_num
        FROM game_outcomes
    """
player_cumulative_stats = """
        INSERT OR REPLACE INTO player_cumulative_stats
        WITH player_games AS (
            SELECT 
                pgs.player_id,
                g.game_date,
                g.season,
                COALESCE(pgs.goals, 0) as goals,
                COALESCE(pgs.assists, 0) as assists,
                COALESCE(pgs.points, 0) as points,
                COALESCE(pgs.shots, 0) as shots,
                COALESCE(pgs.plus_minus, 0) as plus_minus,
                COALESCE(pgs.powerplay_goals, 0) as powerplay_goals,
                COALESCE(pgs.powerplay_assists, 0) as powerplay_assists,
                COALESCE(pgs.shorthanded_goals, 0) as shorthanded_goals,
                COALESCE(pgs.shorthanded_assists, 0) as shorthanded_assists,
                COALESCE(pgs.game_winning_goals, 0) as game_winning_goals,
                COALESCE(pgs.ot_goals, 0) as ot_goals,
                COALESCE(pgs.hits, 0) as hits,
                COALESCE(pgs.blocked, 0) as blocked,
                COALESCE(pgs.pim, 0) as pim,
                -- Convert TOI to minutes (handle MM:SS format)
                CASE 
                    WHEN pgs.toi LIKE '%:%' THEN 
                        CAST(SUBSTR(pgs.toi, 1, INSTR(pgs.toi, ':') - 1) AS REAL) +
                        CAST(SUBSTR(pgs.toi, INSTR(pgs.toi, ':') + 1) AS REAL) / 60.0
                    WHEN pgs.toi IS NOT NULL AND pgs.toi != '' THEN 
                        CAST(pgs.toi AS REAL)
                    ELSE 0
                END as toi_minutes,
                ROW_NUMBER() OVER (PARTITION BY pgs.player_id ORDER BY g.game_date, g.game_id) as game_num
            FROM player_game_stats pgs
            JOIN games g ON pgs.game_id = g.game_id
            WHERE g.game_state IN ('OFF', 'FINAL')
        ),
        cumulative AS (
            SELECT 
                player_id,
                game_date,
                season,
                game_num as games_played,
                SUM(goals) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_goals,
                SUM(assists) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_assists,
                SUM(points) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_points,
                SUM(shots) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_shots,
                SUM(plus_minus) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_plus_minus,
                SUM(toi_minutes) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_toi_minutes,
                SUM(powerplay_goals) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_powerplay_goals,
                SUM(powerplay_assists) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_powerplay_assists,
                SUM(shorthanded_goals) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_shorthanded_goals,
                SUM(shorthanded_assists) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_shorthanded_assists,
                SUM(game_winning_goals) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_game_winning_goals,
                SUM(ot_goals) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_ot_goals,
                SUM(hits) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_hits,
                SUM(blocked) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_blocked,
                SUM(pim) OVER (PARTITION BY player_id ORDER BY game_date, games_played 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_pim
            FROM player_games
        )
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
            total_powerplay_assists,
            total_shorthanded_goals,
            total_shorthanded_assists,
            total_game_winning_goals,
            total_ot_goals,
            total_hits,
            total_blocked,
            total_pim,
            -- Pre-compute averages
            CAST(total_goals AS REAL) / NULLIF(games_played, 0) as avg_goals,
            CAST(total_assists AS REAL) / NULLIF(games_played, 0) as avg_assists,
            CAST(total_points AS REAL) / NULLIF(games_played, 0) as avg_points,
            CAST(total_shots AS REAL) / NULLIF(games_played, 0) as avg_shots,
            CAST(total_plus_minus AS REAL) / NULLIF(games_played, 0) as avg_plus_minus,
            total_toi_minutes / NULLIF(games_played, 0) as avg_toi,
            CAST(total_pim AS REAL) / NULLIF(games_played, 0) as avg_pim,
            -- Efficiency metrics
            CAST(total_goals AS REAL) / NULLIF(total_shots, 0) as shooting_pct,
            (CAST(total_points AS REAL) * 60.0) / NULLIF(total_toi_minutes, 0) as points_per_60
        FROM cumulative
    """


indexs = ["CREATE INDEX IF NOT EXISTS idx_plays_game_id ON plays(game_id);",
"CREATE INDEX IF NOT EXISTS idx_plays_event_id ON plays(event_id);",
"CREATE INDEX IF NOT EXISTS idx_plays_type ON plays(type_desc_key);",
#"CREATE INDEX IF NOT EXISTS idx_player_cumulative_date ON player_cumulative_stats(game_date);"
#"CREATE INDEX IF NOT EXISTS idx_player_cumulative_player ON player_cumulative_stats(player_id, game_date DESC);",
]