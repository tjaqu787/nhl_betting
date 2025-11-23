import sqlite3
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time

class NHLDataScraper:
    BASE_URL = "https://api-web.nhle.com/v1"
    
    def __init__(self, db_path: str = "data/nhl_data.db"):
        self.db_path = db_path
        self.conn = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NHL-Data-Scraper/1.0'
        })
        self.setup_database()
        self.create_feature_views()
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make GET request to NHL API"""
        try:
            url = f"{self.BASE_URL}{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  ✗ API Error for {endpoint}: {e}")
            return None
    
    def setup_database(self):
        """Create all necessary tables"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Teams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_abbrev TEXT PRIMARY KEY,
                team_name TEXT,
                franchise_id INTEGER,
                conference TEXT,
                division TEXT,
                data_json TEXT,
                updated_at TIMESTAMP
            )
        """)
        
        # Games table - comprehensive game information
        cursor.execute("""
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
        """)
        
        # Player game stats - detailed per-game performance
        cursor.execute("""
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
                game_winning_goals INTEGER,
                ot_goals INTEGER,
                faceoff_pct REAL,
                hits INTEGER,
                blocked INTEGER,
                data_json TEXT,
                updated_at TIMESTAMP,
                UNIQUE(player_id, game_id)
            )
        """)
        
        # Team game stats - team-level per-game stats
        cursor.execute("""
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
        """)
        
        # Players table
        cursor.execute("""
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
        """)
        
        # Rosters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rosters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_abbrev TEXT,
                player_id INTEGER,
                season TEXT,
                position_code TEXT,
                last_updated TIMESTAMP,
                UNIQUE(team_abbrev, player_id, season)
            )
        """)
        
        self.conn.commit()
        print(f"✓ Database initialized at {self.db_path}")
    
    def create_feature_views(self):
        """Create SQL views for feature engineering"""
        cursor = self.conn.cursor()
        
        # View 1: Team rolling stats (last N games as of any date)
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS team_rolling_stats AS
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
        """)
        
        # View 2: Team form as of any date (last N games)
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS team_form AS
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
        """)
        
        # View 3: Head-to-head records as of any date
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS head_to_head AS
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
        """)
        
        # View 4: Player form (last N games)
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS player_form AS
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
        """)
        
        # View 5: Rest days (days between games)
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS team_rest_days AS
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
        """)
        
        # View 6: Streak tracking
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS team_streaks AS
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
        """)
        
        self.conn.commit()
        print("✓ Feature engineering views created")
    
    def fetch_schedule(self, date: str) -> int:
        """Fetch schedule for a specific date (YYYY-MM-DD)"""
        # Using /score endpoint which returns games with basic info + scores
        endpoint = f"/score/{date}"
        data = self._get(endpoint)
        
        if not data or 'games' not in data:
            return 0
        
        cursor = self.conn.cursor()
        count = 0
        
        for game in data['games']:
            cursor.execute("""
                INSERT OR REPLACE INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game.get('id'),
                game.get('season'),
                game.get('gameType'),
                game.get('gameDate'),
                game.get('startTimeUTC'),
                game.get('homeTeam', {}).get('abbrev'),
                game.get('awayTeam', {}).get('abbrev'),
                game.get('homeTeam', {}).get('score'),
                game.get('awayTeam', {}).get('score'),
                game.get('gameState'),
                game.get('period'),
                game.get('venue', {}).get('default'),
                game.get('homeTeam', {}).get('sog'),
                game.get('awayTeam', {}).get('sog'),
                json.dumps(game),
                datetime.now()
            ))
            
            # Also store team info from schedule
            for team_type in ['homeTeam', 'awayTeam']:
                team = game.get(team_type, {})
                if team.get('abbrev'):
                    cursor.execute("""
                        INSERT OR IGNORE INTO teams (team_abbrev, team_name, updated_at)
                        VALUES (?, ?, ?)
                    """, (
                        team.get('abbrev'),
                        team.get('name', {}).get('default'),
                        datetime.now()
                    ))
            
            count += 1
        
        self.conn.commit()
        return count
    
    def fetch_play_by_play(self, game_id: int) -> bool:
        """Fetch detailed play-by-play data for a game (includes full boxscore info)"""
        endpoint = f"/gamecenter/{game_id}/play-by-play"
        data = self._get(endpoint)
        
        if not data:
            return False
        
        cursor = self.conn.cursor()
        
        # Extract roster info which includes player details
        if 'rosterSpots' in data:
            for roster_entry in data['rosterSpots']:
                player_id = roster_entry.get('playerId')
                if not player_id:
                    continue
                
                cursor.execute("""
                    INSERT OR REPLACE INTO players VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    player_id,
                    f"{roster_entry.get('firstName', {}).get('default', '')} {roster_entry.get('lastName', {}).get('default', '')}",
                    roster_entry.get('firstName', {}).get('default'),
                    roster_entry.get('lastName', {}).get('default'),
                    roster_entry.get('positionCode'),
                    None,  # shoots_catches not in roster
                    None,  # height
                    None,  # weight
                    None,  # birth_date
                    None,  # birth_city
                    None,  # birth_country
                    roster_entry.get('teamId'),
                    roster_entry.get('sweaterNumber'),
                    roster_entry.get('headshot'),
                    json.dumps(roster_entry),
                    datetime.now()
                ))
        
        # Extract game-level stats from plays (goals, assists, shots, etc.)
        # We'll build player game stats from the plays
        player_stats = {}
        
        if 'plays' in data:
            for play in data['plays']:
                play_type = play.get('typeDescKey')
                
                # Track shots on goal
                if play_type == 'shot-on-goal' and 'details' in play:
                    shooter_id = play['details'].get('shootingPlayerId')
                    if shooter_id:
                        if shooter_id not in player_stats:
                            player_stats[shooter_id] = {
                                'team': play['details'].get('eventOwnerTeamId'),
                                'shots': 0, 'goals': 0, 'assists': 0
                            }
                        player_stats[shooter_id]['shots'] += 1
                
                # Track goals and assists
                elif play_type == 'goal' and 'details' in play:
                    scorer_id = play['details'].get('scoringPlayerId')
                    if scorer_id:
                        if scorer_id not in player_stats:
                            player_stats[scorer_id] = {
                                'team': play['details'].get('eventOwnerTeamId'),
                                'shots': 0, 'goals': 0, 'assists': 0
                            }
                        player_stats[scorer_id]['goals'] += 1
                    
                    # Assists
                    for i in [1, 2]:
                        assist_key = f'assist{i}PlayerId'
                        if assist_key in play['details']:
                            assist_id = play['details'][assist_key]
                            if assist_id:
                                if assist_id not in player_stats:
                                    player_stats[assist_id] = {
                                        'team': play['details'].get('eventOwnerTeamId'),
                                        'shots': 0, 'goals': 0, 'assists': 0
                                    }
                                player_stats[assist_id]['assists'] += 1
        
        # Insert player game stats
        for player_id, stats in player_stats.items():
            cursor.execute("""
                INSERT OR REPLACE INTO player_game_stats VALUES (
                    NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                player_id,
                game_id,
                None,  # team_abbrev - we have team_id in stats
                None,  # position
                stats['goals'],
                stats['assists'],
                stats['goals'] + stats['assists'],
                None,  # plus_minus
                None,  # pim
                stats['shots'],
                None,  # toi
                None,  # powerplay_goals
                None,  # powerplay_assists
                None,  # shorthanded_goals
                None,  # shorthanded_assists
                None,  # game_winning_goals
                None,  # ot_goals
                None,  # faceoff_pct
                None,  # hits
                None,  # blocked
                json.dumps({'team_id': stats['team'], **stats}),
                datetime.now()
            ))
        
        self.conn.commit()
        return True
    
    def fetch_date_range_schedule(self, start_date: str, end_date: str) -> int:
        """Fetch schedule for a date range"""
        print(f"Fetching schedule from {start_date} to {end_date}...")
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        total = 0
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            count = self.fetch_schedule(date_str)
            if count > 0:
                print(f"  ✓ {date_str}: {count} games")
            total += count
            current += timedelta(days=1)
            time.sleep(0.5)
        
        print(f"  Total: {total} games")
        return total
    
    def fetch_game_boxscore(self, game_id: int) -> bool:
        """
        Fetch play-by-play which includes all game data we need.
        This replaces the old boxscore endpoint.
        Returns False if data unavailable (e.g., very old games).
        """
        return self.fetch_play_by_play(game_id)
    
    def fetch_game_officials(self, game_id: int) -> bool:
        """
        Fetch referee and official data from right-rail endpoint.
        Returns True if successful, False otherwise.
        """
        endpoint = f"/gamecenter/{game_id}/right-rail"
        data = self._get(endpoint)
        
        if not data:
            return False
        
        # Extract referee information from gameInfo
        game_info = data.get('gameInfo', {})
        
        if not game_info:
            return False
        
        referees = game_info.get('referees', [])
        linesmen = game_info.get('linesmen', [])
        
        # If no officials data, return False
        if not referees and not linesmen:
            return False
        
        cursor = self.conn.cursor()
        
        # Store each referee
        for idx, ref_obj in enumerate(referees):
            ref_name = ref_obj.get('default', '')
            if ref_name:
                cursor.execute("""
                    INSERT OR REPLACE INTO game_officials 
                    VALUES (NULL, ?, ?, 'REFEREE', ?, ?, ?)
                """, (
                    game_id,
                    ref_name,
                    idx + 1,  # referee number (1 or 2)
                    json.dumps(game_info),
                    datetime.now()
                ))
        
        # Store each linesman
        for idx, lines_obj in enumerate(linesmen):
            lines_name = lines_obj.get('default', '')
            if lines_name:
                cursor.execute("""
                    INSERT OR REPLACE INTO game_officials 
                    VALUES (NULL, ?, ?, 'LINESMAN', ?, ?, ?)
                """, (
                    game_id,
                    lines_name,
                    idx + 1,  # linesman number (1 or 2)
                    json.dumps(game_info),
                    datetime.now()
                ))
        
        self.conn.commit()
        return len(referees) > 0 or len(linesmen) > 0
    
    def fetch_recent_games_stats(self, days: int = 7) -> int:
        """Fetch boxscores for recent completed games"""
        print(f"\nFetching game stats for last {days} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch schedules first
        self.fetch_date_range_schedule(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Now fetch boxscores for completed games
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT game_id FROM games 
            WHERE game_date >= ? AND game_date <= ?
            AND game_state IN ('OFF', 'FINAL')
        """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        
        games = cursor.fetchall()
        total = 0
        
        for (game_id,) in games:
            print(f"  Fetching boxscore {game_id}...")
            if self.fetch_game_boxscore(game_id):
                total += 1
            time.sleep(1)
        
        print(f"  ✓ Fetched stats for {total} games")
        return total
    
    def fetch_historical_season(self, season: str, start_date: str, end_date: str) -> int:
        """Fetch an entire historical season"""
        print(f"\nFetching season {season} from {start_date} to {end_date}...")
        
        # Fetch all schedules
        count = self.fetch_date_range_schedule(start_date, end_date)
        
        # Fetch play-by-play for all completed games in this range
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT game_id FROM games 
            WHERE season = ? AND game_state IN ('OFF', 'FINAL')
        """, (season,))
        
        games = cursor.fetchall()
        total = 0
        failed = 0
        
        print(f"  Fetching play-by-play for {len(games)} completed games...")
        for (game_id,) in games:
            try:
                if self.fetch_play_by_play(game_id):
                    total += 1
                else:
                    failed += 1
                
                if total % 10 == 0:
                    print(f"    Progress: {total}/{len(games)} (failed: {failed})")
                time.sleep(1)
            except Exception as e:
                failed += 1
                if failed % 10 == 0:
                    print(f"    Failed count: {failed}")
                continue
        
        print(f"  ✓ Successfully fetched {total} games, {failed} failed (likely old games without data)")
        return total
    
    def get_available_seasons(self) -> List[str]:
        """Get list of all available seasons from NHL API"""
        print("Discovering available seasons...")
        endpoint = "/season"
        data = self._get(endpoint)
        
        if not data:
            print("  ✗ Could not fetch season list")
            return []
        
        seasons = [str(s) for s in data if isinstance(s, int)]
        print(f"  ✓ Found {len(seasons)} seasons")
        return seasons
    
    def fetch_all_seasons(self, start_year: Optional[int] = None, end_year: Optional[int] = None):
        """
        Fetch all available historical seasons
        Args:
            start_year: Optional starting year (e.g., 2010 for 2010-11 season)
            end_year: Optional ending year (e.g., 2024 for 2023-24 season)
        """
        print("\n" + "="*60)
        print("FETCHING ALL HISTORICAL SEASONS")
        print("="*60 + "\n")
        
        seasons = self.get_available_seasons()
        
        if not seasons:
            print("No seasons found!")
            return
        
        # Filter by year range if specified
        if start_year or end_year:
            filtered_seasons = []
            for season in seasons:
                if len(season) == 8:
                    season_start = int(season[:4])
                    if start_year and season_start < start_year:
                        continue
                    if end_year and season_start > end_year:
                        continue
                    filtered_seasons.append(season)
            seasons = filtered_seasons
            print(f"Filtered to {len(seasons)} seasons based on year range")
        
        # Sort seasons
        seasons.sort()
        
        total_games = 0
        successful_seasons = 0
        
        for season in seasons:
            try:
                # Parse season string (e.g., "20232024")
                if len(season) != 8:
                    continue
                
                start_year_str = season[:4]
                end_year_str = season[4:]
                
                # NHL season typically runs Oct-June
                start_date = f"{start_year_str}-10-01"
                end_date = f"{end_year_str}-06-30"
                
                print(f"\n{'='*60}")
                print(f"Season {season} ({start_year_str}-{end_year_str[-2:]})")
                print(f"{'='*60}")
                
                games = self.fetch_historical_season(season, start_date, end_date)
                total_games += games
                successful_seasons += 1
                
            except Exception as e:
                print(f"  ✗ Error fetching season {season}: {e}")
                continue
        
        print("\n" + "="*60)
        print(f"COMPLETE: {successful_seasons} seasons, {total_games} games")
        print("="*60)
    
    
    def get_team_features_as_of_date(self, team: str, as_of_date: str, lookback_days: int = 30):
        """Get all features for a team as of a specific date"""
        cursor = self.conn.cursor()
        
        # Query team form
        cursor.execute("""
            SELECT * FROM team_form
            WHERE team = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC
            LIMIT 1
        """, (team, as_of_date))
        
        return cursor.fetchone()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
        self.session.close()
        print("\n✓ Database connection closed")


def main():
    """Main execution function"""
    scraper = NHLDataScraper()
    
    print("\n" + "="*60)
    print("NHL Data Scraper - Comprehensive Edition")
    print("="*60 + "\n")
    
    try:
        # Choose your mode:
        
        # MODE 1: Fetch recent data only (for live predictions)
        # scraper.fetch_recent_games_stats(days=30)
        
        # MODE 3: Fetch ALL available seasons
        scraper.fetch_all_seasons()
        
        # MODE 4: Fetch specific year range
        # scraper.fetch_all_seasons(start_year=2018, end_year=2024)
        
        # MODE 5: Fetch single historical season
        # scraper.fetch_historical_season(
        #     season="20232024",
        #     start_date="2023-10-01",
        #     end_date="2024-06-30"
        # )
        
        print("\n" + "="*60)
        print("Data Collection Complete!")
        print("="*60)
        print(f"\nDatabase: {scraper.db_path}")
        print("\nAvailable feature views:")
        print("  - team_rolling_stats: Per-game stats for all teams")
        print("  - team_form: Aggregated form over last N days")
        print("  - head_to_head: H2H records between teams")
        print("  - player_form: Player performance over last N games")
        print("  - team_rest_days: Days of rest between games")
        print("  - team_streaks: Win/loss streak tracking")
        print("\nExample query:")
        print("  SELECT * FROM team_form WHERE team = 'TOR' AND as_of_date = '2024-10-20';")
        
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()


if __name__ == "__main__":
    main()