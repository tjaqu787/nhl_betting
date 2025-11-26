import sqlite3
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time
import schema

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
        cursor.execute(schema.teams)
        
        # Games table - comprehensive game information
        cursor.execute(schema.games)
        
        # Player game stats - detailed per-game performance
        cursor.execute(schema.player_game_stats)
        
        # Team game stats - team-level per-game stats
        cursor.execute(schema.team_game_stats)
        
        # Players table
        cursor.execute(schema.players)
        
        # Rosters table
        cursor.execute(schema.rosters)
        
        # Plays Table
        cursor.execute(schema.plays)

        # View 1: Team rolling stats (last N games as of any date)
        cursor.execute(schema.team_rolling_stats_view)
        
        # View 2: Team form as of any date (last N games)
        cursor.execute(schema.team_form_view)
        
        # View 3: Head-to-head records as of any date
        cursor.execute(schema.head_to_head_view)
        
        # View 4: Player form (last N games)
        cursor.execute(schema.player_form_view)
        
        # View 5: Rest days (days between games)
        cursor.execute(schema.team_rest_days_view)
        
        # View 6: Streak tracking
        cursor.execute(schema.team_streaks_view)
        
        # Create table
        print('creating cumulative stats')
        #cursor.execute(schema.player_cumulative_stats)
        
        # creat indexs
        for index in schema.indexs:
            cursor.execute(index)


        self.conn.commit()
        print(f"✓ Database initialized at {self.db_path}")
    

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
                period_desc = play.get('periodDescriptor', {})
                
                cursor.execute("""
                    INSERT OR REPLACE INTO plays VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    None,  # play_id (autoincrement)
                    game_id,
                    play.get('eventId'),
                    period_desc.get('number'),
                    period_desc.get('periodType'),
                    play.get('timeInPeriod'),
                    play.get('timeRemaining'),
                    play.get('typeCode'),
                    play.get('typeDescKey'),
                    play.get('sortOrder'),
                    json.dumps(play.get('details', {})),
                    datetime.now()
                ))
        
                
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
                    NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
                None,  # ot_goals
                None,  # faceoff_pct
                json.dumps({'team_id': stats['team'], **stats}),
                datetime.now()
            ))
            # Store all plays

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