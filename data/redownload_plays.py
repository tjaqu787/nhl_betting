"""
Download Play-by-Play Data for All Games
Run this to populate the plays table for all games in the database
"""

import sqlite3
import time
from datetime import datetime
import sys
import os

# Add parent directory to path to import from data module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.data_downloader import NHLDataScraper


def add_plays_table(db_path: str = "data/nhl_data.db"):
    """Add plays table to database if it doesn't exist"""
    print("\n" + "="*60)
    print("ADDING PLAYS TABLE SCHEMA")
    print("="*60 + "\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create plays table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plays (
            play_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            event_id INTEGER UNIQUE NOT NULL,
            period_number INTEGER,
            period_type TEXT,
            time_in_period TEXT,
            time_remaining TEXT,
            type_code INTEGER,
            type_desc_key TEXT,
            sort_order INTEGER,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    """)
    
    # Create indices for performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_plays_game_id 
        ON plays(game_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_plays_event_id 
        ON plays(event_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_plays_type 
        ON plays(type_desc_key)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_plays_period 
        ON plays(game_id, period_number)
    """)
    
    conn.commit()
    conn.close()
    
    print("✓ Created plays table")
    print("✓ Created indices")
    print("="*60 + "\n")


def get_games_without_plays(db_path: str = "data/nhl_data.db", limit: int = None):
    """Get all completed games that don't have play-by-play data yet"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
        SELECT g.game_id, g.game_date, g.season, g.home_team, g.away_team,
               g.home_score, g.away_score
        FROM games g
        WHERE g.game_state IN ('OFF', 'FINAL')
        AND NOT EXISTS (
            SELECT 1 FROM plays p 
            WHERE p.game_id = g.game_id
        )
        ORDER BY g.game_date DESC
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    games = cursor.fetchall()
    conn.close()
    
    return games


def get_games_with_incomplete_plays(db_path: str = "data/nhl_data.db", min_plays: int = 10):
    """Get games that have some plays but might be incomplete"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT g.game_id, g.game_date, g.season, g.home_team, g.away_team,
               COUNT(p.play_id) as play_count
        FROM games g
        LEFT JOIN plays p ON g.game_id = p.game_id
        WHERE g.game_state IN ('OFF', 'FINAL')
        GROUP BY g.game_id
        HAVING play_count < ?
        ORDER BY g.game_date DESC
    """, (min_plays,))
    
    games = cursor.fetchall()
    conn.close()
    
    return games


def download_plays_for_games(db_path: str = "data/nhl_data.db", 
                             max_games: int = None,
                             rate_limit_delay: float = 0.5):
    """
    Download play-by-play data for all games missing plays.
    
    Args:
        db_path: Path to NHL database
        max_games: Optional limit on number of games to process
        rate_limit_delay: Seconds to wait between API calls
    """
    print("\n" + "="*80)
    print("DOWNLOADING PLAY-BY-PLAY DATA")
    print("="*80 + "\n")
    
    # Ensure plays table exists
    add_plays_table(db_path)
    
    # Get games without plays
    print("Querying games without play-by-play data...")
    games = get_games_without_plays(db_path, limit=max_games)
    total = len(games)
    
    print(f"Found {total} games without play-by-play data\n")
    
    if total == 0:
        print("✓ All games already have play-by-play data!")
        return
    
    # Initialize scraper
    scraper = NHLDataScraper(db_path)
    
    successful = 0
    failed = 0
    no_data = 0
    
    print("Starting download...")
    print("-" * 80)
    
    for idx, (game_id, game_date, season, home_team, away_team, home_score, away_score) in enumerate(games, 1):
        try:
            # Progress update
            if idx % 10 == 0 or idx == 1:
                pct = (idx / total) * 100
                print(f"\nProgress: {idx}/{total} ({pct:.1f}%)")
                print(f"  ✓ Success: {successful} | ⊘ No Data: {no_data} | ✗ Failed: {failed}")
                print(f"  Current: Game {game_id} ({game_date})")
                print(f"    {away_team} @ {home_team}: {away_score}-{home_score}")
            
            # Fetch play-by-play
            result = scraper.fetch_play_by_play(game_id)
            
            if result:
                successful += 1
                if idx % 10 == 0:
                    print(f"  ✓ Downloaded plays for game {game_id}")
            else:
                no_data += 1
                if no_data % 50 == 0:
                    print(f"  ⊘ No data available for game {game_id} (old game?)")
            
            # Rate limiting - be respectful to NHL API
            time.sleep(rate_limit_delay)
            
        except KeyboardInterrupt:
            print("\n\n✗ Interrupted by user")
            break
        except Exception as e:
            failed += 1
            if failed % 10 == 0:
                print(f"  ✗ Error on game {game_id}: {str(e)[:100]}")
            continue
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"Total games processed: {idx}")
    print(f"  ✓ Successfully downloaded: {successful}")
    print(f"  ⊘ No data available: {no_data} (likely pre-2010 games)")
    print(f"  ✗ Failed: {failed}")
    
    # Show statistics
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(DISTINCT game_id) as games_with_plays,
               COUNT(*) as total_plays,
               AVG(play_count) as avg_plays_per_game
        FROM (
            SELECT game_id, COUNT(*) as play_count
            FROM plays
            GROUP BY game_id
        )
    """)
    
    stats = cursor.fetchone()
    if stats and stats[0] > 0:
        print("\nDatabase statistics:")
        print(f"  Games with plays: {stats[0]}")
        print(f"  Total plays: {stats[1]}")
        print(f"  Avg plays per game: {stats[2]:.1f}")
        
        # Show play type distribution
        cursor.execute("""
            SELECT type_desc_key, COUNT(*) as count
            FROM plays
            GROUP BY type_desc_key
            ORDER BY count DESC
            LIMIT 10
        """)
        
        play_types = cursor.fetchall()
        print("\nMost common play types:")
        for play_type, count in play_types:
            print(f"  {play_type}: {count:,}")
    
    print("="*80 + "\n")
    
    conn.close()
    scraper.close()


def download_plays_for_season(season: str, db_path: str = "data/nhl_data.db"):
    """Download plays for a specific season"""
    print(f"\n{'='*80}")
    print(f"DOWNLOADING PLAYS FOR SEASON {season}")
    print(f"{'='*80}\n")
    
    # Ensure plays table exists
    add_plays_table(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all games for this season without plays
    cursor.execute("""
        SELECT g.game_id, g.game_date, g.home_team, g.away_team
        FROM games g
        WHERE g.season = ?
        AND g.game_state IN ('OFF', 'FINAL')
        AND NOT EXISTS (
            SELECT 1 FROM plays p WHERE p.game_id = g.game_id
        )
        ORDER BY g.game_date
    """, (season,))
    
    games = cursor.fetchall()
    conn.close()
    
    print(f"Found {len(games)} games in season {season} without plays\n")
    
    if len(games) == 0:
        print("✓ All games for this season already have plays!")
        return
    
    # Download using main function
    scraper = NHLDataScraper(db_path)
    
    successful = 0
    failed = 0
    
    for idx, (game_id, game_date, home_team, away_team) in enumerate(games, 1):
        try:
            if idx % 25 == 0:
                print(f"Progress: {idx}/{len(games)} ({idx/len(games)*100:.1f}%)")
            
            if scraper.fetch_play_by_play(game_id):
                successful += 1
            else:
                failed += 1
            
            time.sleep(0.5)
            
        except Exception as e:
            failed += 1
            continue
    
    print(f"\n✓ Season {season} complete: {successful} successful, {failed} failed")
    scraper.close()


def verify_play_data(db_path: str = "data/nhl_data.db"):
    """Verify play data integrity"""
    print("\n" + "="*60)
    print("VERIFYING PLAY DATA")
    print("="*60 + "\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check for games with suspiciously few plays
    cursor.execute("""
        SELECT g.game_id, g.game_date, g.home_team, g.away_team,
               COUNT(p.play_id) as play_count
        FROM games g
        LEFT JOIN plays p ON g.game_id = p.game_id
        WHERE g.game_state IN ('OFF', 'FINAL')
        GROUP BY g.game_id
        HAVING play_count < 50 AND play_count > 0
        ORDER BY play_count
        LIMIT 20
    """)
    
    suspicious = cursor.fetchall()
    
    if suspicious:
        print(f"⚠ Found {len(suspicious)} games with < 50 plays:")
        for game_id, game_date, home, away, count in suspicious[:10]:
            print(f"  Game {game_id} ({game_date}): {away} @ {home} - {count} plays")
        print()
    
    # Check for duplicate event_ids
    cursor.execute("""
        SELECT event_id, COUNT(*) as count
        FROM plays
        GROUP BY event_id
        HAVING count > 1
        LIMIT 10
    """)
    
    duplicates = cursor.fetchall()
    if duplicates:
        print(f"⚠ Found {len(duplicates)} duplicate event_ids")
    else:
        print("✓ No duplicate event_ids")
    
    # Check coverage by season
    cursor.execute("""
        SELECT g.season,
               COUNT(DISTINCT g.game_id) as total_games,
               COUNT(DISTINCT p.game_id) as games_with_plays,
               ROUND(COUNT(DISTINCT p.game_id) * 100.0 / COUNT(DISTINCT g.game_id), 1) as coverage_pct
        FROM games g
        LEFT JOIN plays p ON g.game_id = p.game_id
        WHERE g.game_state IN ('OFF', 'FINAL')
        GROUP BY g.season
        ORDER BY g.season DESC
        LIMIT 15
    """)
    
    coverage = cursor.fetchall()
    print("\nPlay coverage by season:")
    print("-" * 60)
    for season, total, with_plays, pct in coverage:
        print(f"  {season}: {with_plays}/{total} games ({pct}%)")
    
    print("\n" + "="*60 + "\n")
    conn.close()


def main():
    """Main execution"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        command = "help"
    
    if command == "all":
        # Download plays for all games
        max_games = None
        if len(sys.argv) > 2:
            try:
                max_games = int(sys.argv[2])
            except ValueError:
                pass
        
        download_plays_for_games(max_games=max_games)
    
    elif command == "season":
        # Download plays for specific season
        if len(sys.argv) < 3:
            print("Error: Please specify season (e.g., 20232024)")
            return
        
        season = sys.argv[2]
        download_plays_for_season(season)
    
    elif command == "verify":
        # Verify data integrity
        verify_play_data()
    
    elif command == "test":
        # Test on small sample
        print("Testing on 10 games...")
        download_plays_for_games(max_games=10, rate_limit_delay=1.0)
    
    else:
        print("\nNHL Play-by-Play Downloader")
        print("="*60)
        print("\nUsage:")
        print("  python download_plays.py all [limit]    # Download for all games")
        print("  python download_plays.py season 20232024 # Download for specific season")
        print("  python download_plays.py test           # Test on 10 games")
        print("  python download_plays.py verify         # Verify data integrity")
        print("\nExamples:")
        print("  python download_plays.py all            # Download all")
        print("  python download_plays.py all 100        # Download 100 games")
        print("  python download_plays.py season 20232024")
        print("\nRecommended: Start with 'test' to verify it works")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()