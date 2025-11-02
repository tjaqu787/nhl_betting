"""
Standalone Script: Add Referee Data to Existing NHL Database
Run this once to add game_officials table and fetch referee data for all games
"""

import sqlite3
import requests
import json
from datetime import datetime
import time


def add_officials_tables(db_path: str = "data/nhl_data.db"):
    """Add game_officials table to existing database"""
    print("\n" + "="*60)
    print("ADDING OFFICIALS SCHEMA")
    print("="*60 + "\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Game officials table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_officials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            official_name TEXT,
            official_type TEXT,
            official_number INTEGER,
            data_json TEXT,
            updated_at TIMESTAMP,
            UNIQUE(game_id, official_name, official_type)
        )
    """)
    
    # Create indices
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_game_officials_game_id 
        ON game_officials(game_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_game_officials_name 
        ON game_officials(official_name)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_game_officials_type 
        ON game_officials(official_type)
    """)
    
    conn.commit()
    conn.close()
    
    print("✓ Created game_officials table")
    print("✓ Created indices")
    print("="*60 + "\n")


def fetch_game_officials(game_id: int, session: requests.Session, debug: bool = False) -> dict:
    """
    Fetch officials data for a game from NHL API.
    Uses the /right-rail endpoint which has gameInfo.referees and gameInfo.linesmen
    Returns list of officials or None if not available.
    """
    try:
        # Use right-rail endpoint which has the officials data
        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail"
        response = session.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if debug:
            print(f"\nAPI Response keys: {list(data.keys())}")
            if 'gameInfo' in data:
                print(f"gameInfo keys: {list(data['gameInfo'].keys())}")
                if 'referees' in data['gameInfo']:
                    print(f"Sample referee: {data['gameInfo']['referees'][0]}")
        
        # Extract officials from gameInfo
        officials = []
        
        if 'gameInfo' in data:
            game_info = data['gameInfo']
            
            # Referees - format: [{"default": "Name"}, ...]
            if 'referees' in game_info and game_info['referees']:
                for idx, ref in enumerate(game_info['referees'], 1):
                    if isinstance(ref, dict) and 'default' in ref:
                        name = ref['default'].strip()
                        if name:
                            officials.append({
                                'name': name,
                                'type': 'Referee',
                                'number': idx  # Use position as number since no sweater number
                            })
            
            # Linesmen - format: [{"default": "Name"}, ...]
            if 'linesmen' in game_info and game_info['linesmen']:
                for idx, linesman in enumerate(game_info['linesmen'], 1):
                    if isinstance(linesman, dict) and 'default' in linesman:
                        name = linesman['default'].strip()
                        if name:
                            officials.append({
                                'name': name,
                                'type': 'Linesman',
                                'number': idx  # Use position as number
                            })
        
        return officials if officials else None
        
    except requests.exceptions.RequestException as e:
        if debug:
            print(f"Request error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        if debug:
            print(f"Parse error: {e}")
        return None


def save_officials_to_db(game_id: int, officials: list, conn: sqlite3.Connection):
    """Save officials data to database"""
    cursor = conn.cursor()
    
    for official in officials:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO game_officials 
                (game_id, official_name, official_type, official_number, data_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                game_id,
                official['name'],
                official['type'],
                official['number'],
                json.dumps(official),
                datetime.now()
            ))
        except sqlite3.Error as e:
            print(f"  DB Error for {official['name']}: {e}")
            continue
    
    conn.commit()


def fetch_all_officials(db_path: str = "data/nhl_data.db", max_games: int = None):
    """
    Fetch referee data for all games in database.
    
    Args:
        db_path: Path to NHL database
        max_games: Optional limit on number of games to process (for testing)
    """
    print("\n" + "="*80)
    print("FETCHING REFEREE DATA FOR ALL GAMES")
    print("="*80 + "\n")
    
    # Ensure schema exists
    add_officials_tables(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all completed games that don't have official data yet
    print("Querying games...")
    cursor.execute("""
        SELECT g.game_id, g.game_date, g.home_team, g.away_team
        FROM games g
        WHERE g.game_state IN ('OFF', 'FINAL')
        AND NOT EXISTS (
            SELECT 1 FROM game_officials o 
            WHERE o.game_id = g.game_id
        )
        ORDER BY g.game_date DESC
    """)
    
    games = cursor.fetchall()
    total = len(games)
    
    if max_games:
        games = games[:max_games]
        total = len(games)
    
    print(f"Found {total} games without official data\n")
    
    if total == 0:
        print("✓ All games already have official data!")
        conn.close()
        return
    
    # Create session for requests
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'NHL-Referee-Fetcher/1.0'
    })
    
    successful = 0
    no_data = 0
    failed = 0
    
    
    for idx, (game_id, game_date, home_team, away_team) in enumerate(games, 1):
        try:
            # Progress update
            if idx % 25 == 0:
                pct = (idx / total) * 100
                print(f"Progress: {idx}/{total} ({pct:.1f}%) - "
                      f"✓ {successful} | ⊘ {no_data} | ✗ {failed}")
            
            # Fetch officials
            officials = fetch_game_officials(game_id, session)
            
            if officials:
                save_officials_to_db(game_id, officials, conn)
                successful += 1
            else:
                no_data += 1
            
            # Rate limiting - be respectful to NHL API
            time.sleep(.250)
            
        except Exception as e:
            failed += 1
            if failed % 10 == 0:
                print(f"  ✗ Error on game {game_id} ({game_date}): {str(e)[:50]}")
            continue
    
    session.close()
    
    print("\n" + "="*80)
    print("FETCH COMPLETE")
    print("="*80)
    print(f"Total games processed: {total}")
    print(f"  ✓ Successfully fetched: {successful}")
    print(f"  ⊘ No data available: {no_data} (likely older games)")
    print(f"  ✗ Failed: {failed}")
    
    # Show statistics
    cursor.execute("""
        SELECT COUNT(DISTINCT game_id) as games_with_refs,
               COUNT(*) as total_entries,
               COUNT(DISTINCT official_name) as unique_officials
        FROM game_officials
    """)
    
    stats = cursor.fetchone()
    if stats and stats[0] > 0:
        print("\nDatabase statistics:")
        print(f"  Games with officials: {stats[0]}")
        print(f"  Total official entries: {stats[1]}")
        print(f"  Unique officials: {stats[2]}")
        
        # Show top officials
        cursor.execute("""
            SELECT official_name, official_type, COUNT(*) as games
            FROM game_officials
            GROUP BY official_name, official_type
            ORDER BY games DESC
            LIMIT 5
        """)
        
        top_officials = cursor.fetchall()
        print("\nMost frequent officials:")
        for name, role, count in top_officials:
            print(f"  {name} ({role}): {count} games")
    
    print("="*80 + "\n")
    
    conn.close()


def test_single_game(db_path: str = "data/nhl_data.db"):
    """Test referee fetch on a single recent game"""
    print("\n" + "="*60)
    print("TESTING SINGLE GAME WITH DEBUG")
    print("="*60 + "\n")
    
    # Ensure schema exists
    add_officials_tables(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get most recent game
    cursor.execute("""
        SELECT game_id, game_date, home_team, away_team
        FROM games
        WHERE game_state IN ('OFF', 'FINAL')
        ORDER BY game_date DESC
        LIMIT 1
    """)
    
    game = cursor.fetchone()
    if not game:
        print("No completed games found")
        conn.close()
        return
    
    game_id, game_date, home_team, away_team = game
    print(f"Testing game {game_id}")
    print(f"  {away_team} @ {home_team}")
    print(f"  Date: {game_date}\n")
    
    # Fetch officials WITH DEBUG
    session = requests.Session()
    session.headers.update({'User-Agent': 'NHL-Referee-Fetcher/1.0'})
    
    officials = fetch_game_officials(game_id, session, debug=True)
    
    if officials:
        print(f"\n✓ Found {len(officials)} officials:")
        for official in officials:
            print(f"  {official['type']} #{official['number']}: {official['name']}")
        
        # Save to database
        save_officials_to_db(game_id, officials, conn)
        print("\n✓ Saved to database")
    else:
        print("\n⊘ No official data found")
        print("This could mean:")
        print("  1. API structure has changed")
        print("  2. Game is too old (pre-2020)")
        print("  3. Data not available for this game")
    
    session.close()
    conn.close()
    print("\n" + "="*60 + "\n")


def main():
    """Main execution"""
    import sys
    
    print("\n" + "="*80)
    print("NHL REFEREE DATA MIGRATION TOOL")
    print("="*80)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        command = "help"
    
    if command == "test":
        # Test on single game
        test_single_game()
    
    elif command == "fetch":
        # Fetch all referee data
        max_games = None
        if len(sys.argv) > 2:
            try:
                max_games = int(sys.argv[2])
                print(f"Limited to {max_games} games\n")
            except ValueError:
                pass
        
        fetch_all_officials(max_games=max_games)
    
    elif command == "schema":
        # Just add schema
        add_officials_tables()
        print("✓ Schema added. Run 'python add_referees.py fetch' to download data")
    
    else:
        print("\nUsage:")
        print("  python add_referees.py test        # Test on most recent game")
        print("  python add_referees.py fetch       # Fetch all referee data")
        print("  python add_referees.py fetch 100   # Fetch for 100 games (testing)")
        print("  python add_referees.py schema      # Just add table schema")
        print("\nRecommended: Start with 'test' to verify it works")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()