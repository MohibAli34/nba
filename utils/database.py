import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import os

DATABASE_PATH = 'nba_props_cache.db'


def init_database():
    """Initialize SQLite database with all necessary tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Table 1: Player game logs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_game_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            season TEXT NOT NULL,
            game_date TEXT NOT NULL,
            game_id TEXT,
            matchup TEXT,
            pts INTEGER,
            ast INTEGER,
            reb INTEGER,
            fg3m INTEGER,
            min REAL,
            fgm INTEGER,
            fga INTEGER,
            fg_pct REAL,
            ftm INTEGER,
            fta INTEGER,
            ft_pct REAL,
            oreb INTEGER,
            dreb INTEGER,
            stl INTEGER,
            blk INTEGER,
            tov INTEGER,
            pf INTEGER,
            plus_minus INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(player_id, season, game_date, game_id)
        )
    ''')
    
    # Table 2: Player metadata (position, team, etc.)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_metadata (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL,
            position TEXT,
            team TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table 3: Team stats
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_abbrev TEXT NOT NULL,
            season TEXT NOT NULL,
            def_rating REAL,
            off_rating REAL,
            pace REAL,
            pts_allowed REAL,
            stats_json TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_abbrev, season)
        )
    ''')
    
    # Table 4: Defense vs position rankings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS defense_vs_position (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position TEXT NOT NULL,
            team TEXT NOT NULL,
            rank INTEGER,
            pts REAL,
            fg_pct REAL,
            ft_pct REAL,
            tpm REAL,
            reb REAL,
            ast REAL,
            stl REAL,
            blk REAL,
            tov REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(position, team)
        )
    ''')
    
    # Table 5: Cache metadata / housekeeping
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache_metadata (
            key TEXT PRIMARY KEY,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
    ''')
    
    # Helpful indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_season ON player_game_logs(player_id, season)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_game_date ON player_game_logs(game_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_team_season ON team_stats(team_abbrev, season)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_def_pos ON defense_vs_position(position, team)')
    
    conn.commit()
    conn.close()
    print("[OK] Database initialized successfully")


# ---------- PLAYER GAME LOG CACHE ----------

def get_cached_player_game_logs(player_id, season):
    """
    Get player game logs from cache.
    Returns: DataFrame or None if not cached.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    
    query = '''
        SELECT * FROM player_game_logs 
        WHERE player_id = ? AND season = ?
        ORDER BY game_date DESC
    '''
    
    try:
        df = pd.read_sql_query(query, conn, params=(player_id, season))
        conn.close()
        
        if df.empty:
            return None
        
        # Convert game_date to datetime
        df['GAME_DATE'] = pd.to_datetime(df['game_date'])
        
        # Rename columns to match nba_api-style fields we use everywhere
        column_map = {
            'pts': 'PTS',
            'ast': 'AST',
            'reb': 'REB',
            'fg3m': 'FG3M',
            'min': 'MIN',
            'matchup': 'MATCHUP',
            'game_id': 'GAME_ID',
            'fgm': 'FGM',
            'fga': 'FGA',
            'fg_pct': 'FG_PCT',
            'ftm': 'FTM',
            'fta': 'FTA',
            'ft_pct': 'FT_PCT',
            'oreb': 'OREB',
            'dreb': 'DREB',
            'stl': 'STL',
            'blk': 'BLK',
            'tov': 'TO',
            'pf': 'PF',
            'plus_minus': 'PLUS_MINUS'
        }
        
        df = df.rename(columns=column_map)
        
        print(f"[OK] Loaded {len(df)} games for player {player_id} from cache ({season})")
        return df
        
    except Exception as e:
        print(f"[ERROR] Error loading from cache: {e}")
        conn.close()
        return None


def save_player_game_logs(player_id, player_name, season, game_logs_df):
    """
    Save player game logs to sqlite cache.
    Upserts by (player_id, season, game_date, game_id).
    """
    if game_logs_df is None or game_logs_df.empty:
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    for _, row in game_logs_df.iterrows():
        try:
            game_date = pd.to_datetime(
                row.get('GAME_DATE', '')
            ).strftime('%Y-%m-%d')
            
            cursor.execute('''
                INSERT OR REPLACE INTO player_game_logs (
                    player_id, player_name, season, game_date, game_id, matchup,
                    pts, ast, reb, fg3m, min, fgm, fga, fg_pct,
                    ftm, fta, ft_pct, oreb, dreb, stl, blk, tov, pf, plus_minus
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                player_id,
                player_name,
                season,
                game_date,
                row.get('GAME_ID', ''),
                row.get('MATCHUP', ''),
                int(row.get('PTS', 0)),
                int(row.get('AST', 0)),
                int(row.get('REB', 0)),
                int(row.get('FG3M', 0)),
                float(row.get('MIN', 0)),
                int(row.get('FGM', 0)),
                int(row.get('FGA', 0)),
                float(row.get('FG_PCT', 0)),
                int(row.get('FTM', 0)),
                int(row.get('FTA', 0)),
                float(row.get('FT_PCT', 0)),
                int(row.get('OREB', 0)),
                int(row.get('DREB', 0)),
                int(row.get('STL', 0)),
                int(row.get('BLK', 0)),
                int(row.get('TO', 0)),
                int(row.get('PF', 0)),
                int(row.get('PLUS_MINUS', 0))
            ))
        except Exception as e:
            print(f"[WARN] Error saving game row for {player_id}: {e}")
            continue
    
    conn.commit()
    conn.close()
    print(f"[OK] Saved {len(game_logs_df)} games for {player_id} / {season}")


def get_last_game_date(player_id, season):
    """Get most recent cached game date for a player+season."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT MAX(game_date) FROM player_game_logs
        WHERE player_id = ? AND season = ?
    ''', (player_id, season))
    
    result = cursor.fetchone()[0]
    conn.close()
    
    if result:
        return pd.to_datetime(result)
    return None


# ---------- TEAM STATS CACHE ----------

def save_team_stats(team_abbrev, season, stats_df):
    """
    Save team-level stats to sqlite.
    Stores a JSON dump of the full row so we can reconstruct later.
    """
    if stats_df is None or stats_df.empty:
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # For now we save one row per team_abbrev.
    row = stats_df.iloc[0]
    
    stats_json = stats_df.to_json(orient='records')
    
    cursor.execute('''
        INSERT OR REPLACE INTO team_stats (
            team_abbrev, season, def_rating, off_rating, pace, pts_allowed, stats_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        team_abbrev,
        season,
        float(row.get('DEF_RATING', 110.0)),
        float(row.get('OFF_RATING', 110.0)),
        float(row.get('PACE', 100.0)),
        float(row.get('OPP_PTS', 110.0)),
        stats_json
    ))
    
    conn.commit()
    conn.close()
    print(f"[OK] Saved team stats for {team_abbrev} / {season}")


def get_cached_team_stats(season):
    """
    Load all cached team stats for a season.
    Returns DataFrame or None.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    
    query = '''
        SELECT team_abbrev, stats_json, last_updated
        FROM team_stats
        WHERE season = ?
    '''
    
    try:
        cursor = conn.cursor()
        cursor.execute(query, (season,))
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return None
        
        # freshness: we consider >24h stale
        latest_update = max([pd.to_datetime(r[2]) for r in results])
        if datetime.now() - latest_update > timedelta(hours=24):
            print("[WARN] Team stats cache is stale (>24h)")
            return None
        
        all_rows = []
        for team_abbrev, stats_json, _ in results:
            parsed = json.loads(stats_json)  # list of dict rows
            all_rows.extend(parsed)
        
        df = pd.DataFrame(all_rows)
        print(f"[OK] Loaded cached team stats for season {season} ({len(df)} rows)")
        return df
        
    except Exception as e:
        print(f"[ERROR] Error loading cached team stats: {e}")
        return None


# ---------- DEFENSE VS POSITION CACHE ----------

def save_defense_vs_position(df):
    """
    Cache defense-vs-position scrape.
    We store each (Position, Team) row.
    """
    if df is None or df.empty:
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    for _, row in df.iterrows():
        cursor.execute('''
            INSERT OR REPLACE INTO defense_vs_position (
                position, team, rank, pts, fg_pct, ft_pct,
                tpm, reb, ast, stl, blk, tov
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row.get('Position', ''),
            row.get('Team', ''),
            int(row.get('Rank', 15)),
            float(row.get('PTS', 0)),
            float(row.get('FG_PCT', 0)),
            float(row.get('FT_PCT', 0)),
            float(row.get('TPM', 0)),
            float(row.get('REB', 0)),
            float(row.get('AST', 0)),
            float(row.get('STL', 0)),
            float(row.get('BLK', 0)),
            float(row.get('TO', 0))
        ))
    
    conn.commit()
    conn.close()
    print(f"[OK] Saved defense vs position data ({len(df)} rows)")


def get_cached_defense_vs_position():
    """
    Load cached defense-vs-position table.
    Returns df or None.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    
    query = '''
        SELECT position, team, rank, pts, fg_pct, ft_pct,
               tpm, reb, ast, stl, blk, tov, last_updated
        FROM defense_vs_position
    '''
    
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return None
        
        latest_update = pd.to_datetime(df['last_updated'].max())
        # treat older than 7 days as stale
        if datetime.now() - latest_update > timedelta(days=7):
            print("[WARN] Defense-vs-position cache is stale (>7d)")
            return None
        
        # rename columns back to what code expects
        df = df.rename(columns={
            'position': 'Position',
            'team': 'Team',
            'rank': 'Rank',
            'pts': 'PTS',
            'fg_pct': 'FG_PCT',
            'ft_pct': 'FT_PCT',
            'tpm': 'TPM',
            'reb': 'REB',
            'ast': 'AST',
            'stl': 'STL',
            'blk': 'BLK',
            'tov': 'TO'
        })
        
        df = df.drop(columns=['last_updated'])
        
        print(f"[OK] Loaded defense vs position from cache ({len(df)} rows)")
        return df
        
    except Exception as e:
        print(f"[ERROR] Error loading defense-vs-position cache: {e}")
        conn.close()
        return None


# ---------- PLAYER METADATA CACHE (POSITION / TEAM) ----------

def get_player_metadata(player_id):
    """
    Return dict with player_name, position, team, last_updated
    or None if this player_id not cached yet.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT player_id, player_name, position, team, last_updated
        FROM player_metadata
        WHERE player_id = ?
    ''', (player_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        'player_id': row[0],
        'player_name': row[1],
        'position': row[2],
        'team': row[3],
        'last_updated': row[4]
    }


def save_player_metadata(player_id, player_name, position, team):
    """
    Upsert player's metadata (id, name, primary position, team).
    This is what get_player_position() in data_fetcher.py calls.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO player_metadata (
            player_id, player_name, position, team, last_updated
        ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (player_id, player_name, position, team))
    
    conn.commit()
    conn.close()
    print(f"[OK] Saved player_metadata for {player_id}: {player_name}, {position}, {team}")


# ---------- STATS ABOUT CACHE ----------

def get_cache_stats():
    """Return high-level stats about cache DB size/content."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    stats = {}
    
    # Total players cached
    cursor.execute('SELECT COUNT(DISTINCT player_id) FROM player_game_logs')
    stats['total_players'] = cursor.fetchone()[0]
    
    # Total games cached
    cursor.execute('SELECT COUNT(*) FROM player_game_logs')
    stats['total_games'] = cursor.fetchone()[0]
    
    # DB size
    if os.path.exists(DATABASE_PATH):
        stats['db_size_mb'] = os.path.getsize(DATABASE_PATH) / (1024 * 1024)
    else:
        stats['db_size_mb'] = 0.0
    
    conn.close()
    return stats


def clear_cache():
    """Delete the entire sqlite file and recreate schema."""
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
        print("[OK] Cache cleared (DB file deleted)")
        init_database()
    else:
        print("[WARN] No cache DB to clear")


def clear_old_seasons(keep_seasons):
    """
    Keep only seasons in keep_seasons list (e.g. ['2024-25','2023-24']),
    purge older seasons from player_game_logs + team_stats.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    placeholders = ','.join('?' * len(keep_seasons))
    
    cursor.execute(f'''
        DELETE FROM player_game_logs 
        WHERE season NOT IN ({placeholders})
    ''', keep_seasons)
    
    cursor.execute(f'''
        DELETE FROM team_stats 
        WHERE season NOT IN ({placeholders})
    ''', keep_seasons)
    
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    print(f"🧹 Removed data from old seasons ({deleted} records)")


# ---------- INIT ON IMPORT ----------

if not os.path.exists(DATABASE_PATH):
    init_database()
