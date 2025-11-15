"""
Game data caching utilities for Firebase.
Fetches and caches all data needed for a specific game matchup.
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from . import firebase_cache
    FIREBASE_CACHE_AVAILABLE = True
except ImportError:
    FIREBASE_CACHE_AVAILABLE = False


def get_game_cache_key(home_team: str, away_team: str, game_date: str = None) -> str:
    """
    Generate a unique cache key for a game.
    
    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        game_date: Optional game date for additional uniqueness
    
    Returns:
        Cache key string
    """
    # Normalize team abbreviations to uppercase for consistency
    home_team = str(home_team).upper().strip()
    away_team = str(away_team).upper().strip()
    
    if game_date:
        # Normalize date format
        game_date = str(game_date).strip()
        return f"game_data_{away_team}_{home_team}_{game_date}"
    return f"game_data_{away_team}_{home_team}"


def fetch_game_data_internal(
    home_team: str,
    away_team: str,
    cur_season: str,
    prev_season: str,
    game_date: str = None
) -> Dict[str, Any]:
    """
    Internal function to fetch ALL game data from APIs.
    This is called only when cache is missing or stale.
    All API calls are wrapped in try/except to ensure the function always returns data.
    
    Returns:
        Dictionary containing all game data (even if some parts failed):
        - rosters (home_roster, away_roster)
        - starters (starters_dict)
        - event_id
        - odds_data
        - shared_data (def_vs_pos_df, team_stats)
    """
    from .data_fetcher import (
        get_players_by_team,
        get_event_id_for_game,
        fetch_fanduel_lines,
        scrape_rotowire_starters,
    )
    from .cached_data_fetcher import (
        scrape_defense_vs_position_cached_db,
        get_team_stats_cached_db,
    )
    
    print(f"[INFO] Fetching fresh game data for {away_team} @ {home_team}...")
    
    # Initialize with defaults - ensures we always return something
    game_data = {
        'home_team': home_team,
        'away_team': away_team,
        'game_date': game_date,
        'cur_season': cur_season,
        'prev_season': prev_season,
        'fetched_at': datetime.now().isoformat(),
        'home_roster': [],
        'away_roster': [],
        'starters_dict': {},
        'event_id': None,
        'odds_data': {},
        'def_vs_pos_df': [],
        'team_stats': [],
    }
    
    # Fetch rosters with error handling
    print(f"[INFO] Fetching rosters...")
    try:
        home_roster = get_players_by_team(home_team, season=cur_season)
        if home_roster.empty:
            home_roster = get_players_by_team(home_team, season=prev_season)
        if not home_roster.empty:
            home_roster["team_abbrev"] = home_team
            game_data['home_roster'] = home_roster.to_dict('records')
    except Exception as e:
        print(f"[WARN] Failed to fetch home roster: {e}")
    
    try:
        away_roster = get_players_by_team(away_team, season=cur_season)
        if away_roster.empty:
            away_roster = get_players_by_team(away_team, season=prev_season)
        if not away_roster.empty:
            away_roster["team_abbrev"] = away_team
            game_data['away_roster'] = away_roster.to_dict('records')
    except Exception as e:
        print(f"[WARN] Failed to fetch away roster: {e}")
    
    # Fetch starters with error handling
    print(f"[INFO] Fetching starters...")
    try:
        starters_dict = scrape_rotowire_starters()
        if starters_dict:
            game_data['starters_dict'] = starters_dict
    except Exception as e:
        print(f"[WARN] Failed to fetch starters: {e}")
    
    # Fetch event ID and odds with error handling
    print(f"[INFO] Fetching event ID and odds...")
    try:
        event_id = get_event_id_for_game(home_team, away_team)
        game_data['event_id'] = event_id
        
        if event_id:
            odds_data = fetch_fanduel_lines(event_id)
            if odds_data:
                game_data['odds_data'] = odds_data
    except Exception as e:
        print(f"[WARN] Failed to fetch event ID/odds: {e}")
    
    # Fetch shared data with error handling
    print(f"[INFO] Fetching shared data...")
    try:
        def_vs_pos_df = scrape_defense_vs_position_cached_db()
        if isinstance(def_vs_pos_df, pd.DataFrame) and not def_vs_pos_df.empty:
            game_data['def_vs_pos_df'] = def_vs_pos_df.to_dict('records')
    except Exception as e:
        print(f"[WARN] Failed to fetch defense vs position: {e}")
    
    try:
        team_stats = get_team_stats_cached_db(season=prev_season)
        if isinstance(team_stats, pd.DataFrame) and not team_stats.empty:
            game_data['team_stats'] = team_stats.to_dict('records')
    except Exception as e:
        print(f"[WARN] Failed to fetch team stats: {e}")
    
    print(f"[INFO] Game data fetched (some parts may have failed, but continuing)")
    return game_data


def get_cached_game_data(
    home_team: str,
    away_team: str,
    cur_season: str,
    prev_season: str,
    game_date: str = None,
    max_age_hours: int = 24
) -> Optional[Dict[str, Any]]:
    """
    Get cached game data from Firebase, or fetch if missing/stale.
    
    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        cur_season: Current season
        prev_season: Previous season
        game_date: Optional game date
        max_age_hours: Maximum cache age in hours (default 24)
    
    Returns:
        Dictionary containing all game data, or None if fetch fails
    """
    if not FIREBASE_CACHE_AVAILABLE:
        print("[WARN] Firebase cache not available, fetching directly...")
        return fetch_game_data_internal(home_team, away_team, cur_season, prev_season, game_date)
    
    cache_key = get_game_cache_key(home_team, away_team, game_date)
    print(f"[GAME_CACHE] Getting game data for {away_team} @ {home_team} (key: {cache_key})")
    
    # Try to get from cache with short timeout (don't hang if Firebase is slow)
    try:
        cached_data = firebase_cache.get_cached_data(cache_key, max_age_hours, timeout_seconds=5.0)
    except Exception as e:
        print(f"[GAME_CACHE] [ERROR] Cache check failed: {e}. Fetching from API instead...")
        cached_data = None
    
    if cached_data is not None:
        print(f"[GAME_CACHE] [OK] Using CACHED game data for {away_team} @ {home_team}")
        # Convert dict records back to DataFrames if needed
        if isinstance(cached_data, dict):
            cached_data = _restore_dataframes(cached_data)
        # Add cache status indicator
        cached_data['_cache_status'] = 'HIT'
        cached_data['_cache_key'] = cache_key
        return cached_data
    
    # Cache miss or stale - fetch from API
    print(f"[GAME_CACHE] [REFRESH] Cache MISS/STALE for {cache_key}, fetching from API...")
    import time
    import signal
    import threading
    start_time = time.time()
    
    # Add timeout mechanism (2 minutes max)
    MAX_FETCH_TIMEOUT = 120  # 2 minutes
    result_container = {'data': None, 'exception': None, 'completed': False}
    
    def fetch_with_progress():
        try:
            print("[INFO] Starting data fetch...")
            result_container['data'] = fetch_game_data_internal(home_team, away_team, cur_season, prev_season, game_date)
            result_container['completed'] = True
            print(f"[INFO] Data fetch completed in {time.time() - start_time:.1f}s")
        except Exception as e:
            result_container['exception'] = e
            result_container['completed'] = True
            print(f"[ERROR] Data fetch failed: {e}")
    
    # Start fetch in thread
    fetch_thread = threading.Thread(target=fetch_with_progress, daemon=True)
    fetch_thread.start()
    fetch_thread.join(timeout=MAX_FETCH_TIMEOUT)
    
    fetch_time = time.time() - start_time
    
    # Check if thread is still running (timed out)
    if fetch_thread.is_alive():
        print(f"[ERROR] API fetch timed out after {MAX_FETCH_TIMEOUT}s. Returning partial data.")
        # Return minimal data structure
        fresh_data = {
            'home_team': home_team,
            'away_team': away_team,
            'game_date': game_date,
            'cur_season': cur_season,
            'prev_season': prev_season,
            'home_roster': [],
            'away_roster': [],
            'starters_dict': {},
            'event_id': None,
            'odds_data': {},
            'def_vs_pos_df': [],
            'team_stats': [],
            '_timeout': True,
        }
    elif result_container['exception']:
        print(f"[GAME_CACHE] [ERROR] API fetch failed: {result_container['exception']}. Returning empty data structure.")
        fresh_data = {
            'home_team': home_team,
            'away_team': away_team,
            'game_date': game_date,
            'cur_season': cur_season,
            'prev_season': prev_season,
            'home_roster': [],
            'away_roster': [],
            'starters_dict': {},
            'event_id': None,
            'odds_data': {},
            'def_vs_pos_df': [],
            'team_stats': [],
        }
    else:
        fresh_data = result_container['data']
        print(f"[GAME_CACHE] [TIME] API fetch completed in {fetch_time:.2f} seconds")
    
    # Store in cache (if we got data)
    if fresh_data:
        # Convert DataFrames to dict for storage
        serializable_data = _prepare_for_storage(fresh_data)
        cache_saved = firebase_cache.set_cached_data(cache_key, serializable_data)
        if cache_saved:
            print(f"[GAME_CACHE] [SAVE] Saved game data to cache: {cache_key}")
        else:
            print(f"[GAME_CACHE] [WARN] Failed to save game data to cache: {cache_key}")
        # Add cache status indicator
        fresh_data['_cache_status'] = 'MISS'
        fresh_data['_cache_key'] = cache_key
        fresh_data['_fetch_time'] = fetch_time
    
    return fresh_data


def _prepare_for_storage(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert DataFrames to dict format for JSON serialization."""
    storage_data = data.copy()
    
    # Convert DataFrames to dict records
    if 'home_roster' in storage_data and isinstance(storage_data['home_roster'], pd.DataFrame):
        storage_data['home_roster'] = storage_data['home_roster'].to_dict('records')
    if 'away_roster' in storage_data and isinstance(storage_data['away_roster'], pd.DataFrame):
        storage_data['away_roster'] = storage_data['away_roster'].to_dict('records')
    if 'def_vs_pos_df' in storage_data and isinstance(storage_data['def_vs_pos_df'], pd.DataFrame):
        storage_data['def_vs_pos_df'] = storage_data['def_vs_pos_df'].to_dict('records')
    if 'team_stats' in storage_data and isinstance(storage_data['team_stats'], pd.DataFrame):
        storage_data['team_stats'] = storage_data['team_stats'].to_dict('records')
    
    return storage_data


def _restore_dataframes(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dict records back to DataFrames."""
    restored_data = data.copy()
    
    # Convert dict records back to DataFrames
    if 'home_roster' in restored_data and isinstance(restored_data['home_roster'], list):
        restored_data['home_roster'] = pd.DataFrame(restored_data['home_roster']) if restored_data['home_roster'] else pd.DataFrame()
    if 'away_roster' in restored_data and isinstance(restored_data['away_roster'], list):
        restored_data['away_roster'] = pd.DataFrame(restored_data['away_roster']) if restored_data['away_roster'] else pd.DataFrame()
    if 'def_vs_pos_df' in restored_data and isinstance(restored_data['def_vs_pos_df'], list):
        restored_data['def_vs_pos_df'] = pd.DataFrame(restored_data['def_vs_pos_df']) if restored_data['def_vs_pos_df'] else pd.DataFrame()
    if 'team_stats' in restored_data and isinstance(restored_data['team_stats'], list):
        restored_data['team_stats'] = pd.DataFrame(restored_data['team_stats']) if restored_data['team_stats'] else pd.DataFrame()
    
    return restored_data

