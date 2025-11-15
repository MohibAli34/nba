import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from . import database
from .data_fetcher import (
    get_player_game_logs_cached,
    get_team_stats_cached,
    scrape_defense_vs_position,
    get_players_by_team,
    get_opponent_recent_games,
    get_head_to_head_history,
)

############################################################
# Internal helpers
############################################################

def _save_player_game_logs_compat(player_id, player_name, season, logs_df):
    """
    Safely call database.save_player_game_logs(...) no matter which
    signature your local database.py is using.

    Some versions are:
        save_player_game_logs(player_id, player_name, season, game_logs_df)
    Others are:
        save_player_game_logs(player_id, season, game_logs_df)
    """
    try:
        # Try the 4-arg version first (with player_name)
        database.save_player_game_logs(
            player_id=player_id,
            player_name=player_name,
            season=season,
            game_logs_df=logs_df
        )
    except TypeError:
        # Fall back to 3-arg version (no player_name kw)
        database.save_player_game_logs(
            player_id,
            season,
            logs_df
        )


def _normalize_team_stats_for_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the raw team stats df (from nba_api or DB) and normalize columns
    to what the rest of the app / features builder expects.

    Output columns:
        TEAM_ABBREVIATION
        DEF_RATING
        PACE
        PTS_ALLOWED  (derived from OPP_PTS in DB or from PTS_ALLOWED/PTS fallback)
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(
            columns=["TEAM_ABBREVIATION", "DEF_RATING", "PACE", "PTS_ALLOWED"]
        )

    df = raw_df.copy()

    # Make sure TEAM_ABBREVIATION exists
    if "TEAM_ABBREVIATION" not in df.columns:
        # sometimes we get TEAM_ABBREVIATION_x from merges, handle that as fallback
        for alt in ["TEAM_ABBREVIATION_x", "TEAM_ABBREVIATION_y", "TEAM"]:
            if alt in df.columns:
                df["TEAM_ABBREVIATION"] = df[alt]
                break

    # PTS_ALLOWED:
    # DB may save this as OPP_PTS.
    if "PTS_ALLOWED" not in df.columns:
        if "OPP_PTS" in df.columns:
            df["PTS_ALLOWED"] = df["OPP_PTS"]
        elif "PTS" in df.columns:
            # worst-case fallback: treat PTS (their own scoring) as allowed
            df["PTS_ALLOWED"] = df["PTS"]
        else:
            df["PTS_ALLOWED"] = 110.0

    # Ensure required columns exist so downstream code doesn't explode
    needed = ["TEAM_ABBREVIATION", "DEF_RATING", "PACE", "PTS_ALLOWED"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    df = df[needed].copy()
    df = df.dropna(subset=["TEAM_ABBREVIATION"]).reset_index(drop=True)
    return df


############################################################
# Public: Player game logs with SQLite caching
############################################################

def get_player_game_logs_cached_db(player_id, player_name, season):
    """
    Main entry point the app uses.

    1. Try to load this player's logs for this season from SQLite cache.
    2. If cache miss / empty:
        - fetch via nba_api (get_player_game_logs_cached)
        - save into SQLite for future calls
    3. Return a clean DataFrame in the same format everywhere else expects
       (GAME_DATE, MATCHUP, PTS, REB, AST, FG3M, MIN, etc.)
    """
    # Step 1: try DB cache
    cached_df = database.get_cached_player_game_logs(player_id, season)
    if cached_df is not None and not cached_df.empty:
        return cached_df

    # Step 2: fetch live via nba_api
    live_df = get_player_game_logs_cached(player_id, season=season)

    if live_df is None or live_df.empty:
        # Nothing to save or return
        return pd.DataFrame()

    # Save to DB (compat wrapper handles signature differences)
    _save_player_game_logs_compat(
        player_id=player_id,
        player_name=player_name,
        season=season,
        logs_df=live_df
    )

    # Return what we just saved, but normalized exactly as DB returns
    cached_df_after = database.get_cached_player_game_logs(player_id, season)
    if cached_df_after is not None and not cached_df_after.empty:
        return cached_df_after

    # Fallback if for some reason DB read fails
    # We still try to mimic DB's output shape
    df = live_df.copy()

    # normalize GAME_DATE column etc. so downstream code won't break
    if 'GAME_DATE' not in df.columns and 'GAME_DATE' in df.columns:
        pass  # already good
    # nba_api returns 'GAME_DATE' already, so usually fine

    return df


############################################################
# Public: Team stats with SQLite caching
############################################################

def get_team_stats_cached_db(season: str) -> pd.DataFrame:
    """
    1. Try pulling all team stats for this season from SQLite.
    2. If stale/missing:
        - fetch from nba_api via get_team_stats_cached()
        - save each team into SQLite with database.save_team_stats()
    3. Return normalized df that features.py expects.
    """

    # 1. Try DB
    cached_stats_df = database.get_cached_team_stats(season)
    if cached_stats_df is not None and not cached_stats_df.empty:
        return _normalize_team_stats_for_features(cached_stats_df)

    # 2. Fetch live from nba_api
    api_df = get_team_stats_cached(season=season)
    if api_df is None or api_df.empty:
        # Can't get anything, return empty frame with correct schema
        return pd.DataFrame(
            columns=["TEAM_ABBREVIATION", "DEF_RATING", "PACE", "PTS_ALLOWED"]
        )

    # Normalize the live data into the shape we expect
    norm_df = _normalize_team_stats_for_features(api_df)

    # Save each team row to DB for future calls
    for _, row in norm_df.iterrows():
        team_abbrev = row["TEAM_ABBREVIATION"]

        # Build a mini 1-row df with columns the DB layer expects.
        # database.save_team_stats() in database.py expects columns:
        #   DEF_RATING, OFF_RATING, PACE, OPP_PTS
        # We don't really have OFF_RATING easily here, DB code will default.
        # We'll map PTS_ALLOWED -> OPP_PTS for DB.
        one_team_df = pd.DataFrame([{
            "TEAM_ABBREVIATION": team_abbrev,
            "DEF_RATING": row.get("DEF_RATING", 110.0),
            "PACE": row.get("PACE", 100.0),
            "OPP_PTS": row.get("PTS_ALLOWED", 110.0),
        }])

        # CRITICAL: pass stats_df=..., not stats=...
        database.save_team_stats(
            team_abbrev=team_abbrev,
            season=season,
            stats_df=one_team_df
        )

    # Re-read from DB so we have a consistent source going forward
    cached_stats_df = database.get_cached_team_stats(season)
    if cached_stats_df is not None and not cached_stats_df.empty:
        return _normalize_team_stats_for_features(cached_stats_df)

    # Fallback (should basically never hit)
    return norm_df


############################################################
# Public: Defense vs position with SQLite caching
############################################################

def scrape_defense_vs_position_cached_db():
    """
    Returns the defense-vs-position DataFrame (Position, Team, Rank, etc.)
    Uses SQLite as a cache:
      - Try DB first
      - If stale/missing, scrape HashtagBasketball, then save to DB
    """
    cached_df = database.get_cached_defense_vs_position()
    if cached_df is not None and not cached_df.empty:
        return cached_df

    # Cache miss -> scrape live
    fresh_df = scrape_defense_vs_position()
    if fresh_df is None or fresh_df.empty:
        # no data at all
        return pd.DataFrame()

    # Save to DB for reuse
    database.save_defense_vs_position(fresh_df)

    # Return fresh scraped
    return fresh_df


############################################################
# Public: Preload helper (optional optimization)
############################################################

def preload_game_data(home_team, away_team, season=None):
    """
    Warm up cache for an upcoming game:
      - rosters
      - each player's logs (current + prior season)
      - opponent recent games
      - team stats
      - defense vs position

    You can call this right after the user selects a game so that by the
    time we render the board, most stuff is hot in cache.

    season: optional override for "current season". If None we infer.
    """
    # We infer NBA seasons the same way app.py does
    def _get_current_nba_season():
        now = datetime.now()
        year = now.year
        month = now.month
        if month >= 10:
            return f"{year}-{str(year + 1)[2:]}"
        else:
            return f"{year - 1}-{str(year)[2:]}"

    def _get_prior_nba_season():
        cur = _get_current_nba_season()
        start_year = int(cur.split('-')[0])
        return f"{start_year - 1}-{str(start_year)[2:]}"

    current_season = season or _get_current_nba_season()
    prior_season = _get_prior_nba_season()

    # 1. cache team stats / defense vs position
    _ = get_team_stats_cached_db(prior_season)
    _ = scrape_defense_vs_position_cached_db()

    # 2. get rosters and pull logs for each player up front
    rosters = {
        home_team: get_players_by_team(home_team, season=current_season),
        away_team: get_players_by_team(away_team, season=current_season),
    }
    for team_abbrev in [home_team, away_team]:
        if rosters[team_abbrev] is None or rosters[team_abbrev].empty:
            rosters[team_abbrev] = get_players_by_team(team_abbrev, season=prior_season)

    for team_abbrev, roster_df in rosters.items():
        if roster_df is None or roster_df.empty:
            continue

        for _, row in roster_df.iterrows():
            player_name = row['full_name']
            pid = row['player_id']

            # warm current/prior logs into DB
            _ = get_player_game_logs_cached_db(pid, player_name, current_season)
            _ = get_player_game_logs_cached_db(pid, player_name, prior_season)

            # warm h2h / opponent recents lightly (not saving in DB here, but we
            # hit nba_api so caching via @st.cache_data happens upstream)
            # pick the other team as opponent:
            opponent = away_team if team_abbrev == home_team else home_team
            _ = get_opponent_recent_games(opponent, season=prior_season, last_n=10)
            _ = get_head_to_head_history(
                pid,
                opponent,
                seasons=[prior_season, '2023-24']
            )

    # Not returning anything; this is purely a warm-up hook.
    return True
