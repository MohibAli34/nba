import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashteamstats,
    teamgamelog,
    scoreboardv2,
    leaguegamefinder,
)
from nba_api.live.nba.endpoints import scoreboard

# 🔁 local sqlite cache helper (your database.py module)
from . import database


# -------------------------------------------------
# Rate limiting helper for nba_api calls
# -------------------------------------------------
def rate_limit():
    time.sleep(0.6)


# -------------------------------------------------
# Basic player / team lookup utilities
# -------------------------------------------------
@st.cache_data(ttl=86400)
def get_all_active_players():
    """Get all active NBA players."""
    all_players = players.get_players()
    return pd.DataFrame(all_players)


def get_player_id(player_name):
    """Get player ID from name."""
    result = players.find_players_by_full_name(player_name)
    if result:
        return result[0]["id"]
    return None


# -------------------------------------------------
# Game logs (direct from nba_api; *not* the sqlite cache version)
# -------------------------------------------------
@st.cache_data(ttl=3600)
def get_player_game_logs_cached(player_id, season="2024-25"):
    """
    Get player game logs for a season via nba_api.
    This does NOT read/write sqlite. cached_data_fetcher wraps this.
    """
    try:
        rate_limit()
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season",
        )
        df = gamelog.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error fetching game logs for player {player_id} in {season}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_team_stats_cached(season="2024-25"):
    """
    Get team per-game metrics from nba_api.
    cached_data_fetcher.get_team_stats_cached_db() will save this to sqlite.
    """
    try:
        rate_limit()
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
        )
        df = team_stats.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error fetching team stats for {season}: {e}")
        return pd.DataFrame()


# -------------------------------------------------
# Opponent / matchup context
# -------------------------------------------------
@st.cache_data(ttl=3600)
def get_opponent_recent_games(opponent_abbrev, season="2024-25", last_n=10):
    """Get opponent's recent games for defensive trend stuff."""
    try:
        all_teams = teams.get_teams()
        team_info = [t for t in all_teams if t["abbreviation"] == opponent_abbrev]

        if not team_info:
            return pd.DataFrame()

        team_id = team_info[0]["id"]

        rate_limit()
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star="Regular Season",
        )
        df = gamelog.get_data_frames()[0]

        return df.head(last_n)
    except Exception as e:
        print(f"Error fetching opponent recent games: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_head_to_head_history(
    player_id, opponent_abbrev, seasons=["2024-25", "2023-24"]
):
    """
    Get player's historical game logs vs one opponent across multiple seasons.
    """
    h2h_games = []

    for season in seasons:
        try:
            rate_limit()
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star="Regular Season",
            )
            df = gamelog.get_data_frames()[0]

            if not df.empty and "MATCHUP" in df.columns:
                opponent_games = df[df["MATCHUP"].str.contains(opponent_abbrev, na=False)]
                if not opponent_games.empty:
                    h2h_games.append(opponent_games)
        except Exception as e:
            print(f"Error fetching h2h for {season}: {e}")
            continue

    if h2h_games:
        return pd.concat(h2h_games, ignore_index=True)

    return pd.DataFrame()


# -------------------------------------------------
# Odds / FanDuel (you said we keep this OFF for now, but leaving interface)
# -------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_fanduel_lines(event_id, api_key="f6aac04a6ab847bab31a7db076ef89e8"):
    """
    Fetch player props for a specific NBA event from The Odds API.
    Returns dict[player_name][stat_code] = {'line': float, 'over_price': ..., 'under_price': ...}
    """
    if not event_id or not api_key:
        return {}

    try:
        url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "bookmakers": "fanduel",
            "markets": "player_points,player_assists,player_rebounds,player_threes,player_points_rebounds_assists",
            "oddsFormat": "american",
        }

        print(f"🔍 Fetching player props for event {event_id}...")
        response = requests.get(url, params=params, timeout=15)

        if response.status_code != 200:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return {}

        data = response.json()
        print("✅ Got odds response")

        player_props = {}

        for bookmaker in data.get("bookmakers", []):
            if bookmaker.get("key") != "fanduel":
                continue

            print("📊 Found FanDuel odds")

            for market in bookmaker.get("markets", []):
                market_key = market.get("key")

                stat_map = {
                    "player_points": "PTS",
                    "player_assists": "AST",
                    "player_rebounds": "REB",
                    "player_threes": "FG3M",
                    "player_points_rebounds_assists": "PRA",
                }

                stat_code = stat_map.get(market_key)
                if not stat_code:
                    continue

                print(
                    f"   Market: {market_key} ({len(market.get('outcomes', []))} players)"
                )

                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description", "")
                    line = outcome.get("point")
                    price = outcome.get("price")
                    outcome_type = outcome.get("name", "")  # "Over" or "Under"

                    if player_name and line is not None:
                        if player_name not in player_props:
                            player_props[player_name] = {}

                        if stat_code not in player_props[player_name]:
                            player_props[player_name][stat_code] = {
                                "line": float(line),
                                "over_price": None,
                                "under_price": None,
                            }

                        # stash both Over/Under prices
                        if outcome_type == "Over":
                            player_props[player_name][stat_code]["over_price"] = price
                        elif outcome_type == "Under":
                            player_props[player_name][stat_code]["under_price"] = price

        print(f"✅ Parsed props for {len(player_props)} players")
        return player_props

    except Exception as e:
        print(f"❌ Error fetching FanDuel lines: {e}")
        import traceback

        traceback.print_exc()
        return {}


@st.cache_data(ttl=3600)
def get_event_id_for_game(
    home_team, away_team, api_key="f6aac04a6ab847bab31a7db076ef89e8"
):
    """
    Look up The Odds API event ID for a given NBA matchup (away @ home).
    """
    try:
        url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
        params = {"apiKey": api_key}

        print(f"🔍 Looking for event: {away_team} @ {home_team}")
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            print(f"❌ Events API error: {response.status_code}")
            return None

        events = response.json()

        # Abbrev → full names (Odds API usually uses full team names)
        team_name_map = {
            "LAL": "Los Angeles Lakers",
            "MIN": "Minnesota Timberwolves",
            "BOS": "Boston Celtics",
            "NYK": "New York Knicks",
            "BKN": "Brooklyn Nets",
            "PHI": "Philadelphia 76ers",
            "MIA": "Miami Heat",
            "MIL": "Milwaukee Bucks",
            "GSW": "Golden State Warriors",
            "LAC": "LA Clippers",
            "CHI": "Chicago Bulls",
            "CLE": "Cleveland Cavaliers",
            "DET": "Detroit Pistons",
            "IND": "Indiana Pacers",
            "ATL": "Atlanta Hawks",
            "CHA": "Charlotte Hornets",
            "WAS": "Washington Wizards",
            "ORL": "Orlando Magic",
            "TOR": "Toronto Raptors",
            "SAC": "Sacramento Kings",
            "PHX": "Phoenix Suns",
            "POR": "Portland Trail Blazers",
            "DEN": "Denver Nuggets",
            "UTA": "Utah Jazz",
            "OKC": "Oklahoma City Thunder",
            "DAL": "Dallas Mavericks",
            "HOU": "Houston Rockets",
            "SAS": "San Antonio Spurs",
            "MEM": "Memphis Grizzlies",
            "NOP": "New Orleans Pelicans",
        }

        home_full = team_name_map.get(home_team, home_team)
        away_full = team_name_map.get(away_team, away_team)

        for event in events:
            event_home = event.get("home_team", "")
            event_away = event.get("away_team", "")

            # crude match
            if (home_full in event_home) and (away_full in event_away):
                event_id = event.get("id")
                print(f"✅ Found event ID: {event_id}")
                return event_id

        print(f"⚠️ No event found for {away_team} @ {home_team}")
        return None

    except Exception as e:
        print(f"❌ Error getting event ID: {e}")
        return None


def get_player_fanduel_line(player_name, stat, odds_data):
    """
    Convenience helper to look up 1 prop line inside the big odds blob.
    stat is one of: PTS, REB, AST, FG3M, PRA
    """
    if not odds_data or player_name not in odds_data:
        return None

    pdata = odds_data[player_name]
    if stat not in pdata:
        return None

    return pdata[stat]


# -------------------------------------------------
# Schedule / upcoming games
# -------------------------------------------------
@st.cache_data(ttl=1800)
def get_todays_games():
    """Get today's NBA games via live scoreboard snapshot."""
    try:
        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()

        games = []
        today = datetime.now()

        if "scoreboard" in games_data and "games" in games_data["scoreboard"]:
            for game in games_data["scoreboard"]["games"]:
                game_status = game.get("gameStatus", 1)

                game_date_display = today.strftime("%a, %b %d")
                game_time_utc = game.get("gameTimeUTC", "")
                if game_time_utc:
                    try:
                        game_dt = datetime.fromisoformat(
                            game_time_utc.replace("Z", "+00:00")
                        )
                        game_time_display = game_dt.strftime("%I:%M %p")
                    except:
                        game_time_display = ""
                else:
                    game_time_display = ""

                games.append(
                    {
                        "home": game.get("homeTeam", {}).get("teamTricode", ""),
                        "away": game.get("awayTeam", {}).get("teamTricode", ""),
                        "date": today.strftime("%Y-%m-%d"),
                        "date_display": game_date_display,
                        "time_display": game_time_display,
                        "status": game_status,
                    }
                )

        return games
    except Exception as e:
        print(f"Error fetching today's games: {e}")
        return []


@st.cache_data(ttl=900)
def get_upcoming_games(days: int = 7):
    """
    Get upcoming NBA games (next 3 days: today, tomorrow, day after tomorrow) from the NBA schedule JSON dump.
    Falls back to get_todays_games() if that fails.
    """
    print("\n=== Fetching upcoming games from NBA schedule JSON ===")
    try:
        schedule_url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
        response = requests.get(schedule_url, timeout=10)

        if response.status_code != 200:
            print(f"  Failed to fetch schedule: HTTP {response.status_code}")
            return get_todays_games()

        schedule_data = response.json()
        league_schedule = schedule_data.get("leagueSchedule", {})
        game_dates = league_schedule.get("gameDates", [])

        if not game_dates:
            print("  No game dates found in schedule")
            return get_todays_games()

        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        day_after_tomorrow = today + timedelta(days=2)
        
        # Only include games on these 3 dates
        allowed_dates = [today, tomorrow, day_after_tomorrow]

        upcoming = []

        for game_date_obj in game_dates:
            game_date_str = game_date_obj.get("gameDate", "")
            if not game_date_str:
                continue

            try:
                game_date = datetime.strptime(game_date_str, "%m/%d/%Y %H:%M:%S")
            except:
                continue

            # Filter: only include games on today, tomorrow, or day after tomorrow
            if game_date.date() not in allowed_dates:
                continue

            games = game_date_obj.get("games", [])
            for game in games:
                away_team = game.get("awayTeam", {})
                home_team = game.get("homeTeam", {})

                away_abbrev = away_team.get("teamTricode", "")
                home_abbrev = home_team.get("teamTricode", "")

                if not away_abbrev or not home_abbrev:
                    continue

                # try home/awayTeamTime (UTC-ish) and render local-ish string
                game_time_str = game.get("awayTeamTime", "") or game.get(
                    "homeTeamTime", ""
                )
                time_display = ""
                if game_time_str:
                    try:
                        game_dt = datetime.fromisoformat(
                            game_time_str.replace("Z", "+00:00")
                        )
                        time_display = game_dt.strftime("%I:%M %p")
                    except:
                        pass

                game_date_display = game_date.strftime("%a, %b %d")

                upcoming.append(
                    {
                        "home": home_abbrev,
                        "away": away_abbrev,
                        "date": game_date.strftime("%Y-%m-%d"),
                        "date_display": game_date_display,
                        "time_display": time_display,
                        "status": "Scheduled",
                    }
                )

        print(f"  Found {len(upcoming)} upcoming games")

        if not upcoming:
            print("  No upcoming games found, falling back to live scoreboard")
            return get_todays_games()

        return upcoming

    except Exception as e:
        print(f"Error fetching upcoming games: {e}")
        import traceback

        traceback.print_exc()
        return get_todays_games()


def get_all_nba_teams():
    """Return list of all NBA team abbreviations."""
    all_teams = teams.get_teams()
    return sorted([team["abbreviation"] for team in all_teams])


@st.cache_data(ttl=3600)
def get_player_current_team(player_id, season="2025-26"):
    """
    Infer player's current/most recent team abbrev by looking at their most recent game.
    """
    try:
        rate_limit()
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season",
        )
        df = gamelog.get_data_frames()[0]

        if not df.empty and "MATCHUP" in df.columns:
            matchup = df.iloc[0]["MATCHUP"]
            if " vs. " in matchup:
                return matchup.split(" vs. ")[0]
            elif " @ " in matchup:
                return matchup.split(" @ ")[0]
        return None
    except Exception as e:
        print(f"Error fetching player team: {e}")
        return None


# -------------------------------------------------
# 🆕 Position resolver with persistent sqlite caching
# -------------------------------------------------
@st.cache_data(ttl=3600)
def get_player_position(player_id, season="2024-25"):
    """
    Return player's primary position ('G', 'F', or 'C'), but:
    - First check sqlite player_metadata for a saved position.
    - If missing, resolve it using nba_api (roster, commonplayerinfo, static data),
      THEN save it to sqlite so future runs (even after restart) are instant.

    This is the piece that makes repeated matchups fast.
    """

    def simplify_position(pos_string):
        """Convert any raw/multi-position string to 'G', 'F', or 'C'."""
        if not pos_string or str(pos_string).strip() == "" or str(pos_string) == "nan":
            return None

        pos_string = str(pos_string).strip().upper()
        print(f"🔍 Parsing position: '{pos_string}'")

        # Handle multi-position like "F-C", "C-F", "G-F", etc.
        if "-" in pos_string:
            parts = [p.strip() for p in pos_string.split("-")]
            first_pos = parts[0]

            print(f"   → Multi-position: {pos_string}")
            print(f"   → Using PRIMARY (first): {first_pos}")

            if "G" in first_pos:  # G, PG, SG
                print(f"   → Primary is Guard: returning G")
                return "G"
            elif "C" in first_pos:  # C
                print(f"   → Primary is Center: returning C")
                return "C"
            else:  # F, PF, SF
                print(f"   → Primary is Forward: returning F")
                return "F"

        # Single position fallbacks
        if "G" in pos_string:  # G, PG, SG
            print(f"   → Guard: returning G")
            return "G"
        elif "C" in pos_string:  # C
            print(f"   → Center: returning C")
            return "C"
        else:  # F, PF, SF
            print(f"   → Forward: returning F")
            return "F"

    print(f"\n🏀 Getting position for player_id: {player_id}, season: {season}")

    # 1. FAST PATH: check sqlite metadata first
    try:
        meta = database.get_player_metadata(player_id)
        if meta and meta.get("position"):
            cached_simplified = simplify_position(meta["position"])
            if cached_simplified:
                print(f"   ✅ (DB cached) Position: {cached_simplified}\n")
                return cached_simplified
    except Exception as e:
        print(f"   ⚠️ DB metadata lookup failed: {e}")

    # If not cached, we'll determine it the slow way, then write it back.
    team_abbrev_for_db = None
    final_position_letter = None
    player_name_for_db = None

    try:
        # METHOD 1: infer team from recent game, then pull that team's roster
        try:
            from nba_api.stats.endpoints import commonteamroster

            rate_limit()
            gamelog_resp = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star="Regular Season",
            )
            df_log = gamelog_resp.get_data_frames()[0]

            if not df_log.empty and "MATCHUP" in df_log.columns:
                matchup = df_log.iloc[0]["MATCHUP"]
                if " vs. " in matchup:
                    team_abbrev_for_db = matchup.split(" vs. ")[0]
                elif " @ " in matchup:
                    team_abbrev_for_db = matchup.split(" @ ")[0]

                print(f"   Team found: {team_abbrev_for_db}")

                # find that team's NBA ID
                all_teams = teams.get_teams()
                tinfo = [
                    t for t in all_teams if t["abbreviation"] == team_abbrev_for_db
                ]
                if tinfo:
                    team_id = tinfo[0]["id"]

                    # get season roster
                    rate_limit()
                    roster = commonteamroster.CommonTeamRoster(
                        team_id=team_id, season=season
                    )
                    roster_df = roster.get_data_frames()[0]

                    if not roster_df.empty:
                        row_match = roster_df[roster_df["PLAYER_ID"] == player_id]
                        if not row_match.empty:
                            raw_pos = row_match.iloc[0].get("POSITION", "")
                            print(f"   📋 Roster says: '{raw_pos}'")

                            final_position_letter = simplify_position(raw_pos)

                            # record player name for DB too
                            player_name_for_db = row_match.iloc[0].get(
                                "PLAYER", None
                            )

        except Exception as e:
            print(f"   ⚠️ Roster method failed: {e}")

        # METHOD 2: fallback to CommonPlayerInfo
        if not final_position_letter:
            try:
                from nba_api.stats.endpoints import commonplayerinfo

                rate_limit()
                info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
                info_df = info.get_data_frames()[0]

                if not info_df.empty and "POSITION" in info_df.columns:
                    raw_pos = info_df.iloc[0]["POSITION"]
                    print(f"   📊 API says: '{raw_pos}'")
                    final_position_letter = simplify_position(raw_pos)

                if (
                    not player_name_for_db
                    and not info_df.empty
                    and "DISPLAY_FIRST_LAST" in info_df.columns
                ):
                    player_name_for_db = info_df.iloc[0]["DISPLAY_FIRST_LAST"]

            except Exception as e:
                print(f"   ⚠️ API method failed: {e}")

        # METHOD 3: final fallback - static info from nba_api.stats.static.players
        if not final_position_letter:
            try:
                pdata = players.find_player_by_id(player_id)
                if pdata and "position" in pdata:
                    raw_pos = pdata.get("position", "")
                    if raw_pos:
                        print(f"   📁 Static says: '{raw_pos}'")
                        final_position_letter = simplify_position(raw_pos)

                if not player_name_for_db and pdata and "full_name" in pdata:
                    player_name_for_db = pdata["full_name"]

            except Exception as e:
                print(f"   ⚠️ Static method failed: {e}")

        # If STILL nothing, default to Forward
        if not final_position_letter:
            print("   ⚠️ All methods failed - defaulting to F\n")
            final_position_letter = "F"

        # 2. persist to sqlite so future runs (even after restart) are instant
        try:
            if not player_name_for_db:
                # best-effort fallback name
                pdata = players.find_player_by_id(player_id)
                if pdata and "full_name" in pdata:
                    player_name_for_db = pdata["full_name"]

            database.save_player_metadata(
                player_id=player_id,
                player_name=player_name_for_db or f"Player {player_id}",
                position=final_position_letter,
                team=team_abbrev_for_db or "",
            )
        except Exception as e:
            print(f"   ⚠️ could not save player_metadata: {e}")

        print(f"   ✅ Position: {final_position_letter}\n")
        return final_position_letter

    except Exception as outer_e:
        print(f"   ❌ ERROR determining position: {outer_e}")
        # last-resort safety
        return "F"


# -------------------------------------------------
# Next game info from live scoreboard
# -------------------------------------------------
@st.cache_data(ttl=1800)
def get_team_next_game(team_abbrev):
    """
    Get a team's next upcoming game using today's live scoreboard snapshot.
    """
    try:
        time.sleep(0.5)

        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()

        if "scoreboard" in games_data and "games" in games_data["scoreboard"]:
            games = games_data["scoreboard"]["games"]

            for game in games:
                home_team = game.get("homeTeam", {}).get("teamTricode", "")
                away_team = game.get("awayTeam", {}).get("teamTricode", "")

                if team_abbrev == home_team:
                    return {
                        "opponent": away_team,
                        "is_home": True,
                        "matchup_string": f"{away_team} @ {home_team}",
                        "found": True,
                    }
                elif team_abbrev == away_team:
                    return {
                        "opponent": home_team,
                        "is_home": False,
                        "matchup_string": f"{away_team} @ {home_team}",
                        "found": True,
                    }

        return None

    except Exception as e:
        print(f"Error fetching next game: {e}")
        return None


# -------------------------------------------------
# Defense vs Position scrape (HashtagBasketball)
# -------------------------------------------------
@st.cache_data(ttl=86400)
def scrape_defense_vs_position():
    """
    Scrape defensive rankings vs position from HashtagBasketball.
    Returns DataFrame with Position, Team, Rank, PTS, FG_PCT, FT_PCT, TPM, REB, AST, STL, BLK, TO
    (We also compute our own Rank per position.)
    """
    try:
        url = "https://hashtagbasketball.com/nba-defense-vs-position"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        time.sleep(1.0)

        print("Fetching defense vs position data...")
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"Failed to fetch defense vs position: {response.status_code}")
            return pd.DataFrame()

        tables = pd.read_html(response.content)

        if len(tables) < 4:
            print("Expected table not found")
            return pd.DataFrame()

        raw_df = tables[3].copy()
        df_clean = pd.DataFrame()

        # Col 0: Position
        df_clean["Position"] = raw_df.iloc[:, 0]

        # Col 1: "<TEAM> <rank>" → split first token = team code
        team_rank = raw_df.iloc[:, 1].astype(str).str.split(expand=True)
        df_clean["Team"] = team_rank[0]

        # Parse each stats col ("<value> <rank>")
        pts_split = raw_df.iloc[:, 2].astype(str).str.split(expand=True)
        df_clean["PTS"] = pts_split[0].astype(float)

        fg_pct_split = raw_df.iloc[:, 3].astype(str).str.split(expand=True)
        df_clean["FG_PCT"] = fg_pct_split[0].astype(float)

        ft_pct_split = raw_df.iloc[:, 4].astype(str).str.split(expand=True)
        df_clean["FT_PCT"] = ft_pct_split[0].astype(float)

        tpm_split = raw_df.iloc[:, 5].astype(str).str.split(expand=True)
        df_clean["TPM"] = tpm_split[0].astype(float)

        reb_split = raw_df.iloc[:, 6].astype(str).str.split(expand=True)
        df_clean["REB"] = reb_split[0].astype(float)

        ast_split = raw_df.iloc[:, 7].astype(str).str.split(expand=True)
        df_clean["AST"] = ast_split[0].astype(float)

        stl_split = raw_df.iloc[:, 8].astype(str).str.split(expand=True)
        df_clean["STL"] = stl_split[0].astype(float)

        blk_split = raw_df.iloc[:, 9].astype(str).str.split(expand=True)
        df_clean["BLK"] = blk_split[0].astype(float)

        to_split = raw_df.iloc[:, 10].astype(str).str.split(expand=True)
        df_clean["TO"] = to_split[0].astype(float)

        # Composite score → rank within each Position bucket
        for pos in df_clean["Position"].unique():
            pos_rows = df_clean[df_clean["Position"] == pos]

            composite_scores = (
                pos_rows["PTS"] * 1.0  # more points allowed = softer
                + pos_rows["FG_PCT"] * 0.3
                + pos_rows["TPM"] * 0.2
                - pos_rows["STL"] * 0.2  # lots of steals = tougher (subtract)
                - pos_rows["BLK"] * 0.15  # lots of blocks = tougher (subtract)
            )

            df_clean.loc[df_clean["Position"] == pos, "Rank"] = (
                composite_scores.rank(method="min").astype(int)
            )

        print(
            f"Successfully scraped {len(df_clean)} team-position combos with full defensive stats"
        )
        return df_clean

    except Exception as e:
        print(f"Error scraping defense vs position: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()


# -------------------------------------------------
# Starter scraping from Rotowire
# -------------------------------------------------
@st.cache_data(ttl=1800)
def scrape_rotowire_starters():
    """
    Scrape projected starters from Rotowire NBA lineups page.
    Returns a dict mapping player names (normalized) to is_starter (bool).
    """
    try:
        url = "https://www.rotowire.com/basketball/nba-lineups.php"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        print("🔍 Scraping Rotowire for projected starters...")
        time.sleep(1.0)  # Be respectful with rate limiting
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch Rotowire: HTTP {response.status_code}")
            return {}
        
        soup = BeautifulSoup(response.text, "html.parser")
        starters_dict = {}
        
        # Rotowire structure: lineup sections with .lineup.is-nba class
        lineup_sections = soup.select(".lineup.is-nba, .lineup")
        
        for section in lineup_sections:
            # Check if this is a starter section (usually first 5 players in lineup)
            # Rotowire typically shows starters first, then bench
            players = section.select(".lineup__player")
            
            # Look for player links/names
            for idx, player_elem in enumerate(players):
                # Try to find player name in <a> tag or direct text
                player_link = player_elem.find("a")
                if player_link:
                    name = player_link.get_text(strip=True)
                else:
                    name = player_elem.get_text(strip=True)
                
                if not name or len(name) < 2:
                    continue
                
                # First 5 players in a lineup are typically starters
                # Also check for explicit starter indicators
                parent_classes = " ".join(player_elem.get("class", [])).lower()
                section_classes = " ".join(section.get("class", [])).lower()
                
                is_starter = (
                    idx < 5 or  # First 5 players
                    "starter" in parent_classes or
                    "starter" in section_classes or
                    "starting" in parent_classes or
                    "starting" in section_classes
                )
                
                name_normalized = " ".join(name.split())
                # Only set if not already set (starters take precedence)
                if name_normalized not in starters_dict or is_starter:
                    starters_dict[name_normalized] = is_starter
        
        # Fallback: look for any lineup__player elements
        if not starters_dict:
            all_players = soup.select(".lineup__player")
            for player_elem in all_players[:50]:  # Limit to first 50 to avoid noise
                player_link = player_elem.find("a")
                if player_link:
                    name = player_link.get_text(strip=True)
                else:
                    name = player_elem.get_text(strip=True)
                
                if name and len(name) > 2:
                    name_normalized = " ".join(name.split())
                    # Default to starter if we can't determine (conservative approach)
                    starters_dict[name_normalized] = True
        
        print(f"✅ Found {sum(1 for v in starters_dict.values() if v)} starters out of {len(starters_dict)} players")
        return starters_dict
        
    except Exception as e:
        print(f"❌ Error scraping Rotowire starters: {e}")
        import traceback
        traceback.print_exc()
        return {}


def normalize_player_name(name):
    """
    Normalize player name for matching (remove extra spaces, handle common variations).
    """
    if not name:
        return ""
    # Remove extra whitespace
    normalized = " ".join(str(name).split())
    return normalized.strip()


def is_player_starter(player_name, starters_dict):
    """
    Check if a player is a starter based on Rotowire data.
    Returns True if starter, False if bench, None if unknown.
    """
    if not starters_dict:
        return None
    
    normalized = normalize_player_name(player_name)
    
    # Direct match
    if normalized in starters_dict:
        return starters_dict[normalized]
    
    # Try partial matching (first name + last name variations)
    name_parts = normalized.split()
    if len(name_parts) >= 2:
        # Try "First Last" and "Last, First" formats
        first_last = f"{name_parts[0]} {name_parts[-1]}"
        last_first = f"{name_parts[-1]}, {name_parts[0]}"
        
        for key in starters_dict:
            key_normalized = normalize_player_name(key)
            if first_last in key_normalized or key_normalized in first_last:
                return starters_dict[key]
            if last_first in key_normalized or key_normalized in last_first:
                return starters_dict[key]
    
    return None


# -------------------------------------------------
# Roster / per-team players helper
# -------------------------------------------------
def get_players_by_team(team_abbrev, season="2024-25"):
    """
    Get all players who have played for a specific team in a season.
    Returns DataFrame with: player_id, full_name, position
    """
    try:
        all_teams = teams.get_teams()
        team_info = [t for t in all_teams if t["abbreviation"] == team_abbrev]

        if not team_info:
            return pd.DataFrame()

        team_id = team_info[0]["id"]

        rate_limit()
        from nba_api.stats.endpoints import commonteamroster

        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        df = roster.get_data_frames()[0]

        if not df.empty:
            players_list = []
            for _, row in df.iterrows():
                players_list.append(
                    {
                        "player_id": row["PLAYER_ID"],
                        "full_name": row["PLAYER"],
                        "position": row.get("POSITION", "F"),
                    }
                )
            return pd.DataFrame(players_list)

        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching team roster: {e}")
        return pd.DataFrame()


# -------------------------------------------------
# Defense profile vs position for a team
# -------------------------------------------------
def get_team_defense_rank_vs_position(team_abbrev, player_position, def_vs_pos_df):
    """
    Get a team's defensive rank + contextual stats vs a given player position.

    Returns dict with:
        rank, percentile, rating,
        pts_allowed, fg_pct, tpm_allowed, stl, blk, etc.
    """
    if def_vs_pos_df.empty:
        return {
            "rank": 15,
            "total": 30,
            "percentile": 50,
            "rating": "Average",
            "pts_allowed": 110.0,
            "fg_pct": 45.0,
            "tpm_allowed": 3.0,
            "stl": 1.5,
            "blk": 1.0,
        }

    # Map player role to the site's buckets
    position_mapping = {
        "G": ["PG", "SG"],  # Guards
        "F": ["SF", "PF"],  # Forwards
        "C": ["C"],  # Centers
    }

    # HashtagBasketball sometimes uses slightly different team codes
    team_abbrev_map = {
        "NYK": "NY",  # New York Knicks
        "NOP": "NO",  # New Orleans Pelicans
        "SAS": "SA",  # San Antonio Spurs
        "GSW": "GS",  # Golden State Warriors
        "PHX": "PHO",  # Phoenix Suns
        # most other teams match
    }

    # normalize team code
    search_team = team_abbrev_map.get(team_abbrev, team_abbrev)

    # which sub-positions we average (ex: "G" -> PG+SG)
    positions_to_check = position_mapping.get(player_position, ["SF"])

    rows = []
    for pos in positions_to_check:
        hit = def_vs_pos_df[
            (def_vs_pos_df["Team"] == search_team)
            & (def_vs_pos_df["Position"] == pos)
        ]
        if not hit.empty:
            rows.append(hit.iloc[0])

    if not rows:
        return {
            "rank": 15,
            "total": 30,
            "percentile": 50,
            "rating": "Average",
            "pts_allowed": 110.0,
            "fg_pct": 45.0,
            "tpm_allowed": 3.0,
            "stl": 1.5,
            "blk": 1.0,
        }

    # average stats across PG+SG or SF+PF etc.
    avg_rank = sum(r["Rank"] for r in rows) / len(rows)
    avg_pts = sum(r["PTS"] for r in rows) / len(rows)
    avg_fg_pct = sum(r["FG_PCT"] for r in rows) / len(rows)
    avg_ft_pct = sum(r["FT_PCT"] for r in rows) / len(rows)
    avg_tpm = sum(r["TPM"] for r in rows) / len(rows)
    avg_reb = sum(r["REB"] for r in rows) / len(rows)
    avg_ast = sum(r["AST"] for r in rows) / len(rows)
    avg_stl = sum(r["STL"] for r in rows) / len(rows)
    avg_blk = sum(r["BLK"] for r in rows) / len(rows)
    avg_to = sum(r["TO"] for r in rows) / len(rows)

    # percentile (lower rank = tougher defense)
    percentile = ((30 - avg_rank) / 30) * 100

    # readable rating
    if avg_rank <= 10:
        rating = "Elite"
    elif avg_rank <= 15:
        rating = "Above Average"
    elif avg_rank <= 20:
        rating = "Average"
    else:
        rating = "Below Average"

    return {
        "rank": int(avg_rank),
        "total": 30,
        "percentile": round(percentile, 1),
        "rating": rating,
        "positions_checked": positions_to_check,
        # Defensive profile numbers:
        "pts_allowed": round(avg_pts, 1),
        "fg_pct": round(avg_fg_pct, 1),
        "ft_pct": round(avg_ft_pct, 1),
        "tpm_allowed": round(avg_tpm, 1),
        "reb_allowed": round(avg_reb, 1),
        "ast_allowed": round(avg_ast, 1),
        "stl": round(avg_stl, 1),
        "blk": round(avg_blk, 1),
        "to_forced": round(avg_to, 1),
    }
