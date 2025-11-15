"""
api/get_data.py
Main entry for local dev: orchestrates helpers + model.
Loads .env from project root (parent of api/).
"""

import os
import sys
import json
import time # <-- FIX: Import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# --- TEST MODE ---
# Set to False to process all games (production).
TEST_MODE_ENABLED = False # <-- FIX: Set to False for production
# --- END TEST MODE ---

# make project root visible for imports when running `python api/get_data.py`
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    from api import helpers
    from api import model
except Exception:
    import helpers
    import model

# Load .env from project root (parent directory of this file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # fallback to default load (still tries current working dir)
    load_dotenv()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", None)

def get_current_nba_season():
    """
    Calculates the current NBA season string (e.g., "2025-26")
    based on the current date (season typically rolls over in October).
    """
    now = datetime.now()
    if now.month >= 10: # Season starts in October
        start_year = now.year
        end_year = str(now.year + 1)[-2:]
    else: # Season is in the early months of the calendar year
        start_year = now.year - 1
        end_year = str(now.year)[-2:]
    return f"{start_year}-{end_year}"

# Use dynamic season calculation instead of hardcoding
CURRENT_SEASON = get_current_nba_season()
prior_year_start = int(CURRENT_SEASON.split('-')[0]) - 1
prior_year_end = str(prior_year_start + 1)[-2:]
PRIOR_SEASON = f"{prior_year_start}-{prior_year_end}"

print(f"[get_data] Using CURRENT_SEASON: {CURRENT_SEASON}")
print(f"[get_data] Using PRIOR_SEASON: {PRIOR_SEASON}")


def get_games_list():
    """
    Returns list of upcoming games without processing player data.
    Used by frontend to know how many games to fetch.
    """
    try:
        games = helpers.get_upcoming_games(days=2)
        return games if games else []
    except Exception as e:
        print(f"[get_data][ERROR] Failed to fetch games list: {e}")
        return []


def get_real_player_data(game_index=None):
    """
    Process a single game (by index) and return player data for that game only.
    If game_index is None, returns empty list (frontend should call this per game).
    """
    start = datetime.now()
    print(f"[get_data] --- API REQUEST START (game_index={game_index}) ---")

    # STEP 1: fetch global data (shared across all games)
    try:
        print("[get_data] STEP 1: Fetching upcoming games, players, DVP and team stats...")
        games = helpers.get_upcoming_games(days=2)
        if not games:
            print("[get_data][INFO] No upcoming games found -> returning []")
            return []
        
        # Validate game_index
        if game_index is None:
            print("[get_data][WARN] game_index is required, returning empty list")
            return []
        
        if not (0 <= game_index < len(games)):
            print(f"[get_data][WARN] Invalid game_index {game_index} (valid range: 0-{len(games)-1}), returning empty list")
            return []
        
        # Get the specific game
        game = games[game_index]
        game_id = game.get("game_id")
        home = game.get("home")
        away = game.get("away")
        desc = game.get("game_description")
        gdate = game.get("game_date")
        
        print(f"[get_data] Processing game {game_index + 1}/{len(games)}: {desc} ({game_id})")
        
        all_players_df = helpers.get_all_active_players()
        dvp_data = helpers.scrape_dvp_stats()
        team_stats_df = helpers.get_team_stats(CURRENT_SEASON)
    except Exception as e:
        print(f"[get_data][ERROR] Step 1 failed: {e}")
        return {"error": f"Failed to fetch global data: {e}"}

    # STEP 2: odds events (optional)
    print("[get_data] STEP 2: Fetching odds events (optional)...")
    try:
        all_odds_events = helpers.fetch_all_odds_events(ODDS_API_KEY)
    except Exception as e:
        print(f"[get_data][WARN] Odds events fetch failed: {e}")
        all_odds_events = []

    # STEP 3: process single game & players
    prop_model = model.PlayerPropModel()
    results = []

    # Find event id in odds list if available (best-effort)
    event_id = None
    for ev in all_odds_events:
        # best-effort match by team names
        if ev.get("id") and (game.get("home_team_name") in (ev.get("home_team") or "") or game.get("away_team_name") in (ev.get("away_team") or "")):
            event_id = ev.get("id")
            break
    game_odds = {}
    if event_id:
        game_odds = helpers.fetch_player_props_for_event(ODDS_API_KEY, event_id)

    # get rosters
    rosters = {}
    for t in (home, away):
        if not t or t == "N/A":
            continue
        try:
            rosters[t] = helpers.get_team_roster(t, season=CURRENT_SEASON)
        except Exception as e:
            print(f"[get_data][WARN] roster fetch failed for {t}: {e}")
            rosters[t] = []

    for team_abbrev, roster in rosters.items():
        opponent_abbrev = away if team_abbrev == home else home
        if not roster:
            print(f"[get_data][WARN] Empty roster for {team_abbrev}, skipping")
            continue

        for player in roster:
            player_id = player.get("id")
            player_name = player.get("full_name")
            print(f"[get_data]  Player {player_name} ({team_abbrev} vs {opponent_abbrev})")

            try:
                logs_current = helpers.get_player_game_logs(player_id, CURRENT_SEASON)
                logs_prior = helpers.get_player_game_logs(player_id, PRIOR_SEASON)

                if logs_current is None and logs_prior is None:
                    print(f"[get_data]    No logs for {player_name} -> skip")
                    continue

                frames = [df for df in [logs_current, logs_prior] if df is not None]
                all_logs = pd.concat(frames, ignore_index=True).sort_values(by="GAME_DATE", ascending=False)
                if all_logs.empty:
                    print(f"[get_data]    Combined logs empty for {player_name} -> skip")
                    continue

                h2h = helpers.get_head_to_head_history(all_logs, opponent_abbrev)

                # position (best-effort)
                pos = helpers.get_player_position(player_name)

                features = model.build_enhanced_feature_vector(
                    player_game_logs=all_logs,
                    opponent_abbrev=opponent_abbrev,
                    team_abbrev=team_abbrev,
                    player_pos=pos,
                    dvp_data=dvp_data,
                    head_to_head_games=h2h, # Pass the filtered DF
                    team_stats_df=team_stats_df
                )

                for stat in prop_model.stat_types:
                    projection = prop_model.predict(features, stat)
                    if projection is None or projection == 0:
                        continue

                    book_line = None
                    ou_str = None
                    try:
                        cand = game_odds.get(player_name, {})
                        if cand:
                            if stat == "pts" and cand.get("pts"):
                                book_line = cand["pts"]["line"]
                                ou_str = cand["pts"]["over_under"]
                            if stat == "ast" and cand.get("ast"):
                                book_line = cand["ast"]["line"]
                                ou_str = cand["ast"]["over_under"]
                            if stat == "reb" and cand.get("reb"):
                                book_line = cand["reb"]["line"]
                                ou_str = cand["reb"]["over_under"]
                    except Exception:
                        pass

                    hit_rates = model.calculate_hit_rates(all_logs, stat, book_line)
                    pdata = {
                        "id": f"{player_id}-{stat}",
                        "gameId": game_id,
                        "gameDate": gdate,
                        "gameDescription": desc,
                        "playerName": player_name,
                        "team": team_abbrev,
                        "opponent": opponent_abbrev,
                        "stat": stat,
                        "projection": round(float(projection), 2),
                        "bookLine": book_line,
                        "overUnder": ou_str,
                        "isStarter": True,
                        "hitRate": {
                            "L5": hit_rates.get("L5", 0),
                            "L10": hit_rates.get("L10", 0),
                            "Season": hit_rates.get("Season", 0)
                        }
                    }
                    results.append(pdata)

            except Exception as e:
                print(f"[get_data][ERROR] Failed processing player {player_name}: {e}")
                continue


    end = datetime.now()
    dur = (end - start).total_seconds()
    print(f"[get_data] --- API REQUEST END (duration: {dur:.1f}s). Results: {len(results)} players")
    return results


# Local Flask wrapper (for python api/get_data.py)
if __name__ == "__main__":
    from flask import Flask, jsonify,request
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    print("[get_data] Loading .env from", env_path if 'env_path' in locals() else "(default)")
    # already loaded above

    @app.route("/api/get_games", methods=["GET"])
    def local_get_games():
        """Returns list of upcoming games without processing player data."""
        try:
            games = get_games_list()
            return jsonify(games), 200
        except Exception as e:
            print(f"[get_data][FATAL] {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/get_data", methods=["GET"])
    def local_get_data():
        """Processes a single game by index and returns player data for that game only."""
        try:
            game_index = request.args.get("game_index", default=None, type=int)
            if game_index is None:
                return jsonify({"error": "game_index parameter is required"}), 400
            data = get_real_player_data(game_index=game_index)
            if isinstance(data, dict) and "error" in data:
                return jsonify(data), 500
            return jsonify(data), 200
        except Exception as e:
            print(f"[get_data][FATAL] {e}")
            return jsonify({"error": str(e)}), 500


    print("[get_data] Starting Flask dev server on port 5000")
    app.run(debug=True, port=5000)