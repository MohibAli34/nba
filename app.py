# hadi bodla can youu see this? 
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
import uuid
import hashlib
import random
import json
from datetime import datetime
from copy import deepcopy
from dotenv import load_dotenv

# Load environment variables from .env file at startup (if it exists)
# On Streamlit Cloud, set environment variables in the dashboard
load_dotenv()

# make sure we can import from utils/
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from utils.data_fetcher import (
    get_player_id,
    get_opponent_recent_games,
    get_head_to_head_history,
    get_player_position,
    get_team_defense_rank_vs_position,
    get_players_by_team,
    get_upcoming_games,
    fetch_fanduel_lines,      # live Odds API -> FanDuel lines
    get_event_id_for_game,    # resolve game -> event id
    get_player_fanduel_line,  # pull line for one player/stat
    scrape_rotowire_starters,  # scrape projected starters
    is_player_starter,         # check if player is starter
)

from utils.cached_data_fetcher import (
    get_player_game_logs_cached_db,
    get_team_stats_cached_db,
    scrape_defense_vs_position_cached_db,
)

from utils.game_data_cache import get_cached_game_data

from utils.database import get_cache_stats, clear_old_seasons

from utils.features import (
    build_enhanced_feature_vector,
)

from utils.model import PlayerPropModel


# ─────────────────────────────
# Page config
# ─────────────────────────────
st.set_page_config(
    page_title="NBA Player Props Model",
    page_icon="🏀",
    layout="wide",
)


# ─────────────────────────────
# Season helpers
# ─────────────────────────────
def get_current_nba_season():
    """
    Guess current NBA season like '2025-26'.
    If today is Oct or later, it's YEAR-(YEAR+1 short).
    Otherwise it's (YEAR-1)-(YEAR short).
    """
    now = datetime.now()
    yr = now.year
    mo = now.month
    if mo >= 10:
        return f"{yr}-{str(yr+1)[2:]}"
    else:
        return f"{yr-1}-{str(yr)[2:]}"


def get_prior_nba_season():
    cur = get_current_nba_season()
    start_year = int(cur.split('-')[0])
    return f"{start_year-1}-{str(start_year)[2:]}"


current_season = get_current_nba_season()
prior_season = get_prior_nba_season()


# ─────────────────────────────
# Model
# ─────────────────────────────
@st.cache_resource
def load_model():
    return PlayerPropModel(alpha=1.0)

model = load_model()


# ─────────────────────────────
# Stat options for dropdown
# ─────────────────────────────
STAT_OPTIONS = {
    "Points": "PTS",
    "Assists": "AST",
    "Rebounds": "REB",
    "Three-Pointers Made": "FG3M",
    "Points + Rebounds + Assists (PRA)": "PRA",
    "Double-Double Probability": "DD",
}


# ─────────────────────────────
# Utility helpers for sportsbook + display
# ─────────────────────────────
def defense_emoji(rank_num: int) -> str:
    """
    Visual difficulty emoji:
    - rank <=10 : tough defense (red)
    - rank <=20 : middling (yellow/orange)
    - else      : soft / target (green)
    """
    if rank_num <= 10:
        return "🔴"
    elif rank_num <= 20:
        return "🟡"
    else:
        return "🟢"
def safe_get_adjusted_line(player_identifier, stat_code, fd_line_val):
    if 'session_state' not in st.__dict__:
        return fd_line_val  # fallback if Streamlit session not ready
    return get_adjusted_line_value(player_identifier, stat_code, fd_line_val)

def _get_stat_series_for_hit_rate(game_logs: pd.DataFrame, stat_code: str):
    if game_logs is None or game_logs.empty:
        return None

    stat_key = (stat_code or "").upper()

    if stat_key == "PRA":
        required_cols = {"PTS", "REB", "AST"}
        if not required_cols.issubset(game_logs.columns):
            return None
        return game_logs["PTS"] + game_logs["REB"] + game_logs["AST"]

    if stat_key in game_logs.columns:
        return game_logs[stat_key]

    return None


def _calc_hit_rate_threshold(
    game_logs: pd.DataFrame,
    stat_code: str,
    threshold_value: float,
    window: int | None = None,
    denominator: int | None = None,
):
    if game_logs is None or game_logs.empty:
        return None
    if threshold_value is None:
        return None

    stat_series = _get_stat_series_for_hit_rate(game_logs, stat_code)
    if stat_series is None:
        return None

    if window is not None:
        stat_series = stat_series.head(window)

    if stat_series.empty:
        return None

    hits = (stat_series > threshold_value).sum()

    if denominator is None:
        denominator = window if window is not None else len(stat_series)

    if denominator is None or denominator <= 0:
        denominator = len(stat_series)

    if denominator <= 0:
        return None

    return (hits / denominator) * 100.0


def calc_hit_rate(game_logs: pd.DataFrame, stat_col: str, line_value: float, window: int = 10):
    """
    % of last `window` games the player went OVER line_value for stat_col.
    If not enough data or no line, return None.
    """
    return _calc_hit_rate_threshold(
        game_logs,
        stat_col,
        line_value,
        window=window,
        denominator=window,
    )


def calc_hit_rate_vs_projection(
    game_logs: pd.DataFrame,
    stat_col: str,
    projection_value: float,
    window: int | None = None,
    denominator: int | None = None,
):
    """Hit rate versus the player's projection instead of the sportsbook line."""
    return _calc_hit_rate_threshold(
        game_logs,
        stat_col,
        projection_value,
        window=window,
        denominator=denominator,
    )


def calc_edge(prediction: float, line_value: float):
    """
    Compare model projection to sportsbook line.
    Returns (edge_str, rec_text, ou_short)
    edge_str ~ "+2.1 (+9.4%)"
    rec_text ~ "✅ OVER looks good" / "❌ UNDER looks good" / "⚪ No clear edge"
    ou_short ~ "OVER", "UNDER", or "—"
    """
    if line_value is None:
        return ("—", "No line", "—")

    if line_value == 0:
        diff = prediction
        pct = 0.0
    else:
        diff = prediction - line_value
        pct = (diff / line_value) * 100.0

    if abs(diff) < 1.5:
        rec_text = "⚪ No clear edge"
        ou_short = "—"
    elif diff > 1.5:
        rec_text = "✅ OVER looks good"
        ou_short = "OVER"
    else:
        rec_text = "❌ UNDER looks good"
        ou_short = "UNDER"

    edge_str = f"{diff:+.1f} ({pct:+.1f}%)"
    return (edge_str, rec_text, ou_short)


# ─────────────────────────────
# Session State Helper Functions
# ─────────────────────────────
def _get_session_state():
    """
    Safely return the current Streamlit session_state object or a dummy object if
    Streamlit has not finished initializing it yet.
    """
    if not hasattr(st, '_session_state'):
        # Create a dummy session state if it doesn't exist yet
        class DummySessionState:
            def __init__(self):
                self._state = {}
            def __getattr__(self, name):
                if name not in self._state:
                    self._state[name] = {}
                return self._state[name]
            def __setattr__(self, name, value):
                if name == '_state':
                    return super().__setattr__(name, value)
                if name not in self._state:
                    self._state[name] = {}
                self._state[name] = value
            def get(self, key, default=None):
                return self._state.get(key, default)
            
        st._session_state = DummySessionState()
    
    return st._session_state


def safe_session_state_get(key, default_value=None):
    """
    Safely get a value from session state, handling initialization errors.
    """
    try:
        session_state = _get_session_state()
        if hasattr(session_state, 'get'):
            return session_state.get(key, default_value)
        return getattr(session_state, key, default_value)
    except Exception as e:
        print(f"Error accessing session state: {e}")
        return default_value


def safe_session_state_set(key, value):
    """
    Safely set a value in session state, handling initialization errors.
    """
    try:
        session_state = _get_session_state()
        if hasattr(session_state, '__setitem__'):
            session_state[key] = value
        else:
            setattr(session_state, key, value)
    except Exception as e:
        print(f"Error setting session state: {e}")

def safe_session_state_init(key, default_value):
    """
    Initialize a session state key with a default value if it does not
    already exist, returning the stored value (or a deepcopy of the
    default when session state is unavailable).
    """
    try:
        session_state = _get_session_state()
        
        # Check if key exists using getattr for attribute-style access
        if hasattr(session_state, key):
            return getattr(session_state, key)
            
        # If not found, set the default value
        default_copy = deepcopy(default_value)
        if hasattr(session_state, '__setitem__'):
            session_state[key] = default_copy
        else:
            setattr(session_state, key, default_copy)
            
        return default_copy
    except Exception as e:
        print(f"Error initializing session state: {e}")
        return deepcopy(default_value)


SESSION_DEFAULTS = {
    "adjusted_lines": {},
    "manual_lines": {},
    "hit_rates": {},
    "stable_ids": {},
    "hit_rate_window_view": "L10",  # Global hit rate window view for all players
}


def initialize_session_state_defaults():
    """
    Ensure that all root session state containers we rely on exist
    before any widgets attempt to mutate them.
    """
    for key, default in SESSION_DEFAULTS.items():
        safe_session_state_init(key, default)


def make_player_stat_key(player_identifier, stat_code):
    return f"{player_identifier}:{stat_code}"


def get_adjusted_line_value(player_identifier, stat_code, default_value=None):
    """Get the adjusted line value from session state with fallback to default."""
    adjusted_lines = safe_session_state_init("adjusted_lines", {})
    key = make_player_stat_key(player_identifier, stat_code)
    if default_value is not None and key not in adjusted_lines:
        adjusted_lines[key] = default_value
    return adjusted_lines.get(key, default_value)


def set_adjusted_line_value(player_identifier, stat_code, value):
    """Set the adjusted line value in session state and trigger a re-render."""
    adjusted_lines = safe_session_state_init("adjusted_lines", {})
    key = make_player_stat_key(player_identifier, stat_code)
    adjusted_lines[key] = value
    # Store the last update time for animation
    st.session_state[f"{key}_last_updated"] = time.time()


def reset_adjusted_line_value(player_identifier, stat_code, default_value=0.0):
    """Reset the adjusted line to its default value."""
    adjusted_lines = safe_session_state_init("adjusted_lines", {})
    key = make_player_stat_key(player_identifier, stat_code)
    if key in adjusted_lines:
        del adjusted_lines[key]
    return default_value


def get_manual_line_value(player_identifier, stat_code, default_value=0.0):
    """Get the manually set line value from session state."""
    manual_lines = safe_session_state_init("manual_lines", {})
    key = make_player_stat_key(player_identifier, stat_code)
    if key not in manual_lines:
        manual_lines[key] = default_value
    return manual_lines[key]


def set_manual_line_value(player_identifier, stat_code, value):
    """Set the manual line value in session state."""
    manual_lines = safe_session_state_init("manual_lines", {})
    key = make_player_stat_key(player_identifier, stat_code)
    manual_lines[key] = value
    st.session_state[f"{key}_last_updated"] = time.time()


def get_line_step_size():
    """Get the current step size for line adjustments."""
    return safe_session_state_init("line_step_size", 0.5)


def set_line_step_size(step_size):
    """Set the step size for line adjustments."""
    st.session_state["line_step_size"] = step_size


def add_custom_css():
    """Add custom CSS for animations and styling."""
    st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }
        @keyframes flashGreen {
            0% { background-color: rgba(16, 185, 129, 0.1); }
            50% { background-color: rgba(16, 185, 129, 0.3); }
            100% { background-color: transparent; }
        }
        @keyframes flashRed {
            0% { background-color: rgba(239, 68, 68, 0.1); }
            50% { background-color: rgba(239, 68, 68, 0.3); }
            100% { background-color: transparent; }
        }
        @keyframes slideInRight {
            from { transform: translateX(100%); }
            to { transform: translateX(0); }
        }
        @keyframes slideInUp {
            from { transform: translateY(100%); }
            to { transform: translateY(0); }
        }
        .hit-rate-updated {
            animation: fadeIn 0.5s ease-in-out;
        }
        .hit-rate-increased {
            animation: flashGreen 1s ease-in-out;
        }
        .hit-rate-decreased {
            animation: flashRed 1s ease-in-out;
        }
        .line-adjust-btn {
            width: 100%;
            height: 100%;
            min-height: 38px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
            font-weight: bold;
        }
        .bet-sheet-sidebar {
            position: fixed;
            right: 0;
            top: 0;
            width: 380px;
            height: 100vh;
            background: white;
            box-shadow: -2px 0 10px rgba(0,0,0,0.1);
            z-index: 999;
            overflow-y: auto;
            padding: 1rem;
        }
        .bet-sheet-item {
            padding: 12px;
            margin: 8px 0;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #f9fafb;
        }
        .bet-sheet-item:hover {
            background: #f3f4f6;
        }
        .bet-sheet-floating-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 998;
            background: #1f77b4;
            color: white;
            border: none;
            border-radius: 50px;
            padding: 12px 24px;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            cursor: pointer;
        }
        .bet-sheet-floating-btn:hover {
            background: #1565a0;
        }
        .toast-notification {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            animation: fadeIn 0.3s ease-in-out;
        }
        .toast-success {
            background: #10b981;
            color: white;
        }
        .toast-info {
            background: #3b82f6;
            color: white;
        }
        .toast-error {
            background: #ef4444;
            color: white;
        }
        @media (max-width: 768px) {
            .bet-sheet-sidebar {
                width: 100%;
                height: 60vh;
                bottom: 0;
                top: auto;
                border-radius: 20px 20px 0 0;
                animation: slideInUp 0.3s ease-in-out;
            }
        }
    </style>
    """, unsafe_allow_html=True)


def get_hit_rate_style(hit_rate, previous_hit_rate=None):
    """Get CSS style for hit rate based on its value and change from previous."""
    if hit_rate is None:
        return ""
    
    base_style = ""
    
    # Animation for recent update
    key = f"hit_rate_{id(hit_rate)}"
    if key in st.session_state and time.time() - st.session_state[key] < 1.5:
        base_style += "animation: fadeIn 0.5s ease-in-out;"
    
    # Color based on hit rate value
    if hit_rate >= 60:
        base_style += "color: #10B981;"  # Green for high hit rate
    elif hit_rate <= 40:
        base_style += "color: #EF4444;"  # Red for low hit rate
    
    # Highlight significant changes
    if previous_hit_rate is not None:
        change = hit_rate - previous_hit_rate
        if abs(change) >= 5:  # Slightly more sensitive to changes
            if change > 0:
                base_style += "font-weight: bold; color: #10B981;"
                st.session_state[f"{key}_increased"] = True
            else:
                base_style += "font-weight: bold; color: #EF4444;"
                st.session_state[f"{key}_decreased"] = True
    
    return base_style


def get_or_create_stable_id(player_identifier, stat_code):
    stable_ids = safe_session_state_init("stable_ids", {})
    key = make_player_stat_key(player_identifier, stat_code)
    if key not in stable_ids:
        stable_ids[key] = str(uuid.uuid4())[:12]
    return stable_ids[key]


def cache_hit_rate(player_identifier, stat_code, value):
    hit_rates = safe_session_state_init("hit_rates", {})
    key = make_player_stat_key(player_identifier, stat_code)
    hit_rates[key] = value


def _get_widget_token(namespace: str) -> str:
    """
    Return a stable UUID token for a widget namespace stored in session state.
    Ensures widget keys remain globally unique across reruns while maintaining
    consistency for the same logical widget.
    """
    tokens = safe_session_state_init("_widget_tokens", {})
    if namespace not in tokens:
        tokens[namespace] = str(uuid.uuid4())
    return tokens[namespace]



# ─────────────────────────────
# Bet Sheet Functions
# ─────────────────────────────
def get_bet_sheet_file_path():
    """Get the path to the bet sheet JSON file."""
    return os.path.join(os.path.dirname(__file__), 'bet_sheet.json')


def load_bet_sheet_from_file():
    """Load bet sheet from JSON file."""
    file_path = get_bet_sheet_file_path()
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('bet_sheet', []), data.get('bet_sheet_settings', {
                    'allow_duplicates': False,
                    'line_step_size': 0.5,
                    'default_hit_rate_window': 'L10'
                })
        except Exception as e:
            st.warning(f"Error loading bet sheet: {e}")
            return [], {
                'allow_duplicates': False,
                'line_step_size': 0.5,
                'default_hit_rate_window': 'L10'
            }
    return [], {
        'allow_duplicates': False,
        'line_step_size': 0.5,
        'default_hit_rate_window': 'L10'
    }


def save_bet_sheet_to_file():
    """Save bet sheet to JSON file."""
    file_path = get_bet_sheet_file_path()
    try:
        # Convert bet sheet to JSON-serializable format
        bet_sheet = st.session_state.get('bet_sheet', [])
        bet_sheet_serializable = []
        for bet in bet_sheet:
            bet_copy = bet.copy()
            # Convert timestamp to string if it's a pd.Timestamp
            if 'timestamp' in bet_copy and isinstance(bet_copy['timestamp'], pd.Timestamp):
                bet_copy['timestamp'] = bet_copy['timestamp'].isoformat()
            bet_sheet_serializable.append(bet_copy)
        
        data = {
            'bet_sheet': bet_sheet_serializable,
            'bet_sheet_settings': st.session_state.get('bet_sheet_settings', {
                'allow_duplicates': False,
                'line_step_size': 0.5,
                'default_hit_rate_window': 'L10'
            })
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving bet sheet: {e}")


def initialize_bet_sheet():
    """Initialize bet sheet in session state if it doesn't exist. Always starts with empty bet sheet."""
    if 'bet_sheet' not in st.session_state:
        # ALWAYS START WITH EMPTY BET SHEET
        st.session_state.bet_sheet = []
        # Still load settings from file if available, or use defaults
        _, settings = load_bet_sheet_from_file()
        st.session_state.bet_sheet_settings = settings
    if 'bet_sheet_settings' not in st.session_state:
        st.session_state.bet_sheet_settings = {
            'allow_duplicates': False,
            'line_step_size': 0.5,
            'default_hit_rate_window': 'L10'
        }
    if 'bet_sheet_toast' not in st.session_state:
        st.session_state.bet_sheet_toast = None
    if 'bet_sheet_show_clear_confirm' not in st.session_state:
        st.session_state.bet_sheet_show_clear_confirm = False


def get_bet_sheet_item_id(player_id, stat_code, line):
    """Generate a unique ID for a bet item based on player, stat, and line."""
    if line is None:
        # Use timestamp for unique ID when line is None
        return f"{player_id}_{stat_code}_{int(time.time() * 1000)}"
    # Always include timestamp to ensure uniqueness even for same player+stat+line
    return f"{player_id}_{stat_code}_{line:.1f}_{int(time.time() * 1000)}"


def add_to_bet_sheet(player_data, stat_display, projection, line, edge, 
                     line_source='scraped', hit_rates=None, allow_duplicate=False):
    """
    Add a player and their stat to the bet sheet.
    
    Args:
        player_data: Dict with player_id, player_name, team_abbrev, stat_code
        stat_display: Display name for the stat
        projection: Model projection value
        line: Line value (may be None)
        edge: Edge string
        line_source: 'scraped', 'manual', or 'adjusted'
        hit_rates: Dict with L5, L10, Season hit rates
        allow_duplicate: Whether to allow duplicate entries
    """
    initialize_bet_sheet()
    
    # Check for exact duplicates (same player + stat + line)
    # Allow duplicates only if stat OR line is different
    # Prevent only if player, stat, AND line are all the same
    if line is not None:
        # Check for exact duplicate: same player, same stat, same line
        existing = next((i for i, bet in enumerate(st.session_state.bet_sheet) 
                        if (bet.get('player_id') == player_data['player_id'] and 
                            bet.get('stat_code') == player_data['stat_code'] and
                            bet.get('line') is not None and
                            abs(bet.get('line', 0) - line) < 0.01)), None)
        if existing is not None:
            line_display = f"{line:.1f}" if line else "N/A"
            st.session_state.bet_sheet_toast = {
                'type': 'info',
                'message': f"Already in bet sheet: {player_data['player_name']} ({stat_display} {line_display})"
            }
            st.rerun()
            return
    else:
        # For None lines, check if there's already an entry with same player+stat+None line
        existing = next((i for i, bet in enumerate(st.session_state.bet_sheet) 
                        if (bet.get('player_id') == player_data['player_id'] and 
                            bet.get('stat_code') == player_data['stat_code'] and
                            bet.get('line') is None)), None)
        if existing is not None and not allow_duplicate:
            st.session_state.bet_sheet_toast = {
                'type': 'info',
                'message': f"Already in bet sheet: {player_data['player_name']} ({stat_display}) - Please set line first"
            }
            st.rerun()
            return
    
    # Create bet item (line can be None) - generate unique ID
    bet_id = get_bet_sheet_item_id(player_data['player_id'], player_data['stat_code'], line)
    
    bet_data = {
        'id': bet_id,
        'player_name': player_data['player_name'],
        'player_id': player_data['player_id'],
        'team_abbrev': player_data.get('team_abbrev', ''),
        'stat_code': player_data['stat_code'],
        'stat_display': stat_display,
        'projection': projection,
        'line': line,
        'line_source': line_source,
        'edge': edge,
        'stake': None,  # Optional stake input
        'hit_rates': hit_rates or {'L5': None, 'L10': None, 'Season': None},
        'hit_rate_window': st.session_state.bet_sheet_settings.get('default_hit_rate_window', 'L10'),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    st.session_state.bet_sheet.append(bet_data)
    
    # Save to file
    save_bet_sheet_to_file()
    
    # Toast message
    if line is not None:
        toast_msg = f"Added {player_data['player_name']} ({stat_display} {line:+.1f}) to Bet Sheet"
    else:
        toast_msg = f"Added {player_data['player_name']} ({stat_display}) to Bet Sheet - Enter line below"
    
    st.session_state.bet_sheet_toast = {
        'type': 'success',
        'message': toast_msg
    }
    st.rerun()


def remove_from_bet_sheet(bet_id):
    """Remove a bet item from the sheet by ID."""
    initialize_bet_sheet()
    st.session_state.bet_sheet = [
        bet for bet in st.session_state.bet_sheet if bet.get('id') != bet_id
    ]
    # Save to file
    save_bet_sheet_to_file()
    st.rerun()


def recalculate_hit_rates_for_bet(bet, cur_season, prev_season):
    """Recalculate hit rates for a bet item based on current line and game logs."""
    player_id = bet.get('player_id')
    player_name = bet.get('player_name')
    stat_code = bet.get('stat_code')
    line = bet.get('line')
    projection_value = bet.get('projection')

    if projection_value is None or stat_code == "DD":
        return bet.get('hit_rates', {})
    
    # Get game logs
    current_logs = get_player_game_logs_cached_db(player_id, player_name, cur_season) if player_name else pd.DataFrame()
    if current_logs is None or current_logs.empty:
        prior_logs = get_player_game_logs_cached_db(player_id, player_name, prev_season) if player_name else pd.DataFrame()
        combined_logs = prior_logs if not prior_logs.empty else pd.DataFrame()
    else:
        combined_logs = current_logs
    
    if combined_logs.empty:
        return bet.get('hit_rates', {})
    
    # Calculate hit rates for all windows using the projection value
    hit_rates = {}
    hit_rates['L5'] = calc_hit_rate_vs_projection(combined_logs, stat_code, projection_value, window=5, denominator=5)
    hit_rates['L10'] = calc_hit_rate_vs_projection(combined_logs, stat_code, projection_value, window=10, denominator=20)
    hit_rates['Season'] = calc_hit_rate_vs_projection(combined_logs, stat_code, projection_value)

    return hit_rates


def update_bet_sheet_item(bet_id, updates, cur_season=None, prev_season=None):
    """Update a bet item with new values and recalculate hit rates if line changed."""
    initialize_bet_sheet()
    for i, bet in enumerate(st.session_state.bet_sheet):
        if bet.get('id') == bet_id:
            # If line is being updated, ensure it's not negative and recalculate hit rates
            if 'line' in updates and cur_season and prev_season:
                new_line = max(0.0, updates['line'])  # Prevent negative values
                updates['line'] = new_line
                bet_copy = bet.copy()
                bet_copy['line'] = new_line
                new_hit_rates = recalculate_hit_rates_for_bet(bet_copy, cur_season, prev_season)
                updates['hit_rates'] = new_hit_rates
                # Also update edge if we have projection
                if 'projection' in bet and new_line:
                    proj = bet.get('projection', 0)
                    edge_str, _, _ = calc_edge(proj, new_line)
                    updates['edge'] = edge_str
            
            st.session_state.bet_sheet[i].update(updates)
            break
    # Save to file
    save_bet_sheet_to_file()
    st.rerun()


def clear_bet_sheet():
    """Clear all items from the bet sheet."""
    initialize_bet_sheet()
    st.session_state.bet_sheet = []
    # Save to file
    save_bet_sheet_to_file()
    st.session_state.bet_sheet_toast = {
        'type': 'success',
        'message': 'Bet sheet cleared'
    }
    st.rerun()


def export_bet_sheet_csv():
    """Export bet sheet to CSV format."""
    initialize_bet_sheet()
    if not st.session_state.bet_sheet:
        return None
    
    rows = []
    for bet in st.session_state.bet_sheet:
        hit_rate = bet.get('hit_rates', {}).get(bet.get('hit_rate_window', 'L10'), None)
        rows.append({
            'Player': bet['player_name'],
            'Team': bet['team_abbrev'],
            'Stat': bet['stat_display'],
            'Line': bet['line'],
            'Projection': bet['projection'],
            'Edge': bet['edge'],
            'Hit Rate (%)': hit_rate if hit_rate is not None else '',
            'Stake': bet.get('stake', ''),
            'Line Source': bet.get('line_source', 'scraped'),
            'Added': bet.get('timestamp', '')
        })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def render_player_detail_body(pdata, cur_season, prev_season, render_index=None):
    # Add custom CSS for animations
    add_custom_css()
    
    # Get or create a stable identifier for this player/stat combo
    player_id = pdata.get("player_id")
    player_name = pdata.get("player_name")
    if not player_id:
        player_id = f"{player_name}_{pdata['team_abbrev']}"
    
    stat_code = pdata["stat_code"]
    stat_display = pdata.get("stat_display", stat_code)
    
    # Get the current line value (default to FanDuel line if available)
    fd_line_val = pdata.get("fd_line_val")
    current_line = get_adjusted_line_value(player_id, stat_code, fd_line_val or 0.0)
    
    # Get the projection and edge
    projection = pdata.get("prediction", 0)
    edge = pdata.get("edge_str", "—")
    
    # Add to Bet Sheet button with unique timestamp-based key
    button_key = f"add_to_betsheet_{player_id}_{stat_code}_{int(time.time() * 1000)}"
    if st.button("📝 Add to Bet Sheet", 
                key=button_key,
                help=f"Add {player_name}'s {stat_display} to your bet sheet"):
        add_to_bet_sheet(
            player_data={
                'player_id': player_id,
                'player_name': player_name,
                'team_abbrev': pdata.get('team_abbrev', ''),
                'stat_code': stat_code
            },
            stat_display=stat_display,
            projection=projection,
            line=current_line if current_line is not None else fd_line_val,
            edge=edge
        )
    
    # Get the step size for adjustments
    step_size = get_line_step_size()
    
    # Get game logs for hit rate calculation
    game_logs = get_player_game_logs_cached_db(player_id, player_name, cur_season) if player_name else pd.DataFrame()
    if game_logs is None:
        game_logs = pd.DataFrame()
    
    # Calculate hit rates for different time windows
    def calculate_hit_rates(logs, line, stat=stat_code):
        if logs.empty or line is None:
            return {"L5": 0, "L10": 0, "Season": 0, "H2H": 0}
            
        # Filter logs to relevant stat
        stat_col = stat.upper() if stat.upper() in logs.columns else None
        if stat_col is None:
            return {"L5": 0, "L10": 0, "Season": 0, "H2H": 0}
            
        # Calculate hits (1 if actual >= line, else 0)
        logs = logs.copy()
        logs['hit'] = (logs[stat_col] >= line).astype(int)
        
        # Calculate hit rates for different windows
        l5 = logs.head(5)['hit'].mean() * 100 if len(logs) >= 5 else 0
        l10 = logs.head(10)['hit'].mean() * 100 if len(logs) >= 10 else 0
        season = logs['hit'].mean() * 100
        
        # Calculate H2H hit rate if available
        h2h_hit_rate = 0
        if 'h2h_history' in pdata and pdata['h2h_history'] is not None and not pdata['h2h_history'].empty:
            h2h_logs = pdata['h2h_history'].copy()
            if stat_col in h2h_logs.columns:
                h2h_logs['hit'] = (h2h_logs[stat_col] >= line).astype(int)
                h2h_hit_rate = h2h_logs['hit'].mean() * 100
        
        return {
            "L5": round(l5, 1) if l5 > 0 else 0,
            "L10": round(l10, 1) if l10 > 0 else 0,
            "Season": round(season, 1) if season > 0 else 0,
            "H2H": round(h2h_hit_rate, 1) if h2h_hit_rate > 0 else 0
        }
    """
    The deep dive panel for a single player.
    Called inside each expander, after we build pdata in the loop.
    
    Args:
        pdata: Dictionary containing player data
        cur_season: Current season string
        prev_season: Previous season string
        render_index: Optional unique index for this render instance
    """
    player_name = pdata["player_name"]
    team_abbrev = pdata["team_abbrev"]
    player_pos = pdata["player_pos"]
    opponent_abbrev = pdata["opponent_abbrev"]
    current_logs = pdata["current_logs"]
    prior_logs = pdata["prior_logs"]
    h2h_history = pdata["h2h_history"]
    opp_def_rank = pdata["opp_def_rank"]
    features = pdata["features"]
    prediction = pdata["prediction"]
    stat_code = pdata["stat_code"]
    stat_display = pdata["stat_display"]

    # sportsbook extras we calculated in build loop
    fd_line_val = pdata["fd_line_val"]           # may be None
    hit_pct_val = pdata["hit_pct_val"]           # may be None
    edge_str = pdata["edge_str"]                 # string or "—"
    rec_text = pdata["rec_text"]                 # recommendation string

    # games played info
    has_current = not current_logs.empty
    has_prior = not prior_logs.empty
    current_games = len(current_logs) if has_current else 0
    prior_games = len(prior_logs) if has_prior else 0
    h2h_games = 0 if h2h_history is None or h2h_history.empty else len(h2h_history)

    # ---- Header metrics
    st.subheader(f"📊 Projections for {player_name} ↩")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(f"{cur_season} Games", current_games)
    with colB:
        st.metric(f"{prev_season} Games", prior_games)
    with colC:
        st.metric(f"vs {opponent_abbrev} History", h2h_games)

    if current_games < 5:
        st.info(
            f"Only {current_games} games in {cur_season}. "
            f"We're leaning more on {prev_season} + head-to-head."
        )

    # ---- Opponent defense vs position
    st.markdown("---")

    position_desc = {
        'G': 'Guards (PG/SG)',
        'F': 'Forwards (SF/PF)',
        'C': 'Centers (C)',
    }.get(player_pos, f'{player_pos} Position')

    st.subheader(f"🛡️ {opponent_abbrev} Defense vs {position_desc}")

    col1, col2, col3 = st.columns(3)

    rank_val = opp_def_rank.get("rank", 15)
    rating_text = opp_def_rank.get("rating", "Average")
    percentile = opp_def_rank.get("percentile", 50.0)

    rating_lower = str(rating_text).lower()
    if "elite" in rating_lower or "above" in rating_lower:
        diff_emoji = "🔴"
    elif "average" in rating_lower and "above" not in rating_lower:
        diff_emoji = "🟡"
    else:
        diff_emoji = "🟢"

    with col1:
        st.metric(
            "Defensive Rank vs Position",
            f"{diff_emoji} #{rank_val} of 30",
            help=f"How {opponent_abbrev} guards this archetype ({player_pos})"
        )
    with col2:
        st.metric(
            "Matchup Difficulty",
            rating_text,
            help="Elite / Above Avg = tough. Below Avg = soft / target spot."
        )
    with col3:
        st.metric(
            "Defense Percentile",
            f"{percentile:.0f}%",
            help="Higher percentile = stronger defense overall."
        )

    if "elite" in rating_lower or "above" in rating_lower:
        st.info(
            f"🔴 Tough matchup: {opponent_abbrev} defends {player_pos} well. "
            "Unders / caution."
        )
    elif "below" in rating_lower:
        st.success(
            f"🟢 Favorable matchup: {opponent_abbrev} struggles "
            f"vs {player_pos}. Overs become more viable."
        )

    # ---- Projection / performance / context
    st.markdown("---")
    colP, colR, colCxt = st.columns([2, 2, 1])

    # Projection panel
    with colP:
        st.subheader("🎯 Model Projection")
        if stat_code == "DD":
            st.metric("Double-Double Probability", f"{prediction:.1f}%")
        else:
            st.metric(
                f"Projected {stat_display}",
                f"{prediction:.1f}"
            )

        if "pts_allowed" in opp_def_rank:
            st.caption(
                f"🛡️ Opp vs {player_pos}: "
                f"{opp_def_rank['pts_allowed']:.1f} pts allowed"
            )
        else:
            st.caption("🛡️ Opponent defense data unavailable")

        if h2h_games > 0 and stat_code != "DD":
            h2h_avg = features.get(f"h2h_{stat_code}_avg", 0)
            st.caption(
                f"📊 vs {opponent_abbrev} Avg: {h2h_avg:.1f} "
                f"({h2h_games} games)"
            )

    # Recent performance with interactive buttons
    with colR:
        st.subheader("📈 Recent Performance")
        
        if stat_code != "DD":
            # Get game logs for detailed stats
            combined_logs = pd.DataFrame()
            if not current_logs.empty:
                combined_logs = current_logs.copy()
            if not prior_logs.empty:
                combined_logs = pd.concat([combined_logs, prior_logs], ignore_index=True)
            
            # Calculate stats for each period
            season_avg = features.get(f"{stat_code}_avg", 0)
            last5 = features.get(f"{stat_code}_last5", season_avg)
            last10 = features.get(f"{stat_code}_last10", season_avg)
            
            # Display quick summary cards at top
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 15px; border-radius: 10px; text-align: center; color: white;">
                    <div style="font-size: 12px; opacity: 0.9;">SEASON AVG</div>
                    <div style="font-size: 24px; font-weight: bold;">{season_avg:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with summary_col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 15px; border-radius: 10px; text-align: center; color: white;">
                    <div style="font-size: 12px; opacity: 0.9;">LAST 5</div>
                    <div style="font-size: 24px; font-weight: bold;">{last5:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with summary_col3:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 15px; border-radius: 10px; text-align: center; color: white;">
                    <div style="font-size: 12px; opacity: 0.9;">LAST 10</div>
                    <div style="font-size: 24px; font-weight: bold;">{last10:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Use tabs for better UX
            tab1, tab2, tab3 = st.tabs(["📊 Season Stats", "🔥 Last 5 Games", "📈 Last 10 Games"])
            
            with tab1:
                st.markdown(f"#### 🏆 Season Overview")
                
                if not combined_logs.empty:
                    # Calculate season stats
                    if stat_code == "PRA":
                        season_values = combined_logs["PTS"] + combined_logs["REB"] + combined_logs["AST"]
                    else:
                        season_values = combined_logs[stat_code] if stat_code in combined_logs.columns else pd.Series()
                    
                    if not season_values.empty:
                        # Main metric with large display
                        col_main, col_side = st.columns([2, 1])
                        with col_main:
                            st.markdown(f"""
                            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea;">
                                <h2 style="margin: 0; color: #667eea;">{season_avg:.1f}</h2>
                                <p style="margin: 5px 0 0 0; color: #666;">Season Average</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_side:
                            st.metric("Games", len(season_values), help="Total games played this season")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Stats grid
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        with stat_col1:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 10px; background: #fff3cd; border-radius: 6px;">
                                <div style="font-size: 11px; color: #856404;">HIGH</div>
                                <div style="font-size: 18px; font-weight: bold; color: #856404;">{season_values.max():.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with stat_col2:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 10px; background: #d1ecf1; border-radius: 6px;">
                                <div style="font-size: 11px; color: #0c5460;">LOW</div>
                                <div style="font-size: 18px; font-weight: bold; color: #0c5460;">{season_values.min():.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with stat_col3:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 10px; background: #d4edda; border-radius: 6px;">
                                <div style="font-size: 11px; color: #155724;">AVG</div>
                                <div style="font-size: 18px; font-weight: bold; color: #155724;">{season_values.mean():.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with stat_col4:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 10px; background: #f8d7da; border-radius: 6px;">
                                <div style="font-size: 11px; color: #721c24;">STD DEV</div>
                                <div style="font-size: 18px; font-weight: bold; color: #721c24;">{season_values.std():.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show recent trend
                        if len(season_values) >= 5:
                            st.markdown("<br>", unsafe_allow_html=True)
                            recent_5 = season_values.head(5).mean()
                            older_5 = season_values.iloc[5:10].mean() if len(season_values) >= 10 else season_values.iloc[5:].mean()
                            trend = recent_5 - older_5
                            trend_emoji = "📈" if trend >= 0 else "📉"
                            trend_color = "#28a745" if trend >= 0 else "#dc3545"
                            st.markdown(f"""
                            <div style="background: linear-gradient(90deg, {trend_color}15 0%, {trend_color}05 100%); 
                                        padding: 15px; border-radius: 8px; border-left: 4px solid {trend_color};">
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <span style="font-size: 24px;">{trend_emoji}</span>
                                    <div>
                                        <div style="font-weight: bold; color: {trend_color};">
                                            Recent Trend: {trend:+.1f}
                                        </div>
                                        <div style="font-size: 12px; color: #666;">
                                            Last 5 games vs Previous 5 games
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown(f"#### 🔥 Last 5 Games Performance")
                
                if not combined_logs.empty and len(combined_logs) >= 5:
                    last5_logs = combined_logs.head(5)
                    
                    if stat_code == "PRA":
                        l5_values = last5_logs["PTS"] + last5_logs["REB"] + last5_logs["AST"]
                    else:
                        l5_values = last5_logs[stat_code] if stat_code in last5_logs.columns else pd.Series()
                    
                    if not l5_values.empty:
                        l5_avg = l5_values.mean()
                        
                        # Main display
                        col_main_l5, col_comp_l5 = st.columns([2, 1])
                        with col_main_l5:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                        padding: 20px; border-radius: 8px; color: white;">
                                <h2 style="margin: 0; color: white;">{l5_avg:.1f}</h2>
                                <p style="margin: 5px 0 0 0; opacity: 0.9;">Last 5 Games Average</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_comp_l5:
                            vs_season = l5_avg - season_avg
                            delta_color = "#28a745" if vs_season >= 0 else "#dc3545"
                            delta_emoji = "⬆️" if vs_season >= 0 else "⬇️"
                            st.markdown(f"""
                            <div style="background-color: {delta_color}15; padding: 15px; border-radius: 8px; 
                                        border-left: 4px solid {delta_color}; text-align: center;">
                                <div style="font-size: 12px; color: #666;">vs Season</div>
                                <div style="font-size: 20px; font-weight: bold; color: {delta_color};">
                                    {delta_emoji} {vs_season:+.1f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Range display
                        range_col1, range_col2 = st.columns(2)
                        with range_col1:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 12px; background: #fff3cd; border-radius: 6px;">
                                <div style="font-size: 11px; color: #856404;">BEST GAME</div>
                                <div style="font-size: 20px; font-weight: bold; color: #856404;">{l5_values.max():.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with range_col2:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 12px; background: #d1ecf1; border-radius: 6px;">
                                <div style="font-size: 11px; color: #0c5460;">LOWEST GAME</div>
                                <div style="font-size: 20px; font-weight: bold; color: #0c5460;">{l5_values.min():.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Show individual game breakdown with better styling
                        st.markdown("#### 📋 Game-by-Game Breakdown")
                        game_data = []
                        for idx, (_, row) in enumerate(last5_logs.iterrows()):
                            if stat_code == "PRA":
                                val = row["PTS"] + row["REB"] + row["AST"]
                            else:
                                val = row[stat_code] if stat_code in row else 0
                            
                            matchup = row.get("MATCHUP", "N/A")
                            date = row.get("GAME_DATE", "N/A")
                            
                            # Format date nicely
                            try:
                                if isinstance(date, str) and len(date) > 10:
                                    date_formatted = date[:10]
                                else:
                                    date_formatted = str(date)[:10] if date else "N/A"
                            except:
                                date_formatted = "N/A"
                            
                            game_data.append({
                                "Game": f"#{idx+1}",
                                "Date": date_formatted,
                                "Matchup": matchup,
                                stat_display: f"{val:.1f}"
                            })
                        
                        if game_data:
                            df_games = pd.DataFrame(game_data)
                            # Style the dataframe
                            st.dataframe(
                                df_games.style.background_gradient(subset=[stat_display], cmap="YlOrRd"),
                                use_container_width=True, 
                                hide_index=True
                            )
                else:
                    st.info("⚠️ Not enough game data available for last 5 games analysis.")
            
            with tab3:
                st.markdown(f"#### 📈 Last 10 Games Performance")
                
                if not combined_logs.empty and len(combined_logs) >= 10:
                    last10_logs = combined_logs.head(10)
                    
                    if stat_code == "PRA":
                        l10_values = last10_logs["PTS"] + last10_logs["REB"] + last10_logs["AST"]
                    else:
                        l10_values = last10_logs[stat_code] if stat_code in last10_logs.columns else pd.Series()
                    
                    if not l10_values.empty:
                        l10_avg = l10_values.mean()
                        
                        # Main display
                        col_main_l10, col_comp_l10 = st.columns([2, 1])
                        with col_main_l10:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                        padding: 20px; border-radius: 8px; color: white;">
                                <h2 style="margin: 0; color: white;">{l10_avg:.1f}</h2>
                                <p style="margin: 5px 0 0 0; opacity: 0.9;">Last 10 Games Average</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_comp_l10:
                            vs_season = l10_avg - season_avg
                            delta_color = "#28a745" if vs_season >= 0 else "#dc3545"
                            delta_emoji = "⬆️" if vs_season >= 0 else "⬇️"
                            st.markdown(f"""
                            <div style="background-color: {delta_color}15; padding: 15px; border-radius: 8px; 
                                        border-left: 4px solid {delta_color}; text-align: center;">
                                <div style="font-size: 12px; color: #666;">vs Season</div>
                                <div style="font-size: 20px; font-weight: bold; color: {delta_color};">
                                    {delta_emoji} {vs_season:+.1f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Stats grid
                        l10_col1, l10_col2, l10_col3 = st.columns(3)
                        with l10_col1:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 12px; background: #fff3cd; border-radius: 6px;">
                                <div style="font-size: 11px; color: #856404;">HIGH</div>
                                <div style="font-size: 20px; font-weight: bold; color: #856404;">{l10_values.max():.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with l10_col2:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 12px; background: #d1ecf1; border-radius: 6px;">
                                <div style="font-size: 11px; color: #0c5460;">LOW</div>
                                <div style="font-size: 20px; font-weight: bold; color: #0c5460;">{l10_values.min():.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with l10_col3:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 12px; background: #d4edda; border-radius: 6px;">
                                <div style="font-size: 11px; color: #155724;">AVG</div>
                                <div style="font-size: 20px; font-weight: bold; color: #155724;">{l10_avg:.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Trend analysis
                        if len(l10_values) >= 10:
                            st.markdown("<br>", unsafe_allow_html=True)
                            first5 = l10_values.tail(5).mean()
                            last5_recent = l10_values.head(5).mean()
                            trend = last5_recent - first5
                            trend_emoji = "📈" if trend >= 0 else "📉"
                            trend_color = "#28a745" if trend >= 0 else "#dc3545"
                            st.markdown(f"""
                            <div style="background: linear-gradient(90deg, {trend_color}15 0%, {trend_color}05 100%); 
                                        padding: 15px; border-radius: 8px; border-left: 4px solid {trend_color};">
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <span style="font-size: 24px;">{trend_emoji}</span>
                                    <div>
                                        <div style="font-weight: bold; color: {trend_color};">
                                            Momentum: {trend:+.1f}
                                        </div>
                                        <div style="font-size: 12px; color: #666;">
                                            Most Recent 5 vs Previous 5 (within L10)
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("⚠️ Not enough game data available for last 10 games analysis.")
            
            # Show blend info with better styling
            st.markdown("<br>", unsafe_allow_html=True)
            wc = features.get("weight_current", 0)
            wp = features.get("weight_prior", 1)
            st.markdown(f"""
            <div style="background-color: #e7f3ff; padding: 12px; border-radius: 6px; border-left: 4px solid #2196F3;">
                <div style="font-size: 12px; color: #666; margin-bottom: 5px;">📊 Data Blend</div>
                <div style="font-weight: bold; color: #1976D2;">
                    {wc*100:.0f}% {cur_season} • {wp*100:.0f}% {prev_season}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 8px; color: white; text-align: center;">
                <h3 style="margin: 0; color: white;">Double-Double Probability</h3>
                <h1 style="margin: 10px 0; color: white;">{prediction:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)

    # Context
    with colCxt:
        st.subheader("🏀 Context")
        rest_days = features.get("rest_days", 3)
        is_b2b = features.get("is_back_to_back", 0)
        st.write(f"**Rest Days:** {rest_days}")
        st.write(f"**Back-to-Back:** {'Yes' if is_b2b else 'No'}")
        st.write(f"**Opponent:** {opponent_abbrev}")

    # ---- Book Line Adjustment section
    st.markdown("---")
    st.subheader("📊 Book Line & Hit Rate")
    
    # Add custom CSS for animations and styling
    add_custom_css()

    # Get player identifier
    player_id = pdata.get("player_id", None)
    if player_id is None or pd.isna(player_id):
        fallback_id = hashlib.md5(f"{player_name}_{team_abbrev}".encode()).hexdigest()[:8]
        player_id = f"fallback_{fallback_id}"
    
    player_identifier = str(player_id)
    stable_id = get_or_create_stable_id(player_identifier, stat_code)
    
    # Generate unique keys for this player/stat combo
    base_key = f"{stable_id}_{stat_code}_{render_index or 0}"
    unique_suffix = uuid.uuid4().hex
    
    # Keys for buttons and inputs
    decrease_key = f"detail_dec_{base_key}_{unique_suffix}"
    increase_key = f"detail_inc_{base_key}_{unique_suffix}"
    reset_key = f"detail_reset_{base_key}_{unique_suffix}"
    manual_key = f"detail_manual_{base_key}_{unique_suffix}"
    step_key = f"step_size_{base_key}_{unique_suffix}"

    
    # Initialize or get adjusted line from session state (safely)
    current_line = get_adjusted_line_value(player_identifier, stat_code, fd_line_val or 0.0)
    
    # Get current step size
    step_size = get_line_step_size()
    
    # Step size selector and reset button (outside form, like matchup board)
    col_step, col_reset = st.columns([2, 3])
    
    with col_step:
        step_options = [0.5, 1.0, 2.0, 5.0]
        try:
            step_index = step_options.index(step_size)
        except ValueError:
            step_index = 0  # Default to 0.5 if step_size is not in options
            step_size = 0.5
            set_line_step_size(step_size)
            
        step_size = st.selectbox(
            "Step Size",
            step_options,
            index=step_index,
            key=f"step_select_{base_key}_{unique_suffix}",
            help="Adjust the increment/decrement step size"
        )
        set_line_step_size(step_size)
    
    with col_reset:
        if st.button("🔄 Reset to Default", 
                      key=f"reset_btn_{base_key}_{unique_suffix}",
                      use_container_width=True):
            reset_adjusted_line_value(player_identifier, stat_code, fd_line_val)
            st.rerun()
    
    # Line adjustment controls - use buttons like matchup board (not form submit buttons)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Get current line value fresh for button handlers
    current_line = get_adjusted_line_value(player_identifier, stat_code, fd_line_val or 0.0)
    
    with col1:
        btn_key_dec = f"detail_dec_{base_key}_{unique_suffix}"
        # Disable ➖ button when line is 0.0 or less
        is_disabled = (current_line is None or current_line <= 0.0)
        if st.button("➖", 
                      key=btn_key_dec, 
                      help=f"Decrease line by {step_size}", 
                      use_container_width=True,
                      disabled=is_disabled):
            if current_line is not None:
                new_value = max(0.0, round(current_line - step_size, 1))  # Prevent negative values
            else:
                new_value = 0.0
            set_adjusted_line_value(player_identifier, stat_code, new_value)
            st.rerun()
    
    with col2:
        if stat_code == "DD":
            st.markdown("**Line:** N/A (DD market)")
        else:
            # Always display current line (show 0.0 instead of "—")
            line_display = float(current_line) if current_line is not None else 0.0
            st.write(f"**{line_display:.1f}**")
            
            # Show original FanDuel line if different and available
            if fd_line_val is not None and current_line is not None and current_line != fd_line_val:
                st.caption(f"Original: {fd_line_val:.1f}")
            
            # Optional: number input for direct editing
            new_line = st.number_input(
                "Edit Line",
                min_value=0.0,
                max_value=100.0,
                step=step_size,
                value=line_display,
                format="%.1f",
                key=f"line_input_{base_key}_{unique_suffix}",
                label_visibility="collapsed"
            )
            
            # Update if changed (ensure non-negative)
            if abs(new_line - line_display) > 0.01:
                new_line = max(0.0, new_line)  # Prevent negative values
                set_adjusted_line_value(player_identifier, stat_code, new_line)
                st.rerun()
    
    with col3:
        btn_key_inc = f"detail_inc_{base_key}_{unique_suffix}"
        if st.button("➕", key=btn_key_inc, help=f"Increase line by {step_size}", use_container_width=True):
            if current_line is not None:
                new_value = round(current_line + step_size, 1)
            else:
                new_value = step_size
            set_adjusted_line_value(player_identifier, stat_code, new_value)
            st.rerun()


    # Note: Manual line input is no longer needed since we always show controls with 0.0 default
    # The number input in col2 above handles manual editing
    
    # Recalculate hit rate and edge based on current (adjusted) line
    # Since we now default to 0.0, check if line is greater than 0
    if current_line is not None and current_line > 0.0 and stat_code != "DD":
        # Use combined logs for hit rate calculation
        combined_logs_for_hit = current_logs if not current_logs.empty else prior_logs
        
        # Calculate hit rates for different time windows
        def calculate_hit_ates(logs, line, stat=stat_code):
            if logs is None or logs.empty or line is None:
                return {"L5": 0, "L10": 0, "Season": 0, "H2H": 0}
                
            stat_col = stat.upper() if stat.upper() in logs.columns else None
            if stat_col is None:
                return {"L5": 0, "L10": 0, "Season": 0, "H2H": 0}
                
            logs = logs.copy()
            logs['hit'] = (logs[stat_col] >= line).astype(int)
            
            l5 = logs.head(5)['hit'].mean() * 100 if len(logs) >= 5 else 0
            l10 = logs.head(10)['hit'].mean() * 100 if len(logs) >= 10 else 0
            season = logs['hit'].mean() * 100 if not logs.empty else 0
            
            h2h_hit_rate = 0
            if 'h2h_history' in pdata and pdata['h2h_history'] is not None and not pdata['h2h_history'].empty:
                h2h_logs = pdata['h2h_history'].copy()
                if stat_col in h2h_logs.columns:
                    h2h_logs['hit'] = (h2h_logs[stat_col] >= line).astype(int)
                    h2h_hit_rate = h2h_logs['hit'].mean() * 100
            
            return {
                "L5": round(l5, 1) if l5 > 0 else 0,
                "L10": round(l10, 1) if l10 > 0 else 0,
                "Season": round(season, 1) if season > 0 else 0,
                "H2H": round(h2h_hit_rate, 1) if h2h_hit_rate > 0 else 0
            }
        
        # Calculate hit rates for different time windows
        hit_rates = calculate_hit_rates(combined_logs_for_hit, current_line)
        
        # Store in session state (safely)
        cache_hit_rate(player_identifier, stat_code, hit_rates["L10"])
        
        # Calculate edge information
        adjusted_edge_str, adjusted_rec_text, adjusted_ou_short = calc_edge(prediction, current_line)
    else:
        hit_rates = {"L5": 0, "L10": 0, "Season": 0, "H2H": 0}
        adjusted_hit_rate = hit_pct_val or 0
        adjusted_edge_str = edge_str
        adjusted_rec_text = rec_text
        adjusted_ou_short = "—"

    # Display line and hit rate information
    st.markdown("### 📊 Performance Metrics")
    
    if stat_code == "DD":
        st.info("Double-double markets don't support line adjustments in this view.")
    elif current_line is None:
        st.warning("No line available for this player/stat.")
    else:
        # Display in two columns: Line/Edge and Hit Rates
        col1, col2 = st.columns(2)
        
        with col1:
            # Line and Edge information
            st.markdown("#### 📏 Line & Edge")
            
            # Show current line in a nice card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 8px; color: white; text-align: center; margin: 10px 0;">
                <div style="font-size: 14px; opacity: 0.9;">CURRENT LINE</div>
                <div style="font-size: 28px; font-weight: bold;">{current_line:.1f}</div>
                {f'<div style="font-size: 12px; opacity: 0.8;">(Original: {fd_line_val:.1f})</div>' 
                  if current_line != fd_line_val and fd_line_val is not None else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Show edge information
            st.markdown(f"**Edge vs Line:** {adjusted_edge_str}")
            st.caption(adjusted_rec_text)
            
            # Show model projection vs line
            if 'prediction' in pdata and pdata['prediction'] is not None:
                proj = pdata['prediction']
                diff = proj - current_line
                diff_pct = (diff / current_line * 100) if current_line > 0 else 0
                
                st.markdown("#### 📈 Projection vs Line")
                st.metric(
                    "Model Projection",
                    f"{proj:.1f}",
                    delta=f"{diff:+.1f} ({diff_pct:+.1f}%)",
                    delta_color=("normal" if diff > 0.5 else "inverse" if diff < -0.5 else "off")
                )
        
        with col2:
            # Hit rate information
            st.markdown("#### 🎯 Hit Rate Analysis")
            
            # Main hit rate display
            hit_rate = hit_rates.get("L10", 0)
            hit_color = "#28a745" if hit_rate >= 50 else "#dc3545" if hit_rate < 30 else "#ffc107"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {hit_color} 0%, {hit_color}dd 100%); 
                        padding: 15px; border-radius: 8px; color: white; text-align: center; margin: 10px 0;">
                <div style="font-size: 14px; opacity: 0.9;">LAST 10 GAMES</div>
                <div style="font-size: 28px; font-weight: bold;">{hit_rate:.0f}%</div>
                <div style="font-size: 12px; opacity: 0.8;">over {current_line:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show hit rate breakdown in a compact format
            st.markdown("##### 📊 Hit Rate Breakdown")
            
            # Create columns for the hit rates
            hr_col1, hr_col2 = st.columns(2)
            
            with hr_col1:
                st.metric("Last 5 Games", f"{hit_rates.get('L5', 0):.0f}%")
                st.metric("Season", f"{hit_rates.get('Season', 0):.0f}%")
            
            with hr_col2:
                st.metric("Last 10 Games", f"{hit_rates.get('L10', 0):.0f}%")
                h2h_hr = hit_rates.get('H2H', 0)
                st.metric("vs Opponent", f"{h2h_hr:.0f}%" if h2h_hr > 0 else "—")
            
            # Show comparison to original line if adjusted
            if current_line != fd_line_val and fd_line_val is not None and hit_pct_val is not None:
                hit_diff = hit_rate - hit_pct_val
                st.caption(
                    f"🔁 vs Original: {hit_diff:+.1f}% "
                    f"(was {hit_pct_val:.0f}% @ {fd_line_val:.1f})"
                )

    # ---- Head to head deep dive
    if h2h_games > 0 and stat_code != "DD":
        st.markdown("---")
        st.subheader(f"🔥 Head-to-Head vs {opponent_abbrev}")

        h2h_avg = features.get(f"h2h_{stat_code}_avg", 0)
        h2h_trend = features.get(f"h2h_{stat_code}_trend", 0)

        colH2H1, colH2H2 = st.columns(2)
        with colH2H1:
            st.markdown("**Average vs Opponent**")
            st.markdown(f"### Avg: {h2h_avg:.1f} ({h2h_games} games)")
            diff = h2h_avg - features.get(f"{stat_code}_avg", 0)
            clr = "green" if diff > 0 else "red"
            st.markdown(f":{clr}[{diff:+.1f} vs season avg]")

        with colH2H2:
            st.markdown("**Recent Trend**")
            if abs(h2h_trend) > 1:
                trending_up = (h2h_trend > 0)
                trend_text = "📈 Trending UP" if trending_up else "📉 Trending DOWN"
                st.markdown(f"### {trend_text}")
                st.markdown(
                    f":{('green' if trending_up else 'red')}[{h2h_trend:+.1f}]"
                )
            else:
                st.markdown("### ➡️ Consistent")

        if not h2h_history.empty:
            st.markdown("**Recent Games vs Opponent:**")
            base_cols = ["GAME_DATE","MATCHUP","PTS","REB","AST","FG3M"]
            show_cols = [c for c in base_cols if c in h2h_history.columns]
            if show_cols:
                h2h_recent = h2h_history.head(5)[show_cols].copy()
                if {"PTS","REB","AST"}.issubset(h2h_recent.columns):
                    h2h_recent["PRA"] = (
                        h2h_recent["PTS"] +
                        h2h_recent["REB"] +
                        h2h_recent["AST"]
                    )
                st.dataframe(h2h_recent, use_container_width=True)

    # ---- Recent game log (last 10)
    st.markdown("---")
    st.subheader("📋 Recent Game Log (Last 10 Games)")

    # stitch 10 most recent between current and prior
    if has_current and len(current_logs) >= 10:
        last10_logs = current_logs.head(10)
        label_season = cur_season
    elif has_current and len(current_logs) < 10:
        need = 10 - len(current_logs)
        last10_logs = pd.concat(
            [current_logs, prior_logs.head(need)], ignore_index=True
        ).head(10)
        label_season = f"{cur_season} + {prev_season}"
    elif has_prior:
        last10_logs = prior_logs.head(10)
        label_season = prev_season
    else:
        last10_logs = pd.DataFrame()
        label_season = "N/A"

    st.caption(f"Showing games from: {label_season}")

    if not last10_logs.empty:
        display_cols = [
            "GAME_DATE","MATCHUP","MIN","PTS","REB","AST",
            "FG3M","FGA","FG_PCT"
        ]
        cols_avail = [c for c in display_cols if c in last10_logs.columns]
        if cols_avail:
            preview_df = last10_logs[cols_avail].copy()
            if {"PTS","REB","AST"}.issubset(preview_df.columns):
                preview_df["PRA"] = (
                    preview_df["PTS"] +
                    preview_df["REB"] +
                    preview_df["AST"]
                )
            st.dataframe(preview_df, use_container_width=True)


# ─────────────────────────────
# Build matchup table + live expanders
# ─────────────────────────────
# ─────────────────────────────
# Build matchup table + live expanders
# ─────────────────────────────
# ─────────────────────────────
# Build matchup table + live expanders
# ─────────────────────────────
def build_matchup_view(
    selected_game: dict,
    stat_code: str,
    stat_display: str,
    cur_season: str,
    prev_season: str,
    model_obj: PlayerPropModel,
    show_only_starters: bool = False,
    player_search_query: str = "",
):
    """
    Stream the matchup board IF we have a selected game.
    If there's no game selected yet, show landing / instructions.
    """

    # If user hasn't picked a game yet
    if not selected_game:
        st.title("🏀 NBA Player Props Projection Model")
        st.markdown(
            "Advanced predictions using historical data, matchup analysis, "
            "and head-to-head history"
        )
        st.markdown("---")
        st.subheader("👋 How to use this tool")
        st.markdown(
            """
1. **Pick a matchup** on the left sidebar under *Select Upcoming Game* 2. **Pick a stat** (*Points*, *Rebounds*, *3PM*, etc.)  
3. Watch the **Matchup Board** fill in player by player  
4. Scroll down and **expand any player** to see the full deep dive (model projection, line vs projection edge, rest days, head-to-head, etc.)

No game is selected yet — choose one in the sidebar to start.
            """
        )
        return

    home_team = selected_game["home"]
    away_team = selected_game["away"]
    game_date = selected_game.get("date", None)
    
    # Debug: Show team abbreviations being used
    print(f"[DEBUG] Selected game: {away_team} @ {home_team} on {game_date}")
    print(f"[DEBUG] Team abbreviations - Home: '{home_team}', Away: '{away_team}'")

    # Title / description
    st.title("🏀 NBA Player Props Projection Model")
    st.markdown(
        "Advanced predictions using historical data, matchup analysis, "
        "and head-to-head history"
    )

    st.subheader("🏟 Matchup Board")
    st.caption(
        "Quick view of projections, sportsbook line, model edge, hit rate, and "
        "defensive matchup for everyone in this game."
    )

    # Fetch all game data from cache or API (checks Firebase first)
    cache_status_placeholder = st.empty()
    with cache_status_placeholder.container():
        st.info("🔄 Checking cache for game data...")
    
    # Always try to load data - error handling is built into the functions
    # Add timeout wrapper with progress updates
    try:
        import time
        import threading
        
        progress_container = st.empty()
        result_container = {'data': None, 'done': False, 'error': None}
        
        def load_data():
            try:
                # NO Streamlit calls in this thread - Streamlit doesn't work in threads!
                result = get_cached_game_data(
                    home_team=home_team,
                    away_team=away_team,
                    cur_season=cur_season,
                    prev_season=prev_season,
                    game_date=game_date,
                    max_age_hours=24
                )
                result_container['data'] = result
            except Exception as e:
                result_container['error'] = e
            finally:
                result_container['done'] = True
        
        # Start loading in thread
        load_thread = threading.Thread(target=load_data, daemon=True)
        load_thread.start()
        
        # Show progress updates every 5 seconds
        start_time = time.time()
        timeout_seconds = 150  # 2.5 minutes max
        progress_messages = [
            ("🔄 Checking cache...", 5),
            ("📊 Fetching rosters...", 15),
            ("🏀 Getting player data...", 30),
            ("⚽ Loading game stats...", 60),
            ("⏳ Still loading... This may take up to 2 minutes", 90),
            ("⚠️ Taking longer than expected...", 120),
        ]
        
        message_idx = 0
        while not result_container['done'] and (time.time() - start_time) < timeout_seconds:
            elapsed = time.time() - start_time
            
            # Update progress message
            if message_idx < len(progress_messages):
                msg_text, threshold = progress_messages[message_idx]
                # Ensure threshold is numeric
                if isinstance(threshold, (int, float)) and elapsed >= float(threshold):
                    progress_container.info(msg_text)
                    message_idx += 1
            
            time.sleep(1)
            load_thread.join(timeout=1)
        
        # Check if timed out
        if not result_container['done']:
            progress_container.warning("⚠️ Request is taking longer than expected. Loading with available data...")
            # Check if we got partial data before timeout
            partial_data = result_container.get('data')
            roster_errors = []
            if partial_data and isinstance(partial_data, dict):
                roster_errors = partial_data.get('_roster_errors', [])
            if not roster_errors:
                roster_errors = [f"Request timed out after 150 seconds"]
            
            game_data = {
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
                '_cache_status': 'TIMEOUT',
                '_roster_errors': roster_errors,
            }
        elif result_container['error']:
            error_obj = result_container['error']
            
            # Format error message properly
            if isinstance(error_obj, Exception):
                error_type = type(error_obj).__name__
                error_msg = str(error_obj) if str(error_obj) else repr(error_obj)
                full_error = f"{error_type}: {error_msg}"
                print(f"[ERROR] Exception in load_data: {full_error}")
                import traceback
                traceback.print_exc()
            else:
                full_error = str(error_obj) if error_obj else "Unknown error"
                print(f"[ERROR] Error in load_data: {full_error}")
            
            progress_container.error(f"❌ Error: {full_error}")
            
            # Try to extract roster errors from error object if it's a dict-like structure
            roster_errors = []
            if isinstance(error_obj, dict):
                roster_errors = error_obj.get('_roster_errors', [])
            elif hasattr(error_obj, '_roster_errors'):
                roster_errors = error_obj._roster_errors
            elif isinstance(error_obj, Exception):
                # If it's an exception, include the full error message
                error_type = type(error_obj).__name__
                error_msg = str(error_obj) if str(error_obj) else repr(error_obj)
                roster_errors = [f"{error_type}: {error_msg}"]
            
            # If no roster errors, use the full error message
            if not roster_errors:
                roster_errors = [full_error if full_error else 'Unknown error occurred during data fetch']
            
            game_data = {
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
                '_cache_status': 'ERROR',
                '_roster_errors': roster_errors,
            }
        else:
            game_data = result_container['data']
            progress_container.empty()  # Clear progress message
    except Exception as e:
        print(f"[ERROR] Unexpected error in get_cached_game_data: {e}")
        import traceback
        traceback.print_exc()
        # Create minimal data structure so app can continue
        game_data = {
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
            '_cache_status': 'ERROR',
            '_roster_errors': [f"Unexpected error: {str(e)}"],
        }
        cache_status_placeholder.warning("⚠️ Some data failed to load, but continuing with available data.")
    
    # Ensure we have a valid game_data dict (should always be true now)
    if not game_data:
        game_data = {
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
            '_cache_status': 'EMPTY',
            '_roster_errors': ['No game data returned from fetch function'],
        }
    
    # Show cache status to user with detailed error info
    cache_status = game_data.get('_cache_status', 'UNKNOWN')
    cache_key = game_data.get('_cache_key', '')
    fetch_time = game_data.get('_fetch_time', 0)
    roster_errors = game_data.get('_roster_errors', [])
    timeout_flag = game_data.get('_timeout', False)
    
    # Store errors before removing metadata
    error_details = {
        'cache_status': cache_status,
        'roster_errors': roster_errors,
        'timeout': timeout_flag,
        'cache_key': cache_key,
        'fetch_time': fetch_time,
    }
    
    if cache_status == 'HIT':
        cache_status_placeholder.success(f"✅ **Data loaded from cache** (instant) | Key: `{cache_key}`")
    elif cache_status == 'MISS':
        cache_status_placeholder.warning(f"🔄 **Data fetched from API** ({fetch_time:.1f}s) | Saved to cache: `{cache_key}`")
        # Show roster errors even for MISS status if rosters are empty
        if roster_errors:
            with st.expander("⚠️ Roster Loading Issues", expanded=True):
                st.warning("**Note:** Game data was fetched, but rosters could not be loaded.")
                st.markdown("**Roster Fetching Errors:**")
                for error in roster_errors:
                    st.error(f"• {error}")
                st.info("💡 The app will continue, but player data may not be available for this matchup.")
    elif cache_status == 'ERROR' or cache_status == 'TIMEOUT' or cache_status == 'EMPTY' or timeout_flag:
        cache_status_placeholder.error(f"❌ **Data load failed** | Status: {cache_status}")
        # Always show error details in an expandable section
        with st.expander("🔍 View Error Details", expanded=True):
            st.error("**Root Cause:** Failed to load player roster data from NBA API")
            
            if roster_errors:
                st.markdown("**Roster Fetching Errors:**")
                for error in roster_errors:
                    st.error(f"• {error}")
            
            if timeout_flag:
                st.error("• Request timed out after 2 minutes")
            
            if cache_status == 'TIMEOUT':
                st.error("• API calls took too long (> 2 minutes)")
            
            st.markdown("**Possible Causes:**")
            st.markdown("""
            1. **NBA API is temporarily unavailable** - Check [NBA API Status](https://stats.nba.com)
            2. **Network issues on Streamlit Cloud** - May be temporary
            3. **API rate limiting** - Too many requests
            4. **Team abbreviations mismatch** - Check if team codes are correct
            
            **Next Steps:**
            - Check Streamlit Cloud logs (Manage app → Logs) for detailed error messages
            - Try refreshing the page in a few minutes
            - Try selecting a different game
            - Check if NBA API is working: https://stats.nba.com
            """)
            
            # Show team abbreviations being used
            st.markdown(f"**Debug Info:**")
            st.code(f"Home Team: {home_team}\nAway Team: {away_team}\nCache Key: {cache_key}\nFetch Time: {fetch_time:.1f}s", language="text")
    else:
        cache_status_placeholder.info(f"ℹ️ **Data loaded** | Status: {cache_status}")
        if roster_errors:
            with st.expander("⚠️ Roster Errors", expanded=False):
                for error in roster_errors:
                    st.warning(f"• {error}")
    
    # Remove cache metadata before processing (but keep errors if needed)
    game_data.pop('_cache_status', None)
    game_data.pop('_cache_key', None)
    game_data.pop('_fetch_time', None)
    game_data.pop('_timeout', None)
    
    # Extract cached data (handle both DataFrame and dict formats)
    home_roster = game_data.get('home_roster')
    if not isinstance(home_roster, pd.DataFrame):
        home_roster = pd.DataFrame(home_roster if home_roster else [])
    
    away_roster = game_data.get('away_roster')
    if not isinstance(away_roster, pd.DataFrame):
        away_roster = pd.DataFrame(away_roster if away_roster else [])
    
    starters_dict = game_data.get('starters_dict', {})
    event_id = game_data.get('event_id')
    odds_data = game_data.get('odds_data', {})
    
    def_vs_pos_df = game_data.get('def_vs_pos_df')
    if not isinstance(def_vs_pos_df, pd.DataFrame):
        def_vs_pos_df = pd.DataFrame(def_vs_pos_df if def_vs_pos_df else [])
    
    team_stats = game_data.get('team_stats')
    if not isinstance(team_stats, pd.DataFrame):
        team_stats = pd.DataFrame(team_stats if team_stats else [])

    if home_roster.empty and away_roster.empty:
        st.warning("⚠️ Couldn't load rosters for this matchup. The app will continue but may have limited functionality.")
        # Create empty roster with all required columns
        combined_roster = pd.DataFrame(columns=["player_id", "full_name", "team_abbrev", "is_starter"])
    else:
        try:
            combined_roster = pd.concat([home_roster, away_roster], ignore_index=True)
            if not combined_roster.empty and "player_id" in combined_roster.columns:
                combined_roster = combined_roster.drop_duplicates(subset=["player_id"])
        except Exception as e:
            print(f"[ERROR] Failed to combine rosters: {e}")
            st.warning("⚠️ Error combining rosters. Using available data.")
            combined_roster = home_roster if not home_roster.empty else away_roster
            if combined_roster.empty:
                combined_roster = pd.DataFrame(columns=["player_id", "full_name", "team_abbrev", "is_starter"])

    # Ensure we have required columns even if rosters are empty
    required_columns = ["player_id", "full_name", "team_abbrev", "is_starter"]
    for col in required_columns:
        if col not in combined_roster.columns:
            combined_roster[col] = None if col != "is_starter" else False

    # Mark starters with error handling - only if we have players
    if not combined_roster.empty and "full_name" in combined_roster.columns:
        try:
            combined_roster["is_starter"] = combined_roster["full_name"].apply(
                lambda name: is_player_starter(name, starters_dict) if pd.notna(name) else False
            )
        except Exception as e:
            print(f"[WARN] Failed to mark starters: {e}")
            combined_roster["is_starter"] = False
    else:
        # Empty roster or missing column - set default
        if "is_starter" not in combined_roster.columns:
            combined_roster["is_starter"] = False
    
    # Filter to only starters if toggle is on - only if we have players
    if show_only_starters and not combined_roster.empty and "is_starter" in combined_roster.columns:
        try:
            combined_roster = combined_roster[combined_roster["is_starter"] == True].copy()
            if combined_roster.empty:
                st.warning("⚠️ No projected starters found for this matchup. Showing all players.")
                # Try to restore from original rosters
                try:
                    if not home_roster.empty or not away_roster.empty:
                        combined_roster = pd.concat([home_roster, away_roster], ignore_index=True)
                        if "player_id" in combined_roster.columns:
                            combined_roster = combined_roster.drop_duplicates(subset=["player_id"])
                        if "full_name" in combined_roster.columns:
                            combined_roster["is_starter"] = combined_roster["full_name"].apply(
                                lambda name: is_player_starter(name, starters_dict) if pd.notna(name) else False
                            )
                        else:
                            combined_roster["is_starter"] = False
                except Exception as e:
                    print(f"[WARN] Failed to restore rosters: {e}")
                    combined_roster = pd.DataFrame(columns=required_columns)
        except Exception as e:
            print(f"[WARN] Failed to filter starters: {e}")

    # Filter by player search query (case-insensitive) - only if we have players
    if player_search_query and not combined_roster.empty and "full_name" in combined_roster.columns:
        try:
            search_lower = player_search_query.lower()
            combined_roster = combined_roster[
                combined_roster["full_name"].str.lower().str.contains(search_lower, na=False)
            ].copy()
            if combined_roster.empty:
                st.info(f"🔍 No players found matching '{player_search_query}'. Try a different search term.")
        except Exception as e:
            print(f"[WARN] Failed to filter by search query: {e}")

    total_players = len(combined_roster)
    
    # Early return if no players - show helpful message
    if total_players == 0:
        st.error("❌ **No player data available for this matchup.**")
        
        # Show detailed explanation in expandable section
        with st.expander("🔍 Why is there no player data?", expanded=True):
            st.markdown("""
            **The app successfully fetched game data, but player rosters could not be loaded.**
            
            **Most likely cause:** The NBA API is timing out or unavailable. Looking at the logs:
            - Game data was fetched successfully (took 120 seconds - near the timeout limit)
            - But roster API calls are failing/timing out
            - This means the NBA stats API (stats.nba.com) is either:
              1. **Too slow** - taking more than 30 seconds per request
              2. **Unavailable** - temporarily down or rate limiting requests
              3. **Blocking requests** - may be blocking requests from Streamlit Cloud
            
            **What you can try:**
            1. **Wait a few minutes** - NBA API issues are often temporary
            2. **Try a different game** - Some matchups may have cached roster data
            3. **Check Streamlit Cloud logs** - Look for specific timeout/error messages
            4. **Try again later** - The API may recover
            
            **Note:** The game data was cached, so once rosters load successfully, they'll be cached too
            and future loads will be much faster.
            """)
            
            # Show roster errors if available
            if roster_errors:
                st.markdown("**Detailed Roster Errors:**")
                for error in roster_errors:
                    st.code(error, language="text")
            
            st.markdown("---")
            st.markdown("""
            **Technical Details:**
            - Cache Status: `MISS` (fetched from API)
            - Fetch Time: 120.0s (hit the timeout limit)
            - Rosters: Empty (NBA API calls timed out)
            - Game Data: Cached successfully
            """)
        
        st.info("💡 **Quick tips:**")
        st.markdown("""
        - APIs are temporarily unavailable
        - Game data hasn't been loaded yet
        - Network issues preventing data fetch
        
        **Try:**
        - Refreshing the page
        - Checking Streamlit Cloud logs
        - Trying a different game
        """)
        return

    # Placeholders for data lists
    table_rows = []
    player_payloads = []
    
    # Status placeholder
    status_placeholder = st.empty()

    # ---
    # Helper function for handling manual line changes from number input
    # ---
    def _handle_manual_line_change(key, player_id, stat_code):
        """Callback to update session state from a number_input."""
        new_value = st.session_state.get(key, 0.0)
        # Ensure non-negative
        new_value = max(0.0, float(new_value))
        set_adjusted_line_value(player_id, stat_code, new_value)

    # ---
    # STAGE 1: DATA COLLECTION LOOP
    # We only build the data lists here. NO st.write, st.button, or UI.
    # Each player is wrapped in try/except so one failure doesn't stop everything.
    # ---
    for idx, prow in combined_roster.iterrows():
        try:
            status_placeholder.write(
                f"Loading {idx+1}/{total_players} players..."
            )
            
            # Safely extract player data with defaults - check if column exists
            if isinstance(prow, pd.Series):
                player_name = prow.get("full_name", "Unknown Player") if "full_name" in prow.index else "Unknown Player"
                pid = prow.get("player_id", None) if "player_id" in prow.index else None
                team_abbrev = prow.get("team_abbrev", home_team) if "team_abbrev" in prow.index else home_team
            else:
                # Fallback for dict-like access
                player_name = prow.get("full_name", "Unknown Player") if hasattr(prow, 'get') else "Unknown Player"
                pid = prow.get("player_id", None) if hasattr(prow, 'get') else None
                team_abbrev = prow.get("team_abbrev", home_team) if hasattr(prow, 'get') else home_team
            
            if pd.isna(player_name) or player_name == "" or player_name == "Unknown Player":
                print(f"[WARN] Skipping player {idx} - missing or invalid name: {player_name}")
                continue
                
            opponent_abbrev = away_team if team_abbrev == home_team else home_team

            if pd.isna(pid) or pid is None:
                fallback_id = hashlib.md5(f"{player_name}_{team_abbrev}".encode()).hexdigest()[:8]
                player_identifier = f"fallback_{fallback_id}"
            else:
                player_identifier = str(pid)

            # position - with error handling
            try:
                player_pos = get_player_position(pid, season=prev_season) if pid else "UNK"
            except Exception as e:
                print(f"[WARN] Failed to get position for {player_name}: {e}")
                player_pos = "UNK"

            # logs - with error handling
            try:
                current_logs = get_player_game_logs_cached_db(
                    pid, player_name, season=cur_season
                ) if pid else pd.DataFrame()
            except Exception as e:
                print(f"[WARN] Failed to get current logs for {player_name}: {e}")
                current_logs = pd.DataFrame()
            
            try:
                prior_logs = get_player_game_logs_cached_db(
                    pid, player_name, season=prev_season
                ) if pid else pd.DataFrame()
            except Exception as e:
                print(f"[WARN] Failed to get prior logs for {player_name}: {e}")
                prior_logs = pd.DataFrame()

            # opponent recent form - with error handling
            try:
                opponent_recent = get_opponent_recent_games(
                    opponent_abbrev,
                    season=prev_season,
                    last_n=10
                )
            except Exception as e:
                print(f"[WARN] Failed to get opponent recent games: {e}")
                opponent_recent = pd.DataFrame()

            # head-to-head - with error handling
            try:
                h2h_history = get_head_to_head_history(
                    pid,
                    opponent_abbrev,
                    seasons=[prev_season, "2023-24"]
                ) if pid else pd.DataFrame()
            except Exception as e:
                print(f"[WARN] Failed to get H2H history for {player_name}: {e}")
                h2h_history = pd.DataFrame()

            # defense rank vs position - with error handling
            try:
                opp_def_rank_info = get_team_defense_rank_vs_position(
                    opponent_abbrev,
                    player_pos,
                    def_vs_pos_df
                )
            except Exception as e:
                print(f"[WARN] Failed to get defense rank: {e}")
                opp_def_rank_info = {"rank": 15, "rating": "Average"}

            # features for model - with error handling
            try:
                feat = build_enhanced_feature_vector(
                    current_logs,
                    opponent_abbrev,
                    team_stats,
                    prior_season_logs=prior_logs,
                    opponent_recent_games=opponent_recent,
                    head_to_head_games=h2h_history,
                    player_position=player_pos
                )
            except Exception as e:
                print(f"[WARN] Failed to build features for {player_name}: {e}")
                # Create default feature vector
                feat = build_enhanced_feature_vector(
                    pd.DataFrame(),
                    opponent_abbrev,
                    pd.DataFrame(),
                    prior_season_logs=pd.DataFrame(),
                    opponent_recent_games=pd.DataFrame(),
                    head_to_head_games=pd.DataFrame(),
                    player_position=player_pos
                )

            # model projection (stat-based) - with error handling
            try:
                if stat_code == "DD":
                    pred_val = model_obj.predict_double_double(feat) * 100.0  # %
                    proj_display = f"{pred_val:.1f}%"
                else:
                    pred_val = model_obj.predict(feat, stat_code)
                    proj_display = f"{pred_val:.1f}"
            except Exception as e:
                print(f"[WARN] Failed to predict for {player_name}: {e}")
                pred_val = 0.0
                proj_display = "N/A"

            # sportsbook line for THIS player/stat from FanDuel odds - with error handling
            try:
                if stat_code == "DD":
                    fd_line_val = None
                else:
                    fd_info = get_player_fanduel_line(player_name, stat_code, odds_data)
                    fd_line_val = fd_info.get("line") if fd_info and isinstance(fd_info, dict) else None
            except Exception as e:
                print(f"[WARN] Failed to get FanDuel line for {player_name}: {e}")
                fd_line_val = None

            # compute edge + hit rate - with error handling
            try:
                if stat_code == "DD":
                    hit_pct_val = None
                    edge_str = "—"
                    rec_text = "Most books don't post DD lines"
                    ou_short = "—"
                else:
                    if fd_line_val is not None and pred_val is not None:
                        hit_pct_val = calc_hit_rate(
                            current_logs if not current_logs.empty else prior_logs,
                            stat_code,
                            fd_line_val,
                            window=10
                        )
                        edge_str, rec_text, ou_short = calc_edge(pred_val, fd_line_val)
                    else:
                        hit_pct_val = None
                        edge_str, rec_text, ou_short = ("—", "No line", "—")
            except Exception as e:
                print(f"[WARN] Failed to calculate edge/hit rate for {player_name}: {e}")
                hit_pct_val = None
                edge_str, rec_text, ou_short = ("—", "Error", "—")

            # Opp Def Rank vs Position w/ emoji color - with error handling
            try:
                rank_num = opp_def_rank_info.get("rank", 15)
                rating_txt = opp_def_rank_info.get("rating", "Average")
                d_emoji = defense_emoji(rank_num)
                opp_def_display = f"{d_emoji} #{rank_num} ({rating_txt})"
            except Exception as e:
                print(f"[WARN] Failed to format defense display: {e}")
                opp_def_display = "🟰 #15 (Average)"

            # Check if player is starter and add ⭐️ - with error handling
            try:
                if not combined_roster.empty and "full_name" in combined_roster.columns:
                    roster_row = combined_roster[combined_roster["full_name"] == player_name]
                    is_starter = roster_row.iloc[0].get("is_starter", False) == True if not roster_row.empty and len(roster_row) > 0 else False
                else:
                    is_starter = False
                player_display_name = f"⭐️ {player_name}" if is_starter else player_name
            except Exception as e:
                print(f"[WARN] Failed to check starter status for {player_name}: {e}")
                is_starter = False
                player_display_name = player_name
            
            # Get or initialize adjusted line from session state (safely)
            # If fd_line_val is None, use 0.0 as the default
            try:
                default_line = fd_line_val if fd_line_val is not None else 0.0
                current_line_table = get_adjusted_line_value(player_identifier, stat_code, default_line)
            except Exception as e:
                print(f"[WARN] Failed to get adjusted line for {player_name}: {e}")
                current_line_table = fd_line_val if fd_line_val is not None else 0.0
            
            logs_for_hits = current_logs if not current_logs.empty else prior_logs

            hit_rate_l5 = hit_rate_l10 = hit_rate_season = None
            try:
                if stat_code != "DD" and pred_val is not None and logs_for_hits is not None and not logs_for_hits.empty:
                    hit_rate_l5 = calc_hit_rate_vs_projection(logs_for_hits, stat_code, pred_val, window=5, denominator=5)
                    hit_rate_l10 = calc_hit_rate_vs_projection(logs_for_hits, stat_code, pred_val, window=10, denominator=20)
                    hit_rate_season = calc_hit_rate_vs_projection(logs_for_hits, stat_code, pred_val)
            except Exception as e:
                print(f"[WARN] Failed to calculate hit rates for {player_name}: {e}")

            try:
                if current_line_table is not None and stat_code != "DD" and pred_val is not None:
                    adjusted_edge_str_table, _, adjusted_ou_short_table = calc_edge(pred_val, current_line_table)
                else:
                    adjusted_edge_str_table = edge_str
                    adjusted_ou_short_table = ou_short
            except Exception as e:
                print(f"[WARN] Failed to calculate adjusted edge for {player_name}: {e}")
                adjusted_edge_str_table = edge_str
                adjusted_ou_short_table = ou_short
            
            # Generate a stable unique ID for this player/stat combination
            try:
                stable_unique_id = get_or_create_stable_id(player_identifier, stat_code)
            except Exception as e:
                print(f"[WARN] Failed to generate stable ID for {player_name}: {e}")
                stable_unique_id = f"{player_identifier}_{stat_code}"

            # Store data for interactive table row
            try:
                table_row_data = {
                    "player_name": player_name,
                    "player_display_name": player_display_name,
                    "player_identifier": player_identifier,
                    "stable_id": stable_unique_id,
                    "team_abbrev": team_abbrev,
                    "team_pos": f"{team_abbrev} · {player_pos}",
                    "proj": proj_display,
                    "proj_value": pred_val,  # Actual numeric value for calculations
                    "fd_line_val": fd_line_val,
                    "current_line": current_line_table,
                    "hit_rate": hit_rate_l10 if hit_rate_l10 is not None else hit_pct_val,
                    "hit_rate_l5": hit_rate_l5,
                    "hit_rate_l10": hit_rate_l10,
                    "hit_rate_season": hit_rate_season,
                    "ou_short": adjusted_ou_short_table,
                    "opp_def": opp_def_display,
                    "stat_code": stat_code,
                    "current_logs": current_logs,
                    "prior_logs": prior_logs,
                }
                table_rows.append(table_row_data)
            except Exception as e:
                print(f"[ERROR] Failed to create table row for {player_name}: {e}")
                continue  # Skip this player

            # prepare data for expander
            try:
                pdata = {
                    "player_name": player_name,
                    "team_abbrev": team_abbrev,
                    "player_pos": player_pos,
                    "opponent_abbrev": opponent_abbrev,
                    "current_logs": current_logs,
                    "prior_logs": prior_logs,
                    "h2h_history": h2h_history,
                    "opp_def_rank": opp_def_rank_info,
                    "features": feat,
                    "prediction": pred_val,
                    "stat_code": stat_code,
                    "stat_display": stat_display,
                    "player_id": player_identifier,
                    "stable_unique_id": stable_unique_id, 
                    "fd_line_val": fd_line_val,
                    "hit_pct_val": hit_pct_val,
                    "edge_str": edge_str,
                    "rec_text": rec_text,
                }
                player_payloads.append(pdata)
            except Exception as e:
                print(f"[ERROR] Failed to create player payload for {player_name}: {e}")
                # Don't skip - continue without this player's detail data
                
        except Exception as e:
            print(f"[ERROR] Failed to process player {idx} ({player_name if 'player_name' in locals() else 'unknown'}): {e}")
            import traceback
            traceback.print_exc()
            continue  # Skip this player and continue with the next one
    
    # End of loop
    status_placeholder.success(f"✅ Loaded all {total_players} players. Rendering UI...")

    # ---
    # STAGE 2: UI RENDERING
    # Now that the loop is done, we draw the UI ONCE.
    # ---

    # Draw the interactive table
    st.markdown("""
    <style>
    .player-table-row {
        padding: 8px;
        border-bottom: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

    header_cols = st.columns([2, 1.5, 1, 1.5, 1, 1, 2, 0.8])
    with header_cols[0]:
        st.markdown("**Player**")
    with header_cols[1]:
        st.markdown("**Team/Pos**")
    with header_cols[2]:
        st.markdown("**Proj**")
    with header_cols[3]:
        st.markdown("**Line**")
    with header_cols[4]:
        st.markdown("**O/U**")
    with header_cols[5]:
        # Compact inline layout: "Hit% L5 L10 Season" on one line
        hit_cols = st.columns([1.2, 1, 1, 1.5])
        with hit_cols[0]:
            st.markdown("**Hit%**")
        with hit_cols[1]:
            # Button-based toggle for L5
            current_view = st.session_state.get("hit_rate_window_view", "L10")
            if st.button("L5", key="hit_toggle_l5", use_container_width=True, 
                        type="primary" if current_view == "L5" else "secondary"):
                st.session_state.hit_rate_window_view = "L5"
                st.rerun()
        with hit_cols[2]:
            # Button-based toggle for L10
            if st.button("L10", key="hit_toggle_l10", use_container_width=True,
                        type="primary" if current_view == "L10" else "secondary"):
                st.session_state.hit_rate_window_view = "L10"
                st.rerun()
        with hit_cols[3]:
            # Button-based toggle for Season
            if st.button("Season", key="hit_toggle_season", use_container_width=True,
                        type="primary" if current_view == "Season" else "secondary"):
                st.session_state.hit_rate_window_view = "Season"
                st.rerun()
    with header_cols[6]:
        st.markdown("**Opp Def**")
    with header_cols[7]:
        st.markdown("**Add**")

    st.markdown("---")

    for idx, row_data in enumerate(table_rows):
        row_cols = st.columns([2, 1.5, 1, 1.5, 1, 1, 2, 0.8])

        with row_cols[0]:
            st.write(row_data["player_display_name"])

        with row_cols[1]:
            st.write(row_data["team_pos"])

        with row_cols[2]:
            st.write(row_data["proj"])

        with row_cols[3]:
            if row_data["stat_code"] == "DD":
                st.write("—")
            else:
                player_identifier = row_data["player_identifier"]
                stable_id_row = row_data["stable_id"]
                current_line = row_data["current_line"]
                
                stable_id_for_key = str(stable_id_row).replace("-", "_").replace(".", "_").replace(" ", "_")
                stat_code_safe = str(row_data['stat_code']).replace("-", "_")
                
                # Tighter columns for a compact "stepper" with manual input
                line_btn_cols = st.columns([1, 1.5, 1])
                
                with line_btn_cols[0]:
                    # ➖ BUTTON
                    btn_key = f"tbl_dec_{stable_id_for_key}_{stat_code_safe}_{idx}"
                    is_disabled = (current_line is None or current_line <= 0.0)
                    st.button(
                        "➖",
                        key=btn_key,
                        help="Decrease by 0.5",
                        disabled=is_disabled,
                        use_container_width=True,
                        on_click=set_adjusted_line_value,  # Use on_click handler
                        args=(  # Pass arguments to the on_click function
                            player_identifier,
                            row_data['stat_code'],
                            max(0.0, current_line - 0.5) if current_line is not None else 0.0
                        )
                    )
                
                with line_btn_cols[1]:
                    # MANUAL SET (Number Input)
                    num_input_key = f"num_in_{stable_id_for_key}_{stat_code_safe}_{idx}"
                    line_value = float(current_line) if current_line is not None else 0.0
                    st.number_input(
                        "Line",
                        min_value=0.0,
                        value=line_value,
                        step=0.5,
                        key=num_input_key,
                        on_change=_handle_manual_line_change,
                        args=(num_input_key, player_identifier, row_data['stat_code']),
                        label_visibility="collapsed",
                        format="%.1f"
                    )
                    # Show original FanDuel line if different and available
                    fd_line_val = row_data.get("fd_line_val")
                    if fd_line_val is not None and current_line is not None and abs(current_line - fd_line_val) > 0.01:
                        st.caption(f"FD: {fd_line_val:.1f}")
                
                with line_btn_cols[2]:
                    # ➕ BUTTON
                    btn_key_inc = f"tbl_inc_{stable_id_for_key}_{stat_code_safe}_{idx}"
                    st.button(
                        "➕",
                        key=btn_key_inc,
                        help="Increase by 0.5",
                        use_container_width=True,
                        on_click=set_adjusted_line_value,  # Use on_click handler
                        args=(  # Pass arguments to the on_click function
                            player_identifier,
                            row_data['stat_code'],
                            (current_line + 0.5) if current_line is not None else 0.5
                        )
                    )

        with row_cols[4]:
            # Show O/U as difference between projection and line
            proj_val = row_data.get("proj_value", 0)
            current_line_ou = row_data.get("current_line")
            if current_line_ou is not None and proj_val is not None:
                diff = proj_val - current_line_ou
                ou_display = f"{diff:+.1f}"
            else:
                ou_display = row_data.get("ou_short", "—")
            st.write(ou_display)

        with row_cols[5]:
            # Get the window selected in the header
            window_to_show = st.session_state.get("hit_rate_window_view", "L10")
            
            # Get the correct hit rate value
            hit_rate_val = None
            if window_to_show == "L5":
                hit_rate_val = row_data.get("hit_rate_l5")
            elif window_to_show == "L10":
                hit_rate_val = row_data.get("hit_rate_l10")
            elif window_to_show == "Season":
                hit_rate_val = row_data.get("hit_rate_season")
            
            # Display the single, relevant hit rate
            if hit_rate_val is not None:
                hit_color = "🟢" if hit_rate_val >= 50 else "🔴" if hit_rate_val < 30 else "🟡"
                st.write(f"{hit_color} **{hit_rate_val:.0f}%**")
            else:
                st.write("—")

        with row_cols[6]:
            st.write(row_data["opp_def"])
        
        with row_cols[7]:
            # This is the "Add to Bet Sheet" button logic
            player_identifier = row_data["player_identifier"]
            stable_id_row = row_data["stable_id"]
            current_line = row_data["current_line"]
            stable_id_for_key = str(stable_id_row).replace("-", "_").replace(".", "_").replace(" ", "_")
            stat_code_safe = str(row_data['stat_code']).replace("-", "_")
            
            # **FIXED KEY** (removed render_count)
            add_btn_key = f"add_bet_{stable_id_for_key}_{stat_code_safe}_{idx}"
            
            button_clicked = st.button("➕", key=add_btn_key, help="Add to Bet Sheet", use_container_width=True)
            
            # This logic block is exactly the same as your file,
            # but now it's guaranteed not to break the main UI loop.
            if button_clicked:
                if 'bet_sheet' not in st.session_state:
                    st.session_state.bet_sheet = []
                
                proj_value = row_data.get("proj_value", 0) or 0
                edge_str = row_data.get("ou_short", "—") or "—"
                player_name = row_data.get("player_name", "Unknown Player") or "Unknown Player"
                team_abbrev = row_data.get("team_abbrev", "") or ""
                stat_code = row_data.get("stat_code", "PTS") or "PTS"
                
                hit_rates = {
                    'L5': row_data.get('hit_rate_l5'),
                    'L10': row_data.get('hit_rate_l10'),
                    'Season': row_data.get('hit_rate_season'),
                }
                if stat_code != "DD" and proj_value is not None:
                    needs_l5 = hit_rates['L5'] is None
                    needs_l10 = hit_rates['L10'] is None
                    needs_season = hit_rates['Season'] is None

                    if needs_l5 or needs_l10 or needs_season:
                        combined_logs = row_data.get("current_logs", pd.DataFrame())
                        if combined_logs.empty:
                            combined_logs = row_data.get("prior_logs", pd.DataFrame())

                        if not combined_logs.empty:
                            if needs_l5:
                                hit_rates['L5'] = calc_hit_rate_vs_projection(combined_logs, stat_code, proj_value, window=5, denominator=5)
                            if needs_l10:
                                hit_rates['L10'] = calc_hit_rate_vs_projection(combined_logs, stat_code, proj_value, window=10, denominator=20)
                            if needs_season:
                                hit_rates['Season'] = calc_hit_rate_vs_projection(combined_logs, stat_code, proj_value)
                
                line_source = 'manual'
                if current_line is not None:
                    fd_line = row_data.get("fd_line_val")
                    if fd_line is not None:
                        line_source = 'adjusted' if abs(current_line - fd_line) > 0.01 else 'scraped'
                
                if current_line is not None and proj_value:
                    try:
                        edge_str, _, _ = calc_edge(proj_value, current_line)
                    except:
                        pass
                
                bet_id = get_bet_sheet_item_id(player_identifier, stat_code, current_line)
                
                bet_data = {
                    'id': bet_id,
                    'player_name': player_name,
                    'player_id': player_identifier,
                    'team_abbrev': team_abbrev,
                    'stat_code': stat_code,
                    'stat_display': stat_display,
                    'projection': proj_value,
                    'line': current_line,
                    'line_source': line_source,
                    'edge': edge_str,
                    'stake': None,
                    'hit_rates': hit_rates,
                    'hit_rate_window': st.session_state.bet_sheet_settings.get('default_hit_rate_window', 'L10') if 'bet_sheet_settings' in st.session_state else 'L10',
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                if 'bet_sheet' not in st.session_state:
                    st.session_state.bet_sheet = []
                
                st.session_state.bet_sheet.append(bet_data)
                
                # Save to file
                save_bet_sheet_to_file()
                
                current_count = len(st.session_state.bet_sheet)
                
                if current_line is not None:
                    toast_msg = f"Added {player_name} ({stat_display} {current_line:+.1f}) - Count: {current_count}"
                else:
                    toast_msg = f"Added {player_name} ({stat_display}) - Count: {current_count}"
                
                st.session_state.bet_sheet_toast = {
                    'type': 'success',
                    'message': toast_msg
                }
                
                st.rerun()

    # ---
    # Draw the expanders
    # ---
    st.markdown("---")
    st.subheader("📂 Player Breakdowns (click any name below to expand)")
    st.caption(
        f"You can start opening players right away. "
        f"({cur_season} vs {prev_season}, matchup context, line edge, trends, etc.)"
    )

    rendered_combinations = set()
    for idx, info in enumerate(player_payloads):
        player_id_check = info.get("player_id", None)
        stat_code_check = info.get("stat_code", "")

        if player_id_check is None:
            continue

        combo_key = (player_id_check, stat_code_check)
        if combo_key in rendered_combinations:
            continue

        rendered_combinations.add(combo_key)

        try:
            if not combined_roster.empty and "full_name" in combined_roster.columns:
                roster_match = combined_roster[combined_roster["full_name"] == info["player_name"]]
                player_is_starter = roster_match.iloc[0].get("is_starter", False) == True if not roster_match.empty and len(roster_match) > 0 else False
            else:
                player_is_starter = False
        except Exception as e:
            print(f"[WARN] Failed to check starter status: {e}")
            player_is_starter = False

        expander_title = f"{'⭐️ ' if player_is_starter else ''}{info['player_name']} ({info['team_abbrev']} · {info['player_pos']})"

        with st.expander(expander_title, expanded=False):
            # Pass the render_index to ensure unique keys inside the expander
            render_player_detail_body(info, cur_season, prev_season, render_index=idx)

    # final status
    status_placeholder.success("✅ Done.")
def main():
    # Initialize database on startup (creates tables if they don't exist)
    try:
        from utils.database import init_database
        init_database()
    except Exception as e:
        print(f"[WARN] Database initialization issue: {e}. App will continue but caching may be limited.")
    
    # Initialize Firebase cache if available
    try:
        from utils import firebase_cache
        firebase_cache.initialize_firebase()
    except Exception as e:
        print(f"[WARN] Could not initialize Firebase: {e}")
    
    initialize_session_state_defaults()
    initialize_bet_sheet()

    # ─────────────────────────────
    # Top Navigation Area
    # ─────────────────────────────
    
    # Bet Sheet Expander at the top
    bet_sheet_list = st.session_state.get('bet_sheet', [])
    bet_count = len(bet_sheet_list) if bet_sheet_list else 0
    with st.expander(f"📋 Bet Sheet ({bet_count})", expanded=False):
        render_bet_sheet_area(current_season, prior_season)
    
    st.markdown("---")
    
    # Title
    st.title("🏀 NBA Player Props Projection Model")
    
    # Settings & Filters Header
    st.header("⚙️ Settings & Filters")
    
    # Create columns for navigation controls
    nav_cols = st.columns([3, 2, 2, 3])
    
    # Column 1: Game Selection
    with nav_cols[0]:
        st.subheader("📅 Select Upcoming Game")
        upcoming_games = get_upcoming_games(days=7)
        selected_game = None
        game_map = {}

        if upcoming_games:
            sidebar_options = ["-- Select a Game --"]
            for g in upcoming_games:
                # ex: "Sat, Oct 25 - CHI @ ORL (7:30 PM)"
                date_disp = g.get("date_display", "")
                tm = f" ({g['time_display']})" if g.get("time_display") else ""
                label = f"{date_disp} - {g['away']} @ {g['home']}{tm}"
                sidebar_options.append(label)
                game_map[label] = g

            picked_label = st.selectbox(
                f"Upcoming games (next 3 days) - {len(upcoming_games)} found",
                options=sidebar_options,
                index=0,  # default to instruction line
            )

            if picked_label != "-- Select a Game --":
                selected_game = game_map[picked_label]
                st.info(f"Matchup: {selected_game['away']} @ {selected_game['home']}")
        else:
            st.warning("⚠️ No upcoming games in next 7 days.")
            picked_label = None
            selected_game = None
    
    # Column 2: Stat Selection
    with nav_cols[1]:
        st.subheader("📊 Stat to Project")
        stat_display_list = list(STAT_OPTIONS.keys())
        stat_display_choice = st.selectbox(
            "Choose stat to preview on the board",
            options=stat_display_list,
            index=0,
            label_visibility="collapsed"
        )
        stat_code_choice = STAT_OPTIONS[stat_display_choice]
    
    # Column 3: Player Filter
    with nav_cols[2]:
        st.subheader("👥 Player Filter")
        show_only_starters = st.radio(
            "Filter players",
            options=["Show All Players", "Show Only Starters"],
            index=0,
            help="Filter to show only projected starters (⭐) or all players",
            label_visibility="collapsed"
        )
    
    # Column 4: Search Player
    with nav_cols[3]:
        st.subheader("🔍 Search Player")
        player_search_query = st.text_input(
            "Search by player name",
            value="",
            placeholder="Type player name...",
            help="Filter players by name (case-insensitive)",
            label_visibility="collapsed"
        )
    
    st.markdown("---")

    # ─────────────────────────────
    # Manual Entry Modal (if needed)
    # ─────────────────────────────
    
    # Handle manual entry modal
    if st.session_state.get('bet_sheet_show_manual_modal', False):
        manual_entry = st.session_state.get('bet_sheet_manual_entry')
        if manual_entry:
            with st.expander("✏️ Enter Line Manually", expanded=True):
                st.info(f"**{manual_entry['player_data']['player_name']}** - {manual_entry['stat_display']}")
                st.caption("No line available. Please enter a line manually to add to bet sheet.")
                
                manual_line = st.number_input(
                    "Enter line",
                    min_value=0.0,
                    step=0.5,
                    key="manual_line_input"
                )
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ Add to Bet Sheet", key="confirm_manual_line", use_container_width=True):
                        if manual_line > 0:
                            # Calculate hit rates if we have game logs
                            player_id = manual_entry['player_data']['player_id']
                            player_name = manual_entry['player_data']['player_name']
                            stat_code = manual_entry['player_data']['stat_code']
                            
                            # Get game logs for hit rate calculation
                            current_logs = get_player_game_logs_cached_db(player_id, player_name, current_season) if player_name else pd.DataFrame()
                            if current_logs is None or current_logs.empty:
                                prior_logs = get_player_game_logs_cached_db(player_id, player_name, prior_season) if player_name else pd.DataFrame()
                                combined_logs = prior_logs if not prior_logs.empty else pd.DataFrame()
                            else:
                                combined_logs = current_logs
                            
                            hit_rates = {}
                            projection_value = manual_entry.get('projection')
                            if projection_value is not None and not combined_logs.empty:
                                hit_rates['L5'] = calc_hit_rate_vs_projection(combined_logs, stat_code, projection_value, window=5, denominator=5)
                                hit_rates['L10'] = calc_hit_rate_vs_projection(combined_logs, stat_code, projection_value, window=10, denominator=20)
                                hit_rates['Season'] = calc_hit_rate_vs_projection(combined_logs, stat_code, projection_value)
                            
                            # Calculate edge
                            projection_for_edge = manual_entry.get('projection')
                            edge_str, _, _ = calc_edge(projection_for_edge, manual_line) if projection_for_edge is not None else ("—", "—", "—")
                            
                            add_to_bet_sheet(
                                player_data=manual_entry['player_data'],
                                stat_display=manual_entry['stat_display'],
                                projection=projection_for_edge,
                                line=manual_line,
                                edge=edge_str,
                                line_source='manual',
                                hit_rates=hit_rates,
                                allow_duplicate=st.session_state.bet_sheet_settings.get('allow_duplicates', False)
                            )
                            st.session_state.bet_sheet_show_manual_modal = False
                            st.session_state.bet_sheet_manual_entry = None
                            st.rerun()
                
                with col_btn2:
                    if st.button("❌ Cancel", key="cancel_manual_line", use_container_width=True):
                        st.session_state.bet_sheet_show_manual_modal = False
                        st.session_state.bet_sheet_manual_entry = None
                        st.rerun()
    
    # ─────────────────────────────
    # Main render call
    # ─────────────────────────────
    build_matchup_view(
        selected_game=selected_game,
        stat_code=stat_code_choice,
        stat_display=stat_display_choice,
        cur_season=current_season,
        prev_season=prior_season,
        model_obj=model,
        show_only_starters=(show_only_starters == "Show Only Starters"),
        player_search_query=player_search_query.strip() if player_search_query else "",
    )

    # footer
    st.markdown("---")
    st.markdown(
        "**Data Sources:** NBA.com (via nba_api) | "
        "**Model:** Enhanced Ridge Regression  \n"
        "**Features:** Season blending, H2H history, opponent recent form, "
        "positional defense  \n"
        "**Sportsbook Lines:** FanDuel via The Odds API  \n"
        "**Note:** Projections are informational only. "
        "Always verify lines and context."
    )
    
    # ─────────────────────────────
    # Cache Stats (moved from sidebar)
    # ─────────────────────────────
    st.markdown("---")
    with st.expander("💾 Cache Stats", expanded=False):
        cache_stats = get_cache_stats()
        st.write(f"**Players cached:** {cache_stats.get('total_players', 0)}")
        st.write(f"**Games cached:** {cache_stats.get('total_games', 0):,}")
        st.write(f"**DB Size:** {cache_stats.get('db_size_mb', 0):.1f} MB")

        if st.button("🗑️ Clear Old Seasons"):
            clear_old_seasons([current_season, prior_season])
            st.success("Old seasons cleared!")
            st.rerun()

def render_bet_sheet_area(cur_season, prev_season):
    """
    Render the bet sheet as a collapsible area in the main page.
    """
    # FORCE initialize - make absolutely sure it exists
    if 'bet_sheet' not in st.session_state:
        st.session_state.bet_sheet = []
    if 'bet_sheet_settings' not in st.session_state:
        st.session_state.bet_sheet_settings = {
            'allow_duplicates': False,
            'line_step_size': 0.5,
            'default_hit_rate_window': 'L10'
        }
    if 'bet_sheet_toast' not in st.session_state:
        st.session_state.bet_sheet_toast = None
    
    add_custom_css()
    
    # Get count - make sure we're reading the actual list
    bet_sheet_list = st.session_state.get('bet_sheet', [])
    bet_count = len(bet_sheet_list) if bet_sheet_list else 0
    
    # Render toast notification if present
    if st.session_state.bet_sheet_toast:
        toast = st.session_state.bet_sheet_toast
        toast_class = f"toast-{toast['type']}"
        st.markdown(f"""
        <div class="toast-notification {toast_class}">
            {toast['message']}
        </div>
        <script>
            setTimeout(function() {{
                var toast = document.querySelector('.toast-notification');
                if (toast) toast.style.display = 'none';
            }}, 3000);
        </script>
        """, unsafe_allow_html=True)
        st.session_state.bet_sheet_toast = None
    
    # Bet Sheet section - render directly (expander is handled in main function)
    if bet_count == 0:
        st.info("Your bet sheet is empty. Add players from the matchup board using the ➕ button.")
    else:
        # Check if we're editing a specific item
        editing_id = st.session_state.get('bet_sheet_editing')
        if editing_id:
            editing_bet = next((b for b in st.session_state.bet_sheet if b.get('id') == editing_id), None)
            if editing_bet:
                with st.expander("✏️ Edit Bet Item", expanded=True):
                    render_bet_item_edit_modal(editing_bet, cur_season, prev_season)
                st.markdown("---")
        
        # Sort and filter controls
        sort_option = st.selectbox(
            "Sort by",
            ["Player Name", "Hit Rate", "Line", "Stat"],
            key="bet_sheet_sort"
        )
        
        # Bulk controls
        col_bulk1, col_bulk2 = st.columns(2)
        
        # Check if we should show the confirmation buttons
        if st.session_state.get('bet_sheet_show_clear_confirm', False):
            st.warning("⚠️ Are you sure you want to clear all bets?")
            # Show confirmation buttons
            with col_bulk1:
                if st.button("✅ Yes, Clear All", key="confirm_clear_bets", use_container_width=True, type="primary"):
                    clear_bet_sheet()
                    st.session_state.bet_sheet_show_clear_confirm = False
                    st.rerun()
            with col_bulk2:
                if st.button("❌ Cancel", key="cancel_clear_bets", use_container_width=True):
                    st.session_state.bet_sheet_show_clear_confirm = False
                    st.rerun()
        else:
            # Show the normal "Clear All" and "Export" buttons
            with col_bulk1:
                if st.button("🗑️ Clear All", key="clear_all_bets", use_container_width=True):
                    # Don't clear yet! Just set the flag to show confirmation
                    st.session_state.bet_sheet_show_clear_confirm = True
                    st.rerun()
            
            with col_bulk2:
                csv_data = export_bet_sheet_csv()
                if csv_data:
                    st.download_button(
                        label="📥 Export",
                        data=csv_data,
                        file_name=f"bet_sheet_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="export_bet_sheet",
                        use_container_width=True
                    )
        
        st.markdown("---")
        
        # Sort bets - use the actual list from session state
        sorted_bets = bet_sheet_list.copy() if bet_sheet_list else []
        if sort_option == "Player Name":
            sorted_bets.sort(key=lambda x: x['player_name'])
        elif sort_option == "Hit Rate":
            sorted_bets.sort(key=lambda x: x.get('hit_rates', {}).get('L10', 0) or 0, reverse=True)
        elif sort_option == "Line":
            sorted_bets.sort(key=lambda x: x.get('line', 0) or 0)
        elif sort_option == "Stat":
            sorted_bets.sort(key=lambda x: x['stat_display'])
        
        # Render each bet item (skip the one being edited)
        for idx, bet in enumerate(sorted_bets):
            if bet.get('id') != editing_id:
                render_bet_item(bet, cur_season, prev_season, index=idx)


def render_bet_item(bet, cur_season, prev_season, index=None):
    """Render a single bet item row in the bet sheet."""
    bet_id = bet.get('id', '')
    player_name = bet['player_name']
    team = bet['team_abbrev']
    stat_display = bet['stat_display']
    
    # Use index to ensure unique keys even if bet_id is duplicated
    unique_key_suffix = f"_{index}" if index is not None else f"_{bet_id}"
    
    # Get current line value fresh from bet sheet (in case it was updated)
    current_bet = next((b for b in st.session_state.bet_sheet if b.get('id') == bet_id), bet)
    line = current_bet.get('line')
    projection = current_bet.get('projection', 0)
    edge = current_bet.get('edge', '—')
    stake = current_bet.get('stake')
    hit_rates = current_bet.get('hit_rates', {})
    hit_rate_window = current_bet.get('hit_rate_window', 'L10')
    
    # Get current hit rate
    current_hit_rate = hit_rates.get(hit_rate_window)
    
    # Bet item container
    with st.container():
        st.markdown(f"""
        <div class="bet-sheet-item">
        """, unsafe_allow_html=True)
        
        # Header row: Player name and remove button
        col_header1, col_header2 = st.columns([4, 1])
        with col_header1:
            st.markdown(f"**{player_name}** ({team})")
        with col_header2:
            if st.button("🗑️", key=f"remove_{bet_id}{unique_key_suffix}", help="Remove"):
                remove_from_bet_sheet(bet_id)
        
        # Stat display
        st.markdown(f"**{stat_display}**")
        
        # Line input/adjustment controls
        if line is None:
            # Show number input for entering line when line is None
            manual_line_key = f"manual_line_{bet_id}{unique_key_suffix}"
            new_line = st.number_input(
                "Enter Line",
                min_value=0.0,
                value=0.0,
                step=0.5,
                key=manual_line_key,
                label_visibility="visible"
            )
            # Add apply button for manual entry
            if st.button("✅ Apply Line", key=f"apply_manual_{bet_id}{unique_key_suffix}", use_container_width=True):
                if new_line > 0:
                    new_line = max(0.0, new_line)  # Ensure non-negative
                    update_bet_sheet_item(bet_id, {'line': new_line, 'line_source': 'manual'}, cur_season, prev_season)
        else:
            # Show +/- adjustment buttons when line exists - match matchup board pattern
            col_line1, col_line2, col_line3 = st.columns([1, 2, 1])
            step_size = st.session_state.bet_sheet_settings.get('line_step_size', 0.5)
            
            # Get current line value fresh for button handlers
            current_line = next((b.get('line') for b in st.session_state.bet_sheet if b.get('id') == bet_id), line)
            
            with col_line1:
                btn_key_dec = f"betsheet_dec_{bet_id}{unique_key_suffix}"
                # Disable ➖ button when line is 0.0 or less
                is_disabled = (current_line is None or current_line <= 0.0)
                if st.button("➖", key=btn_key_dec, help=f"Decrease by {step_size}", use_container_width=True, disabled=is_disabled):
                    if current_line is not None:
                        new_line = max(0.0, round(current_line - step_size, 1))  # Prevent negative values
                    else:
                        new_line = 0.0
                    update_bet_sheet_item(bet_id, {'line': new_line}, cur_season, prev_season)
            
            with col_line2:
                # Display current line value (show 0.0 instead of "—")
                line_display = float(current_line) if current_line is not None else 0.0
                st.write(f"**{line_display:.1f}**")
                # Also allow manual editing via number input
                edited_line = st.number_input(
                    "Line",
                    min_value=0.0,
                    value=float(current_line) if current_line is not None else 0.0,
                    step=step_size,
                    key=f"edit_line_{bet_id}{unique_key_suffix}",
                    label_visibility="collapsed"
                )
                # Update if line changed (ensure non-negative)
                if current_line is not None and abs(edited_line - current_line) > 0.01:
                    edited_line = max(0.0, edited_line)  # Prevent negative values
                    update_bet_sheet_item(bet_id, {'line': edited_line}, cur_season, prev_season)
            
            with col_line3:
                btn_key_inc = f"betsheet_inc_{bet_id}{unique_key_suffix}"
                if st.button("➕", key=btn_key_inc, help=f"Increase by {step_size}", use_container_width=True):
                    if current_line is not None:
                        new_line = round(current_line + step_size, 1)
                        update_bet_sheet_item(bet_id, {'line': new_line}, cur_season, prev_season)
        
        # Hit rate display
        if current_hit_rate is not None:
            hit_color = "🟢" if current_hit_rate >= 50 else "🔴" if current_hit_rate < 30 else "🟡"
            st.markdown(f"{hit_color} **{current_hit_rate:.0f}%** ({hit_rate_window})")
        else:
            st.markdown("Hit Rate: —")
        
        # Hit rate window toggle
        window_options = ['L5', 'L10', 'Season']
        current_idx = window_options.index(hit_rate_window) if hit_rate_window in window_options else 1
        new_window = st.radio(
            "Window",
            window_options,
            index=current_idx,
            horizontal=True,
            key=f"window_{bet_id}{unique_key_suffix}",
            label_visibility="collapsed"
        )
        if new_window != hit_rate_window:
            update_bet_sheet_item(bet_id, {'hit_rate_window': new_window}, cur_season, prev_season)
        
        # Stake input
        new_stake = st.number_input(
            "Stake",
            min_value=0.0,
            value=float(stake) if stake else 0.0,
            step=1.0,
            key=f"stake_{bet_id}{unique_key_suffix}",
            label_visibility="visible"
        )
        if abs(new_stake - (stake if stake else 0.0)) > 0.01:  # Use tolerance for float comparison
            update_bet_sheet_item(bet_id, {'stake': new_stake if new_stake > 0 else None}, cur_season, prev_season)
        
        # Edit button
        if st.button("✏️ Edit", key=f"edit_{bet_id}{unique_key_suffix}", use_container_width=True):
            st.session_state.bet_sheet_editing = bet_id
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")


def render_bet_item_edit_modal(bet, cur_season, prev_season):
    """Render the edit modal for a bet item."""
    bet_id = bet.get('id', '')
    
    st.markdown("### ✏️ Edit Bet Item")
    
    # Player info (read-only)
    st.info(f"**{bet['player_name']}** ({bet['team_abbrev']}) - {bet['stat_display']}")
    
    # Line editing with +/- buttons - get fresh value like matchup board
    st.markdown("#### Line Adjustment")
    # Get current line value fresh from bet sheet
    current_bet = next((b for b in st.session_state.bet_sheet if b.get('id') == bet_id), bet)
    current_line = current_bet.get('line', 0)
    step_size = st.session_state.bet_sheet_settings.get('line_step_size', 0.5)
    
    col_line1, col_line2, col_line3 = st.columns([1, 2, 1])
    with col_line1:
        btn_key_dec = f"modal_dec_{bet_id}"
        # Disable ➖ button when line is 0.0 or less
        is_disabled = (current_line is None or current_line <= 0.0)
        if st.button("➖", key=btn_key_dec, help=f"Decrease by {step_size}", use_container_width=True, disabled=is_disabled):
            if current_line is not None:
                new_line = max(0.0, round(current_line - step_size, 1))  # Prevent negative values
            else:
                new_line = 0.0
            update_bet_sheet_item(bet_id, {'line': new_line}, cur_season, prev_season)
    
    with col_line2:
        # Display current line (show 0.0 instead of "—")
        line_display = float(current_line) if current_line is not None else 0.0
        st.write(f"**{line_display:.1f}**")
        new_line_input = st.number_input(
            "Line",
            min_value=0.0,
            value=float(current_line) if current_line else 0.0,
            step=step_size,
            key=f"modal_line_{bet_id}",
            label_visibility="collapsed"
        )
        if current_line is not None and abs(new_line_input - current_line) > 0.01:
            new_line_input = max(0.0, new_line_input)  # Prevent negative values
            update_bet_sheet_item(bet_id, {'line': new_line_input}, cur_season, prev_season)
    
    with col_line3:
        btn_key_inc = f"modal_inc_{bet_id}"
        if st.button("➕", key=btn_key_inc, help=f"Increase by {step_size}", use_container_width=True):
            if current_line is not None:
                new_line = round(current_line + step_size, 1)
                update_bet_sheet_item(bet_id, {'line': new_line}, cur_season, prev_season)
    
    # Hit rate window selection
    st.markdown("#### Hit Rate Window")
    window_options = ['L5', 'L10', 'Season']
    current_window = bet.get('hit_rate_window', 'L10')
    new_window = st.radio(
        "Select window",
        window_options,
        index=window_options.index(current_window) if current_window in window_options else 1,
        key=f"modal_window_{bet_id}"
    )
    if new_window != current_window:
        update_bet_sheet_item(bet_id, {'hit_rate_window': new_window})
        st.rerun()
    
    # Display hit rate for current window
    hit_rates = bet.get('hit_rates', {})
    current_hit_rate = hit_rates.get(new_window)
    if current_hit_rate is not None:
        st.metric("Hit Rate", f"{current_hit_rate:.1f}%")
    else:
        st.info("Hit rate not available for this window")
    
    # Stake input
    st.markdown("#### Stake")
    current_stake = bet.get('stake')
    new_stake = st.number_input(
        "Stake amount",
        min_value=0.0,
        value=float(current_stake) if current_stake else 0.0,
        step=1.0,
        key=f"modal_stake_{bet_id}"
    )
    if new_stake != current_stake:
        update_bet_sheet_item(bet_id, {'stake': new_stake if new_stake > 0 else None})
    
    # Sync to player checkbox (optional feature)
    sync_to_player = st.checkbox(
        "Sync line changes to player state",
        value=False,
        key=f"sync_{bet_id}",
        help="If checked, line changes will also update the player's line in the main view"
    )
    
    # Action buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("💾 Save", key=f"save_{bet_id}", use_container_width=True, type="primary"):
            if sync_to_player:
                # Update player's adjusted line in session state
                player_id = bet['player_id']
                stat_code = bet['stat_code']
                set_adjusted_line_value(player_id, stat_code, new_line_input)
            st.session_state.bet_sheet_editing = None
            st.session_state.bet_sheet_toast = {
                'type': 'success',
                'message': 'Bet item updated'
            }
            st.rerun()
    
    with col_btn2:
        if st.button("❌ Cancel", key=f"cancel_{bet_id}", use_container_width=True):
            st.session_state.bet_sheet_editing = None
            st.rerun()
    
    # Show current values summary
    st.markdown("---")
    st.markdown("#### Current Values")
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric("Line", f"{new_line_input:.1f}")
    with col_sum2:
        st.metric("Stake", f"${new_stake:.0f}" if new_stake > 0 else "—")
    with col_sum3:
        hr_display = f"{current_hit_rate:.1f}%" if current_hit_rate is not None else "—"
        st.metric(f"Hit Rate ({new_window})", hr_display)


def render_bet_sheet():
    """Render the bet sheet as a full page (for navigation)."""
    st.title("📋 Your Bet Sheet")
    
    initialize_bet_sheet()
    
    # Back button
    if st.button("⬅️ Back to Matchup Board"):
        st.session_state.current_page = 'matchup_board'
        st.rerun()
    
    st.markdown("---")
    
    if not st.session_state.bet_sheet:
        st.info("Your bet sheet is empty. Add players from the Matchup Board.")
        return
    
    # Check if we're editing a specific item
    editing_id = st.session_state.get('bet_sheet_editing')
    if editing_id:
        editing_bet = next((b for b in st.session_state.bet_sheet if b.get('id') == editing_id), None)
        if editing_bet:
            with st.expander("✏️ Edit Bet Item", expanded=True):
                render_bet_item_edit_modal(editing_bet, current_season, prior_season)
            st.markdown("---")
    
    # Display all bets
    for idx, bet in enumerate(st.session_state.bet_sheet):
        render_bet_item(bet, current_season, prior_season, index=idx)

if __name__ == "__main__":
    main()
