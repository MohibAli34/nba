import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
import uuid
import hashlib
from datetime import datetime
from copy import deepcopy

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

def calc_hit_rate(game_logs: pd.DataFrame, stat_col: str, line_value: float, window: int = 10):
    """
    % of last `window` games the player went OVER line_value for stat_col.
    If not enough data or no line, return None.
    """
    if game_logs is None or game_logs.empty:
        return None
    if line_value is None:
        return None

    # last N games (most recent at head() because nba_api returns reverse-chronological)
    recent = game_logs.head(window).copy()

    if stat_col == "PRA":
        if not {"PTS", "REB", "AST"}.issubset(recent.columns):
            return None
        recent_vals = recent["PTS"] + recent["REB"] + recent["AST"]
    else:
        if stat_col not in recent.columns:
            return None
        recent_vals = recent[stat_col]

    if len(recent_vals) == 0:
        return None

    hits = (recent_vals > line_value).sum()
    rate = hits / len(recent_vals)
    return rate * 100.0


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
    Safely return the current Streamlit session_state object or None if
    Streamlit has not finished initialising it yet.
    """
    try:
        return st.session_state  # type: ignore[attr-defined]
    except Exception:
        return None


def safe_session_state_get(key, default_value=None):
    """
    Safely get a value from session state, handling initialization errors.
    """
    session_state = _get_session_state()
    if session_state is None:
        return deepcopy(default_value)

    try:
        return session_state.get(key, deepcopy(default_value))
    except Exception:
        return deepcopy(default_value)


def safe_session_state_set(key, value):
    """
    Safely set a value in session state, handling initialization errors.
    """
    session_state = _get_session_state()
    if session_state is None:
        return

    try:
        session_state[key] = value
    except Exception:
        # Session state not ready, skip silently
        pass


def safe_session_state_init(key, default_value):
    """
    Initialise a session state key with a default value if it does not
    already exist, returning the stored value (or a deepcopy of the
    default when session state is unavailable).
    """
    session_state = _get_session_state()
    if session_state is None:
        return deepcopy(default_value)

    if key not in session_state:
        session_state[key] = deepcopy(default_value)

    return session_state[key]


SESSION_DEFAULTS = {
    "adjusted_lines": {},
    "manual_lines": {},
    "hit_rates": {},
    "stable_ids": {},
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
    adjusted_lines = safe_session_state_init("adjusted_lines", {})
    key = make_player_stat_key(player_identifier, stat_code)
    if default_value is not None and key not in adjusted_lines:
        adjusted_lines[key] = default_value
    return adjusted_lines.get(key, default_value)


def set_adjusted_line_value(player_identifier, stat_code, value):
    adjusted_lines = safe_session_state_init("adjusted_lines", {})
    key = make_player_stat_key(player_identifier, stat_code)
    adjusted_lines[key] = value


def reset_adjusted_line_value(player_identifier, stat_code):
    adjusted_lines = safe_session_state_init("adjusted_lines", {})
    key = make_player_stat_key(player_identifier, stat_code)
    if key in adjusted_lines:
        del adjusted_lines[key]


def get_manual_line_value(player_identifier, stat_code, default_value=0.0):
    manual_lines = safe_session_state_init("manual_lines", {})
    key = make_player_stat_key(player_identifier, stat_code)
    if key not in manual_lines:
        manual_lines[key] = default_value
    return manual_lines[key]


def set_manual_line_value(player_identifier, stat_code, value):
    manual_lines = safe_session_state_init("manual_lines", {})
    key = make_player_stat_key(player_identifier, stat_code)
    manual_lines[key] = value


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



def render_player_detail_body(pdata, cur_season, prev_season, render_index=None):
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

    # ---- Sportsbook Line / Hit Rate section
    st.markdown("---")
    st.subheader("📊 Sportsbook Line & Hit Rate")

    # Get player_id from pdata (required for unique keys)
    player_id = pdata.get("player_id", None)
    if player_id is None or pd.isna(player_id):
        # Fallback: use a hash of player_name + team as player_id
        fallback_id = hashlib.md5(f"{player_name}_{team_abbrev}".encode()).hexdigest()[:8]
        player_id = f"fallback_{fallback_id}"

    player_identifier = str(player_id)
    stable_id = get_or_create_stable_id(player_identifier, stat_code)

    # Generate globally unique widget keys using stable tokens
    # Generate stable keys for buttons and manual input
    base_key = f"{stable_id}_{stat_code}_{render_index or 0}"
    unique_suffix = uuid.uuid4().hex

    decrease_key = f"detail_dec_{player_identifier}_{stat_code}_{render_index or 0}_{unique_suffix}"
    increase_key = f"detail_inc_{player_identifier}_{stat_code}_{render_index or 0}_{uuid.uuid4().hex}"
    reset_key    = f"detail_reset_{player_identifier}_{stat_code}_{render_index or 0}_{uuid.uuid4().hex}"
    manual_key   = f"detail_manual_{player_identifier}_{stat_code}_{render_index or 0}_{uuid.uuid4().hex}"

    
    # Initialize or get adjusted line from session state (safely)
    current_line = st.session_state.get(f"line_{player_identifier}_{stat_code}", fd_line_val or 0.0)


    # Handle button clicks
    button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

    with button_col1:
        if st.button("➖", key=decrease_key, help="Decrease line by 0.5", use_container_width=True):
            if current_line is not None:
                new_value = current_line - 0.5
                st.session_state[f"line_{player_identifier}_{stat_code}"] = new_value


    with button_col2:
        if stat_code == "DD":
            st.markdown("**Line:** N/A (DD market)")
        else:
            if current_line is None:
                st.markdown("**Line:** —")
                st.caption("No line available. Enter manually below.")
            else:
                st.markdown(f"**Current Line:** **{current_line:.1f}**")
                if current_line != fd_line_val:
                    if st.button("🔄 Reset", key=reset_key, use_container_width=True):
                        st.session_state[f"line_{player_identifier}_{stat_code}"] = fd_line_val or 0.0


    with button_col3:
        if st.button("➕", key=increase_key, help="Increase line by 0.5", use_container_width=True):
            if current_line is not None:
                new_value = current_line + 0.5
                st.session_state[f"line_{player_identifier}_{stat_code}"] = new_value


    # Manual line input option (if no line available)
    if current_line is None and stat_code != "DD":
        manual_default = get_manual_line_value(player_identifier, stat_code, 0.0)
        manual_line = st.number_input(
            "Enter line manually",
            min_value=0.0,
            value=manual_default,
            step=0.5,
            key=manual_key,
            help="Enter a custom line to calculate hit rate",
        )
        set_manual_line_value(player_identifier, stat_code, manual_line)
        if manual_line > 0 and manual_line != current_line:
            set_adjusted_line_value(player_identifier, stat_code, manual_line)
            st.rerun()
        current_line = get_adjusted_line_value(player_identifier, stat_code, fd_line_val)
    
    # Recalculate hit rate and edge based on current (adjusted) line
    if current_line is not None and stat_code != "DD":
        # Use combined logs for hit rate calculation
        combined_logs_for_hit = current_logs if not current_logs.empty else prior_logs
        adjusted_hit_rate = calc_hit_rate(combined_logs_for_hit, stat_code, current_line, window=10)
        adjusted_edge_str, adjusted_rec_text, adjusted_ou_short = calc_edge(prediction, current_line)
        
        # Store in session state (safely)
        cache_hit_rate(player_identifier, stat_code, adjusted_hit_rate)
    else:
        adjusted_hit_rate = hit_pct_val
        adjusted_edge_str = edge_str
        adjusted_rec_text = rec_text
        adjusted_ou_short = "—"

    colL, colH = st.columns(2)

    with colL:
        st.markdown("**Line / Edge**")
        if stat_code == "DD":
            st.write("Most books don't post DD props here, so no line.")
        else:
            if current_line is None:
                st.write("Line: —")
                st.write("Edge vs Line: —")
                st.caption("No line available for this player/stat.")
            else:
                # Show original line if adjusted
                if current_line != fd_line_val and fd_line_val is not None:
                    st.caption(f"Original line: {fd_line_val:.1f}")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 15px; border-radius: 8px; color: white; text-align: center; margin: 10px 0;">
                    <div style="font-size: 14px; opacity: 0.9;">CURRENT LINE</div>
                    <div style="font-size: 28px; font-weight: bold;">{current_line:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Edge vs Line:** {adjusted_edge_str}")
                st.caption(adjusted_rec_text)

    with colH:
        st.markdown("**Hit Rate (Last 10 Games)**")
        if stat_code == "DD":
            st.write("Hit%: —")
            st.caption("N/A for DD market here.")
        else:
            if adjusted_hit_rate is None:
                st.write("Hit%: —")
                st.caption("We only compute this if we have a line.")
            else:
                # Color code hit rate
                hit_color = "#28a745" if adjusted_hit_rate >= 50 else "#dc3545" if adjusted_hit_rate < 30 else "#ffc107"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {hit_color} 0%, {hit_color}dd 100%); 
                            padding: 15px; border-radius: 8px; color: white; text-align: center; margin: 10px 0;">
                    <div style="font-size: 14px; opacity: 0.9;">HIT RATE</div>
                    <div style="font-size: 28px; font-weight: bold;">{adjusted_hit_rate:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.caption(
                    f"Hit% = % of last 10 games over line of {current_line:.1f}. "
                    "Historical only."
                )
                
                # Show comparison if line was adjusted
                if current_line != fd_line_val and fd_line_val is not None and hit_pct_val is not None:
                    hit_diff = adjusted_hit_rate - hit_pct_val
                    st.caption(f"vs Original: {hit_diff:+.0f}% ({hit_pct_val:.0f}% @ {fd_line_val:.1f})")

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
1. **Pick a matchup** on the left sidebar under *Select Upcoming Game*  
2. **Pick a stat** (*Points*, *Rebounds*, *3PM*, etc.)  
3. Watch the **Matchup Board** fill in player by player  
4. Scroll down and **expand any player** to see the full deep dive (model projection, line vs projection edge, rest days, head-to-head, etc.)

No game is selected yet — choose one in the sidebar to start.
            """
        )
        return

    home_team = selected_game["home"]
    away_team = selected_game["away"]

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

    # shared data we reuse for all players
    def_vs_pos_df = scrape_defense_vs_position_cached_db()
    team_stats = get_team_stats_cached_db(season=prev_season)

    # fetch rosters
    home_roster = get_players_by_team(home_team, season=cur_season)
    if home_roster.empty:
        home_roster = get_players_by_team(home_team, season=prev_season)
    if not home_roster.empty:
        home_roster["team_abbrev"] = home_team

    away_roster = get_players_by_team(away_team, season=cur_season)
    if away_roster.empty:
        away_roster = get_players_by_team(away_team, season=prev_season)
    if not away_roster.empty:
        away_roster["team_abbrev"] = away_team

    if home_roster.empty and away_roster.empty:
        st.error("Couldn't load rosters for this matchup.")
        return

    combined_roster = pd.concat([home_roster, away_roster], ignore_index=True)
    combined_roster = combined_roster.drop_duplicates(subset=["player_id"])

    # Scrape and mark starters
    starters_dict = scrape_rotowire_starters()
    combined_roster["is_starter"] = combined_roster["full_name"].apply(
        lambda name: is_player_starter(name, starters_dict)
    )
    
    # Filter to only starters if toggle is on
    if show_only_starters:
        combined_roster = combined_roster[combined_roster["is_starter"] == True].copy()
        if combined_roster.empty:
            st.warning("⚠️ No projected starters found for this matchup. Showing all players.")
            combined_roster = pd.concat([home_roster, away_roster], ignore_index=True)
            combined_roster = combined_roster.drop_duplicates(subset=["player_id"])
            combined_roster["is_starter"] = combined_roster["full_name"].apply(
                lambda name: is_player_starter(name, starters_dict)
            )

    # Filter by player search query (case-insensitive)
    if player_search_query:
        search_lower = player_search_query.lower()
        combined_roster = combined_roster[
            combined_roster["full_name"].str.lower().str.contains(search_lower, na=False)
        ].copy()
        if combined_roster.empty:
            st.info(f"🔍 No players found matching '{player_search_query}'. Try a different search term.")

    total_players = len(combined_roster)

    # Pre-fetch FanDuel lines for this matchup (one call per matchup)
    # If the Odds API credits die or something fails, these will just be {}
    event_id = get_event_id_for_game(home_team, away_team)
    if event_id:
        odds_data = fetch_fanduel_lines(event_id)
    else:
        odds_data = {}

    # placeholders for streaming UI
    table_placeholder = st.empty()         # board table so far
    status_placeholder = st.empty()        # "Loaded X/Y"
    st.markdown("---")
    st.subheader("📂 Player Breakdowns (click any name below to expand)")
    st.caption(
        f"You can start opening players right away. "
        f"We'll keep adding more below as they finish loading. "
        f"({cur_season} vs {prev_season}, matchup context, line edge, trends, etc.)"
    )
    expanders_placeholder = st.empty()     # list of expanders so far

    table_rows = []
    player_payloads = []

    # stream each player
    for _, prow in combined_roster.iterrows():
        player_name = prow["full_name"]
        pid = prow["player_id"]
        team_abbrev = prow["team_abbrev"]
        opponent_abbrev = away_team if team_abbrev == home_team else home_team

        if pd.isna(pid):
            fallback_id = hashlib.md5(f"{player_name}_{team_abbrev}".encode()).hexdigest()[:8]
            player_identifier = f"fallback_{fallback_id}"
        else:
            player_identifier = str(pid)

        # position
        player_pos = get_player_position(pid, season=prev_season)

        # logs
        current_logs = get_player_game_logs_cached_db(
            pid, player_name, season=cur_season
        )
        prior_logs = get_player_game_logs_cached_db(
            pid, player_name, season=prev_season
        )

        # opponent recent form
        opponent_recent = get_opponent_recent_games(
            opponent_abbrev,
            season=prev_season,
            last_n=10
        )

        # head-to-head
        h2h_history = get_head_to_head_history(
            pid,
            opponent_abbrev,
            seasons=[prev_season, "2023-24"]
        )

        # defense rank vs position
        opp_def_rank_info = get_team_defense_rank_vs_position(
            opponent_abbrev,
            player_pos,
            def_vs_pos_df
        )

        # features for model
        feat = build_enhanced_feature_vector(
            current_logs,
            opponent_abbrev,
            team_stats,
            prior_season_logs=prior_logs,
            opponent_recent_games=opponent_recent,
            head_to_head_games=h2h_history,
            player_position=player_pos
        )

        # model projection (stat-based)
        if stat_code == "DD":
            pred_val = model_obj.predict_double_double(feat) * 100.0  # %
            proj_display = f"{pred_val:.1f}%"
        else:
            pred_val = model_obj.predict(feat, stat_code)
            proj_display = f"{pred_val:.1f}"

        # sportsbook line for THIS player/stat from FanDuel odds
        if stat_code == "DD":
            fd_line_val = None
        else:
            fd_info = get_player_fanduel_line(player_name, stat_code, odds_data)
            fd_line_val = fd_info["line"] if fd_info else None

        # compute edge + hit rate
        if stat_code == "DD":
            hit_pct_val = None
            edge_str = "—"
            rec_text = "Most books don't post DD lines"
            ou_short = "—"
        else:
            if fd_line_val is not None:
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

        # Opp Def Rank vs Position w/ emoji color
        rank_num = opp_def_rank_info.get("rank", 15)
        rating_txt = opp_def_rank_info.get("rating", "Average")
        d_emoji = defense_emoji(rank_num)
        opp_def_display = f"{d_emoji} #{rank_num} ({rating_txt})"

        # Check if player is starter and add ⭐️
        roster_row = combined_roster[combined_roster["full_name"] == player_name]
        is_starter = roster_row.iloc[0].get("is_starter", False) == True if not roster_row.empty else False
        player_display_name = f"⭐️ {player_name}" if is_starter else player_name
        
        # Get or initialize adjusted line from session state (safely)
        current_line_table = get_adjusted_line_value(player_identifier, stat_code, fd_line_val)
        
        # Recalculate hit rate based on adjusted line
        if current_line_table is not None and stat_code != "DD":
            combined_logs_for_hit_table = current_logs if not current_logs.empty else prior_logs
            adjusted_hit_rate_table = calc_hit_rate(combined_logs_for_hit_table, stat_code, current_line_table, window=10)
            adjusted_edge_str_table, adjusted_rec_text_table, adjusted_ou_short_table = calc_edge(pred_val, current_line_table)
        else:
            adjusted_hit_rate_table = hit_pct_val
            adjusted_edge_str_table = edge_str
            adjusted_ou_short_table = ou_short
        
        # Generate a stable unique ID for this player/stat combination
        # Store it in session state so it persists across renders (safely)
        stable_unique_id = get_or_create_stable_id(player_identifier, stat_code)

        # Store data for interactive table row
        table_row_data = {
            "player_name": player_name,
            "player_display_name": player_display_name,
            "player_identifier": player_identifier,
            "stable_id": stable_unique_id,
            "team_pos": f"{team_abbrev} · {player_pos}",
            "proj": proj_display,
            "fd_line_val": fd_line_val,
            "current_line": current_line_table,
            "hit_rate": adjusted_hit_rate_table,
            "ou_short": adjusted_ou_short_table,
            "opp_def": opp_def_display,
            "stat_code": stat_code,
            "current_logs": current_logs,
            "prior_logs": prior_logs,
        }
        table_rows.append(table_row_data)

        # prepare data for expander
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
            "player_id": player_identifier,  # Sanitised player identifier for session keys
            "stable_unique_id": stable_unique_id,  # Stable ID for button keys

            # sportsbook stuff to show in detail view
            "fd_line_val": fd_line_val,
            "hit_pct_val": hit_pct_val,
            "edge_str": edge_str,
            "rec_text": rec_text,
        }
        player_payloads.append(pdata)

        # re-render interactive table
        table_placeholder.empty()
        with table_placeholder.container():
            st.markdown("""
            <style>
            .player-table-row {
                padding: 8px;
                border-bottom: 1px solid #e0e0e0;
            }
            </style>
            """, unsafe_allow_html=True)

            header_cols = st.columns([2, 1.5, 1, 1.5, 1, 1, 2])
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
                st.markdown("**Hit%**")
            with header_cols[6]:
                st.markdown("**Opp Def**")

            st.markdown("---")

            for row_data in table_rows:
                row_cols = st.columns([2, 1.5, 1, 1.5, 1, 1, 2])

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

                        if current_line is not None:
                            line_btn_cols = st.columns([1, 2, 1])
                            with line_btn_cols[0]:
                                table_namespace = f"table::{stable_id_row}::{row_data['stat_code']}::dec"
                                dec_token = _get_widget_token(table_namespace)
                                btn_key = f"table_{dec_token}"
                                if st.button("➖", key=btn_key, 
                                             help="Decrease by 0.5"):
                                    set_adjusted_line_value(player_identifier, row_data['stat_code'], current_line - 0.5)
                                    st.rerun()

                            with line_btn_cols[1]:
                                st.write(f"**{current_line:.1f}**")
                                if current_line != row_data["fd_line_val"] and row_data["fd_line_val"] is not None:
                                    st.caption(f"({row_data['fd_line_val']:.1f})")

                            with line_btn_cols[2]:
                                table_namespace_inc = f"table::{stable_id_row}::{row_data['stat_code']}::inc"
                                inc_token = _get_widget_token(table_namespace_inc)
                                btn_key_inc = f"table_{inc_token}"
                                if st.button("➕", key=btn_key_inc, 
                                             help="Increase by 0.5"):
                                    set_adjusted_line_value(player_identifier, row_data['stat_code'], current_line + 0.5)
                                    st.rerun()
                        else:
                            st.write("—")

                with row_cols[4]:
                    st.write(row_data["ou_short"])

                with row_cols[5]:
                    hit_rate = row_data["hit_rate"]
                    if hit_rate is not None:
                        hit_color = "🟢" if hit_rate >= 50 else "🔴" if hit_rate < 30 else "🟡"
                        st.write(f"{hit_color} **{hit_rate:.0f}%**")
                    else:
                        st.write("—")

                with row_cols[6]:
                    st.write(row_data["opp_def"])

        # re-render ALL expanders so far
        rendered_combinations = set()
        expanders_placeholder.empty()
        with expanders_placeholder.container():
            for idx, info in enumerate(player_payloads):
                player_id_check = info.get("player_id", None)
                stat_code_check = info.get("stat_code", "")

                if player_id_check is None:
                    continue

                combo_key = (player_id_check, stat_code_check)
                if combo_key in rendered_combinations:
                    continue

                rendered_combinations.add(combo_key)

                roster_match = combined_roster[combined_roster["full_name"] == info["player_name"]]
                player_is_starter = roster_match.iloc[0].get("is_starter", False) == True if not roster_match.empty else False

                expander_title = f"{'⭐️ ' if player_is_starter else ''}{info['player_name']} ({info['team_abbrev']} · {info['player_pos']})"

                with st.expander(expander_title, expanded=False):
                    render_player_detail_body(info, cur_season, prev_season, render_index=idx)

        # status line
        status_placeholder.write(
            f"Loaded {len(table_rows)}/{total_players} players..."
        )

    # final status
    status_placeholder.success("✅ Done.")


def main():
    initialize_session_state_defaults()

    # ─────────────────────────────
    # Sidebar Controls
    # ─────────────────────────────
    st.sidebar.header("⚙️ Settings")

    # Cache stats block
    with st.sidebar.expander("💾 Cache Stats"):
        cache_stats = get_cache_stats()
        st.write(f"**Players cached:** {cache_stats.get('total_players', 0)}")
        st.write(f"**Games cached:** {cache_stats.get('total_games', 0):,}")
        st.write(f"**DB Size:** {cache_stats.get('db_size_mb', 0):.1f} MB")

        if st.button("🗑️ Clear Old Seasons"):
            clear_old_seasons([current_season, prior_season])
            st.success("Old seasons cleared!")
            st.rerun()

    # Upcoming games dropdown with a default "Select" option
    st.sidebar.subheader("📅 Select Upcoming Game")

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

        picked_label = st.sidebar.selectbox(
            f"Upcoming games (next 3 days) - {len(upcoming_games)} found",
            options=sidebar_options,
            index=0,  # default to instruction line
        )

        if picked_label != "-- Select a Game --":
            selected_game = game_map[picked_label]
            st.sidebar.info(
                f"Matchup: {selected_game['away']} @ {selected_game['home']}"
            )
    else:
        st.sidebar.warning("⚠️ No upcoming games in next 3 days.")
        picked_label = None
        selected_game = None

    # Which stat to predict on the board
    st.sidebar.subheader("📊 Stat to Project")
    stat_display_list = list(STAT_OPTIONS.keys())
    stat_display_choice = st.sidebar.selectbox(
        "Choose stat to preview on the board",
        options=stat_display_list,
        index=0,
    )
    stat_code_choice = STAT_OPTIONS[stat_display_choice]

    # Player filter toggle
    st.sidebar.subheader("👥 Player Filter")
    show_only_starters = st.sidebar.radio(
        "Filter players",
        options=["Show All Players", "Show Only Starters"],
        index=0,
        help="Filter to show only projected starters (⭐) or all players"
    )

    # Player search bar
    st.sidebar.subheader("🔍 Search Player")
    player_search_query = st.sidebar.text_input(
        "Search by player name",
        value="",
        placeholder="Type player name...",
        help="Filter players by name (case-insensitive)"
    )

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


if __name__ == "__main__":
    main()
