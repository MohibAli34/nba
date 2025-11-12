import pandas as pd
import numpy as np
from datetime import datetime

# Team abbreviation to full name mapping
TEAM_ABBREV_TO_NAME = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards'
}

DEFAULT_DEF_RATING = 110.0
DEFAULT_PACE = 100.0
DEFAULT_PTS_ALLOWED = 110.0


def calculate_season_averages(game_logs_df):
    """Season-to-date averages for core stats."""
    if game_logs_df.empty:
        return {}
    
    stats = {
        'PTS_avg': game_logs_df['PTS'].mean(),
        'AST_avg': game_logs_df['AST'].mean(),
        'REB_avg': game_logs_df['REB'].mean(),
        'FG3M_avg': game_logs_df['FG3M'].mean(),
        'PRA_avg': (game_logs_df['PTS'] + game_logs_df['REB'] + game_logs_df['AST']).mean(),
        'MIN_avg': game_logs_df['MIN'].mean() if 'MIN' in game_logs_df.columns else 0
    }
    return stats


def calculate_last_n_average(game_logs_df, n=5):
    """
    Rolling average for the most recent N games.
    Assumes game_logs_df is sorted newest-first.
    """
    if game_logs_df.empty:
        return {
            f'PTS_last{n}': 0,
            f'AST_last{n}': 0,
            f'REB_last{n}': 0,
            f'FG3M_last{n}': 0,
            f'PRA_last{n}': 0,
            f'MIN_last{n}': 0
        }
    
    last_n = game_logs_df.head(min(n, len(game_logs_df)))
    stats = {
        f'PTS_last{n}': last_n['PTS'].mean(),
        f'AST_last{n}': last_n['AST'].mean(),
        f'REB_last{n}': last_n['REB'].mean(),
        f'FG3M_last{n}': last_n['FG3M'].mean(),
        f'PRA_last{n}': (last_n['PTS'] + last_n['REB'] + last_n['AST']).mean(),
        f'MIN_last{n}': last_n['MIN'].mean() if 'MIN' in last_n.columns else 0
    }
    return stats


def calculate_double_double_probability(game_logs_df):
    """Probability (as fraction 0-1) of recording a double-double."""
    if game_logs_df.empty:
        return 0.0
    
    def is_double_double(row):
        # "double-double" = 10+ in at least 2 of PTS / REB / AST
        double_stats = sum([
            row['PTS'] >= 10,
            row['REB'] >= 10,
            row['AST'] >= 10
        ])
        return double_stats >= 2
    
    # safe copy to avoid setting a column on the upstream df
    temp = game_logs_df.copy()
    temp['is_dd'] = temp.apply(is_double_double, axis=1)
    return temp['is_dd'].mean()


def calculate_hit_rate(game_logs_df, stat_column, over_under_line, last_n=None):
    """
    Calculate hit rate: % of games where stat_column > over_under_line.
    last_n: if provided, restricts to only last_n most recent games.
    """
    if game_logs_df.empty:
        return 0.0
    
    if last_n:
        games = game_logs_df.head(last_n)
    else:
        games = game_logs_df
    
    if len(games) == 0:
        return 0.0
    
    hits = (games[stat_column] > over_under_line).sum()
    return hits / len(games)


def get_opponent_defensive_stats(opponent_abbrev, team_stats_df, recent_games_df=None):
    """
    Build opponent defensive context using the normalized team_stats_df
    we now get from get_team_stats_cached_db().

    Expected columns in team_stats_df:
        TEAM_ABBREVIATION
        DEF_RATING
        PACE
        PTS_ALLOWED

    recent_games_df is optional opponent_recent_games (their last ~10);
    if provided and it has 'PTS', we use that to compute a "recent_def_rating"
    and a trend delta.
    """
    # If we have nothing, return safe defaults
    if team_stats_df is None or team_stats_df.empty:
        return {
            'def_rating': DEFAULT_DEF_RATING,
            'pace': DEFAULT_PACE,
            'pts_allowed': DEFAULT_PTS_ALLOWED,
            'recent_def_rating': DEFAULT_DEF_RATING,
            'def_trend': 0.0
        }

    # Make sure abbrev comparison is consistent
    opp_code = opponent_abbrev.upper().strip()

    # Try to find this team in the cached stats
    if 'TEAM_ABBREVIATION' in team_stats_df.columns:
        opp_rows = team_stats_df[team_stats_df['TEAM_ABBREVIATION'] == opp_code]
    else:
        # We don't have a usable team key -> fall back
        print("[WARN] TEAM_ABBREVIATION column missing in team_stats_df")
        opp_rows = pd.DataFrame()

    if opp_rows.empty:
        # try "TEAM_NAME" fallback if somehow we cached by name in a weird season
        if 'TEAM_NAME' in team_stats_df.columns:
            full_name = TEAM_ABBREV_TO_NAME.get(opp_code, opp_code)
            opp_rows = team_stats_df[team_stats_df['TEAM_NAME'] == full_name]

    if opp_rows.empty:
        # no match, bail with defaults
        print(f"[WARN] Could not match opponent {opp_code} in team_stats_df")
        return {
            'def_rating': DEFAULT_DEF_RATING,
            'pace': DEFAULT_PACE,
            'pts_allowed': DEFAULT_PTS_ALLOWED,
            'recent_def_rating': DEFAULT_DEF_RATING,
            'def_trend': 0.0
        }

    # Take the first row for that opponent
    row = opp_rows.iloc[0]

    # pull season-level numbers with fallbacks
    pts_allowed = row.get('PTS_ALLOWED', np.nan)
    if pd.isna(pts_allowed):
        pts_allowed = DEFAULT_PTS_ALLOWED

    def_rating = row.get('DEF_RATING', np.nan)
    if pd.isna(def_rating):
        # fallback proxy: use points allowed as a stand-in
        def_rating = pts_allowed

    pace = row.get('PACE', np.nan)
    if pd.isna(pace):
        pace = DEFAULT_PACE

    # recent trend calc: how have they looked lately vs season
    recent_def_rating = def_rating
    def_trend = 0.0

    if recent_games_df is not None and not recent_games_df.empty:
        # If recent_games_df is something like "games opponents played vs them"
        # and has 'PTS' = opponent points scored, we can model "recent defense".
        if 'PTS' in recent_games_df.columns:
            recent_pts_allowed = recent_games_df['PTS'].mean()
            # A quick heuristic: "recent_def_rating" ~= recent PTS allowed,
            # scaled a little to resemble rating. You can adjust this logic later.
            recent_def_rating = recent_pts_allowed * 1.05
            def_trend = pts_allowed - recent_def_rating  # + => getting worse, - => improving

    print(
        f"[DEF CONTEXT] {opp_code}: "
        f"SeasonDef={def_rating:.1f}, "
        f"RecentDef={recent_def_rating:.1f}, "
        f"Pace={pace:.1f}, "
        f"PtsAllowed={pts_allowed:.1f}, "
        f"Trend={def_trend:+.1f}"
    )

    return {
        'def_rating': float(def_rating),
        'pace': float(pace),
        'pts_allowed': float(pts_allowed),
        'recent_def_rating': float(recent_def_rating),
        'def_trend': float(def_trend)
    }


def analyze_head_to_head_performance(h2h_games_df, stat_type):
    """
    Analyze player's historical performance vs this specific opponent.
    Returns avg, number of games, and recent-vs-older trend.
    """
    if h2h_games_df.empty:
        return {
            'h2h_avg': 0.0,
            'h2h_games': 0,
            'h2h_trend': 0.0
        }

    stat_map = {
        'PTS': 'PTS',
        'AST': 'AST',
        'REB': 'REB',
        'FG3M': 'FG3M'
    }

    stat_col = stat_map.get(stat_type, 'PTS')
    if stat_col not in h2h_games_df.columns:
        return {
            'h2h_avg': 0.0,
            'h2h_games': 0,
            'h2h_trend': 0.0
        }

    h2h_avg = h2h_games_df[stat_col].mean()
    h2h_games = len(h2h_games_df)

    # Trend = last 3 vs older games
    if h2h_games >= 3:
        recent_3 = h2h_games_df.head(3)[stat_col].mean()
        older_games = (
            h2h_games_df.iloc[3:][stat_col].mean()
            if h2h_games > 3
            else h2h_avg
        )
        h2h_trend = recent_3 - older_games
    else:
        h2h_trend = 0.0

    print(
        f"[H2H] {stat_col}: "
        f"{h2h_games} games, "
        f"avg={h2h_avg:.1f}, "
        f"trend={h2h_trend:+.1f}"
    )

    return {
        'h2h_avg': h2h_avg,
        'h2h_games': h2h_games,
        'h2h_trend': h2h_trend
    }


def is_back_to_back(game_logs_df, game_index=0):
    """
    Check if the indexed game (0 = most recent) is on a back-to-back.
    We assume logs are sorted newest-first.
    """
    if game_logs_df.empty or len(game_logs_df) < 2:
        return False

    if game_index >= len(game_logs_df) - 1:
        return False

    try:
        current_date = pd.to_datetime(game_logs_df.iloc[game_index]['GAME_DATE'])
        prev_date = pd.to_datetime(game_logs_df.iloc[game_index + 1]['GAME_DATE'])
        return (current_date - prev_date).days == 1
    except Exception:
        return False


def calculate_rest_days(game_logs_df, game_index=0):
    """
    Days since previous game for the indexed game (0 = most recent).
    """
    if game_logs_df.empty or len(game_logs_df) < 2:
        return 3

    if game_index >= len(game_logs_df) - 1:
        return 3

    try:
        current_date = pd.to_datetime(game_logs_df.iloc[game_index]['GAME_DATE'])
        prev_date = pd.to_datetime(game_logs_df.iloc[game_index + 1]['GAME_DATE'])
        return (current_date - prev_date).days
    except Exception:
        return 3


def blend_season_stats(current_logs, prior_logs, min_games_threshold=10):
    """
    Figure out how hard to weight current vs prior season.
    - If player barely played this season -> lean on prior season
    - If player has a lot of new data -> mostly this season
    """
    current_games = len(current_logs) if (current_logs is not None and not current_logs.empty) else 0

    if current_games == 0:
        weight_current = 0.0
        weight_prior = 1.0
    elif current_games < min_games_threshold:
        weight_current = current_games / min_games_threshold
        weight_prior = 1.0 - weight_current
    else:
        weight_current = 0.85
        weight_prior = 0.15

    return weight_current, weight_prior


def build_enhanced_feature_vector(
    player_game_logs,
    opponent_abbrev,
    team_stats_df,
    prior_season_logs=None,
    opponent_recent_games=None,
    head_to_head_games=None,
    player_position='F'
):
    """
    Build the full feature dict used by the model.

    Includes:
    - Blended season stats
    - Recency stats (last5 / last10)
    - Opponent context (defense, pace, trend)
    - Head-to-head vs this exact team
    - Game context (rest, back-to-back)
    - Double-double probability
    """

    features = {}

    # -------------------------
    # Season blending
    # -------------------------
    weight_current, weight_prior = blend_season_stats(player_game_logs, prior_season_logs)
    current_stats = calculate_season_averages(player_game_logs)

    if prior_season_logs is not None and not prior_season_logs.empty:
        prior_stats = calculate_season_averages(prior_season_logs)
    else:
        prior_stats = current_stats

    # blended core rates
    for stat in ['PTS_avg', 'AST_avg', 'REB_avg', 'FG3M_avg', 'PRA_avg']:
        current_val = current_stats.get(stat, 0)
        prior_val = prior_stats.get(stat, 0)
        features[stat] = (weight_current * current_val) + (weight_prior * prior_val)

    # -------------------------
    # Rolling recent form (last5/last10)
    # We want this early-season to include prior season to stabilize.
    # -------------------------
    if (player_game_logs is not None and not player_game_logs.empty) and \
       (prior_season_logs is not None and not prior_season_logs.empty):
        combined_logs = pd.concat([player_game_logs, prior_season_logs], ignore_index=True)
        last5_stats = calculate_last_n_average(combined_logs, n=5)
        last10_stats = calculate_last_n_average(combined_logs, n=10)
    elif player_game_logs is not None and not player_game_logs.empty:
        last5_stats = calculate_last_n_average(player_game_logs, n=5)
        last10_stats = calculate_last_n_average(player_game_logs, n=10)
    elif prior_season_logs is not None and not prior_season_logs.empty:
        last5_stats = calculate_last_n_average(prior_season_logs, n=5)
        last10_stats = calculate_last_n_average(prior_season_logs, n=10)
    else:
        # no data at all, fall back to the blended averages from above
        last5_stats = {}
        last10_stats = {}
        for stat in ['PTS', 'AST', 'REB', 'FG3M', 'PRA']:
            val = features.get(f'{stat}_avg', 0)
            last5_stats[f'{stat}_last5'] = val
            last10_stats[f'{stat}_last10'] = val

    features.update(last5_stats)
    features.update(last10_stats)

    # prior season reference for context
    features['prior_PTS'] = prior_stats.get('PTS_avg', 0)
    features['prior_AST'] = prior_stats.get('AST_avg', 0)
    features['prior_REB'] = prior_stats.get('REB_avg', 0)
    features['prior_FG3M'] = prior_stats.get('FG3M_avg', 0)

    # -------------------------
    # Opponent context
    # -------------------------
    enhanced_opp_stats = get_opponent_defensive_stats(
        opponent_abbrev,
        team_stats_df,
        recent_games_df=opponent_recent_games
    )

    features['opp_def_rating'] = enhanced_opp_stats['def_rating']
    features['opp_recent_def_rating'] = enhanced_opp_stats['recent_def_rating']
    features['opp_def_trend'] = enhanced_opp_stats['def_trend']
    features['opp_pace'] = enhanced_opp_stats['pace']
    features['opp_pts_allowed'] = enhanced_opp_stats['pts_allowed']

    # -------------------------
    # Head-to-head vs opponent
    # -------------------------
    if head_to_head_games is not None and not head_to_head_games.empty:
        for stat in ['PTS', 'AST', 'REB', 'FG3M']:
            h2h_stats = analyze_head_to_head_performance(head_to_head_games, stat)
            features[f'h2h_{stat}_avg'] = h2h_stats['h2h_avg']
            features[f'h2h_{stat}_games'] = h2h_stats['h2h_games']
            features[f'h2h_{stat}_trend'] = h2h_stats['h2h_trend']
    else:
        for stat in ['PTS', 'AST', 'REB', 'FG3M']:
            features[f'h2h_{stat}_avg'] = 0.0
            features[f'h2h_{stat}_games'] = 0
            features[f'h2h_{stat}_trend'] = 0.0

    # -------------------------
    # Game context (fatigue, rest)
    # -------------------------
    features['is_back_to_back'] = 1 if is_back_to_back(player_game_logs) else 0
    features['rest_days'] = calculate_rest_days(player_game_logs)

    # -------------------------
    # Double-double probability
    # -------------------------
    if player_game_logs is not None and not player_game_logs.empty:
        features['dd_probability'] = calculate_double_double_probability(player_game_logs)
    else:
        features['dd_probability'] = 0.0

    # -------------------------
    # Metadata
    # -------------------------
    features['weight_current'] = weight_current
    features['weight_prior'] = weight_prior
    features['current_games_played'] = len(player_game_logs) if (player_game_logs is not None and not player_game_logs.empty) else 0
    features['player_position'] = player_position

    return features
