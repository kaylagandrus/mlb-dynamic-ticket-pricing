#!/usr/bin/env python3
"""
Predict attendance for the Mets' next home game, using 2022-2025 history plus
every 2026 home game played so far.

Run this daily: python predict_next_home_game.py

Each run:
  1. Refreshes mets_2026_master.csv with any newly completed home games
     (re-pulls games, weather, standings, starter ERA, promotions, broadcast,
     game time for anything Final that wasn't in our data yet).
  2. Finds the next upcoming Mets home game.
  3. Builds live features for it: current standings, a weather *forecast*
     (not historical archive) if the game is within ~16 days, the probable
     starter's point-in-time ERA if announced, promotions, and broadcast info.
  4. Trains Gradient Boosting on everything available (2022-2025 + all
     completed 2026 home games) and predicts that one game.
  5. Appends the prediction to mets_2026_predictions_log.csv, so you can
     compare it against the actual attendance once MLB posts it, and see how
     the prediction evolves on subsequent daily runs as more info (forecast,
     probable starter) firms up.
"""

import argparse
import io
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor

# Local (non-iCloud) cache: launchd background runs can't reliably read/write
# files inside iCloud Drive (confirmed by testing: macOS file-coordination/TCC
# conflicts with unattended processes). Use sync_data.py to move files between
# here and the iCloud project folder when running interactively.
DATA_DIR = Path.home() / "Library" / "Application Support" / "mets-prediction" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path, retries=5, delay=1.0, **kwargs):
    """Pandas' C parser reading an iCloud-backed path directly can throw
    'Resource deadlock avoided' when running under launchd (no controlling session),
    it conflicts with iCloud's file coordination in a way plain open() doesn't.
    Read the bytes with open() first, then hand pandas an in-memory buffer."""
    last_err = None
    for attempt in range(retries):
        try:
            with open(path, "r", newline="") as f:
                content = f.read()
            return pd.read_csv(io.StringIO(content), **kwargs)
        except OSError as e:
            last_err = e
            time.sleep(delay)
    raise last_err


def safe_to_csv(df, path, retries=5, delay=1.0, **kwargs):
    """Same iCloud/launchd issue as safe_read_csv, for writes: render to a string
    buffer with pandas, then write it out with plain open() instead of df.to_csv(path)."""
    mode = kwargs.pop("mode", "w")
    buf = io.StringIO()
    df.to_csv(buf, **kwargs)
    content = buf.getvalue()
    last_err = None
    for attempt in range(retries):
        try:
            with open(path, mode, newline="") as f:
                f.write(content)
            return
        except OSError as e:
            last_err = e
            time.sleep(delay)
    raise last_err


TEAM_ID = 121  # New York Mets
LAT, LON = 40.7571, -73.8458  # Citi Field
ACE_ERA_THRESHOLD = 3.00

NUMERIC_FEATURES = [
    "season", "month", "dayofweek", "is_weekend", "avg_temp_f", "avg_precip_mm",
    "is_promo", "is_bobblehead", "n_promotions",
    "is_national", "is_premium_national",
    "mets_win_pct", "opp_win_pct",
    "mets_streak_num",
    "mets_starter_era", "opp_starter_era",
    "is_day_game", "is_game2",
]
CATEGORICAL_FEATURES = ["opponent"]
TARGET = "attendance"


def api_get(url, params=None, retries=3, timeout=15):
    for attempt in range(retries):
        try:
            return requests.get(url, params=params or {}, timeout=timeout).json()
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f"  Warning: {url} failed after {retries} attempts: {e}")
                return {}
            time.sleep(2)


def fetch_attendance(game_pk):
    data = api_get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore")
    for item in data.get("info", []):
        if item.get("label") == "Att":
            val = re.sub(r"[^0-9]", "", item.get("value", ""))
            if val:
                return int(val)
    return None


def get_2026_home_games(known_game_pks=frozenset()):
    """All Mets home games in 2026, Final and Scheduled, with probable pitcher + broadcasts.

    known_game_pks: game_pks we already have attendance for, so we don't waste a
    boxscore call re-fetching something we already know on every daily run.
    """
    data = api_get("https://statsapi.mlb.com/api/v1/schedule", {
        "sportId": 1, "season": 2026, "gameType": "R", "teamId": TEAM_ID,
        "hydrate": "linescore,team,probablePitcher,broadcasts",
    })
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g["teams"]["home"]["team"]["id"] != TEAM_ID:
                continue
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            attendance = g.get("attendance") or home.get("attendance")
            tv_national = [b["name"] for b in g.get("broadcasts", [])
                            if b.get("type") == "TV" and b.get("isNational")]
            games.append({
                "date": d["date"],
                "season": 2026,
                "opponent": away["team"]["name"],
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "attendance": attendance,
                "status": g["status"]["detailedState"],
                "game_pk": g["gamePk"],
                "mets_starter_id": (home.get("probablePitcher") or {}).get("id"),
                "mets_starter_name": (home.get("probablePitcher") or {}).get("fullName"),
                "opp_starter_id": (away.get("probablePitcher") or {}).get("id"),
                "opp_starter_name": (away.get("probablePitcher") or {}).get("fullName"),
                "is_national": 1 if tv_national else 0,
                "national_network": ", ".join(tv_national) if tv_national else None,
                "game_hour_et": None,
                "game_date_utc": g.get("gameDate"),
                "game_number": g.get("gameNumber", 1),
                "is_doubleheader": int(g.get("doubleHeader", "N") != "N"),
            })
    df = pd.DataFrame(games)
    if df.empty:
        return df

    if df["game_date_utc"].notna().any():
        dt = pd.to_datetime(df["game_date_utc"], utc=True)
        hour_et = (dt.dt.hour - 4) % 24
        df["game_hour_et"] = hour_et + dt.dt.minute / 60
    df["is_day_game"] = (df["game_hour_et"] < 17).astype("Int64")
    df["is_game2"] = (df["game_number"] == 2).astype(int)

    missing_att = df[(df["status"] == "Final") & df["attendance"].isna() & ~df["game_pk"].isin(known_game_pks)]
    if len(missing_att) > 0:
        print(f"  Backfilling attendance for {len(missing_att)} new games via boxscore...")
        for idx, row in missing_att.iterrows():
            df.at[idx, "attendance"] = fetch_attendance(row["game_pk"])
    return df


def get_weather_asof(target_date, as_of_date):
    """The forecast as it was actually issued `offset` days before target_date, using
    Open-Meteo's previous-runs API (archives real historical forecast model runs, not
    just current conditions). Only has data for forecasts issued 1-7 days ahead, returns
    None if the offset is out of that range or the model run has no data for it."""
    offset = (pd.Timestamp(target_date) - pd.Timestamp(as_of_date)).days
    if not (1 <= offset <= 7):
        return None
    temp_col, precip_col = f"temperature_2m_previous_day{offset}", f"precipitation_previous_day{offset}"
    data = api_get("https://previous-runs-api.open-meteo.com/v1/forecast", {
        "latitude": LAT, "longitude": LON,
        "start_date": target_date, "end_date": target_date,
        "hourly": f"{temp_col},{precip_col}",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    })
    hourly = data.get("hourly", {})
    temps = [t for t in hourly.get(temp_col, [])[16:21] if t is not None]
    precip = [p for p in hourly.get(precip_col, [])[16:21] if p is not None]
    if not temps:
        return None
    return {
        "avg_temp_f": round(sum(temps) / len(temps), 1),
        "avg_precip_mm": round(sum(precip) / len(precip), 1) if precip else None,
    }


def get_weather(target_date, forecast_needed, as_of_date=None):
    if forecast_needed and as_of_date:
        asof_weather = get_weather_asof(target_date, as_of_date)
        if asof_weather is not None:
            return asof_weather
        # offset out of the previous-runs API's supported range (>7 days ahead):
        # fall through and try archive/live below as the best available approximation

    base_url = "https://api.open-meteo.com/v1/forecast" if forecast_needed \
        else "https://archive-api.open-meteo.com/v1/archive"
    data = api_get(base_url, {
        "latitude": LAT, "longitude": LON,
        "start_date": target_date, "end_date": target_date,
        "hourly": "temperature_2m,precipitation",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    })
    hourly = data.get("hourly", {})
    temps = [t for t in hourly.get("temperature_2m", [])[16:21] if t is not None]
    precip = [p for p in hourly.get("precipitation", [])[16:21] if p is not None]
    if not temps and not forecast_needed:
        # target_date hasn't happened yet even now, archive has nothing for it yet;
        # fall back to today's live forecast as the closest available approximation
        return get_weather(target_date, forecast_needed=True)
    return {
        "avg_temp_f": round(sum(temps) / len(temps), 1) if temps else None,
        "avg_precip_mm": round(sum(precip) / len(precip), 1) if precip else None,
    }


def get_standings(as_of_date):
    data = api_get("https://statsapi.mlb.com/api/v1/standings", {
        "leagueId": "103,104", "season": as_of_date[:4], "date": as_of_date,
    })
    teams = {}
    for record in data.get("records", []):
        for tr in record.get("teamRecords", []):
            team_id = tr["team"]["id"]
            streak = tr.get("streak") or {}
            streak_num = streak.get("streakNumber")
            if streak_num is not None and streak.get("streakType") == "losses":
                streak_num = -streak_num
            lr = tr.get("leagueRecord", {})
            teams[team_id] = {
                "win_pct": float(lr.get("pct") or 0),
                "streak_num": streak_num,
            }
    return teams


def get_team_id_map():
    data = api_get("https://statsapi.mlb.com/api/v1/teams", {"sportId": 1})
    return {t["name"]: t["id"] for t in data.get("teams", [])}


def ip_to_innings_decimal(ip_str):
    if ip_str in (None, ""):
        return 0.0
    whole, _, frac = str(ip_str).partition(".")
    thirds = {"": 0, "0": 0, "1": 1, "2": 2}.get(frac, 0)
    return int(whole) + thirds / 3


def get_starter_era_entering(pitcher_id, season, game_date):
    if pd.isna(pitcher_id):
        return None
    data = api_get(f"https://statsapi.mlb.com/api/v1/people/{int(pitcher_id)}/stats",
                    {"stats": "gameLog", "group": "pitching", "season": season})
    starts = []
    for split in data.get("stats", [{}])[0].get("splits", []) if data.get("stats") else []:
        if split["stat"].get("gamesStarted", 0) != 1:
            continue
        if split["date"] < game_date:
            starts.append((split["stat"].get("earnedRuns", 0),
                            ip_to_innings_decimal(split["stat"].get("inningsPitched"))))
    total_er = sum(s[0] for s in starts)
    total_ip = sum(s[1] for s in starts)
    return round(9 * total_er / total_ip, 2) if total_ip > 0 else None


def load_promotions_for_date(target_date):
    path = DATA_DIR / "mets_promotions.csv"
    if not path.exists():
        return {"n_promotions": 0, "is_bobblehead": 0}
    promos = safe_read_csv(path)
    day_promos = promos[promos["date"] == target_date]
    return {
        "n_promotions": len(day_promos),
        "is_bobblehead": int((day_promos["type"] == "bobblehead").any()),
    }


PREMIUM_NETWORKS = {"FOX", "FOX, FOX", "ESPN/ESPN App", "ESPN/ESPN App, ESPN/ESPN App",
                    "Apple TV", "Apple TV, Apple TV", "TBS (out-of-market only)",
                    "TBS (out-of-market only), TBS (out-of-market only)",
                    "TBS (out-of-market only), TBS", "Roku", "Roku, Roku",
                    "FS1", "FS1, FS1"}


def build_features(games_df, team_id_map, today_str, backfill=False):
    """Attach weather, standings, starter ERA, promotions to every row in games_df.

    backfill=True (used by --as-of): reconstruct the true historical forecast instead
    of today's live one, and blank out probable starters/ERA entirely, since there's no
    way to know whether a starter had actually been announced as of that past date."""
    rows = []
    for _, row in games_df.iterrows():
        is_future = row["date"] > today_str
        weather = get_weather(row["date"], forecast_needed=is_future,
                               as_of_date=today_str if backfill else None)
        standings_date = today_str if is_future else row["date"]
        standings = get_standings(standings_date)
        mets_s = standings.get(TEAM_ID, {})
        opp_id = team_id_map.get(row["opponent"])
        opp_s = standings.get(opp_id, {}) if opp_id else {}
        promo = load_promotions_for_date(row["date"])
        if backfill and is_future:
            mets_starter_id = opp_starter_id = None
            mets_starter_name = opp_starter_name = None
        else:
            mets_starter_id, opp_starter_id = row["mets_starter_id"], row["opp_starter_id"]
            mets_starter_name, opp_starter_name = row["mets_starter_name"], row["opp_starter_name"]
        mets_era = get_starter_era_entering(mets_starter_id, 2026, row["date"])
        opp_era = get_starter_era_entering(opp_starter_id, 2026, row["date"])

        dt = pd.to_datetime(row["date"])
        rows.append({
            **row.to_dict(),
            **weather,
            **promo,
            "mets_starter_id": mets_starter_id,
            "mets_starter_name": mets_starter_name,
            "opp_starter_id": opp_starter_id,
            "opp_starter_name": opp_starter_name,
            "is_promo": int(promo["n_promotions"] > 0),
            "mets_win_pct": mets_s.get("win_pct"),
            "opp_win_pct": opp_s.get("win_pct"),
            "mets_streak_num": mets_s.get("streak_num"),
            "mets_starter_era": mets_era,
            "opp_starter_era": opp_era,
            "is_premium_national": int(row.get("national_network") in PREMIUM_NETWORKS),
            "month": dt.month,
            "dayofweek": dt.dayofweek,
            "is_weekend": int(dt.dayofweek in (4, 5, 6)),
        })
    return pd.DataFrame(rows)


def backfill_actual_attendance(completed_2026, log_path):
    """Fill in actual_attendance for any logged prediction whose game has since been
    played, using the attendance already pulled via fetch_attendance() in completed_2026
    (the same boxscore-based source every other attendance number in this project uses).
    Matches by (game_date, opponent) since the log doesn't store game_pk."""
    if not log_path.exists():
        return 0
    log_df = safe_read_csv(log_path)
    if log_df.empty:
        return 0
    attendance_by_game = completed_2026.set_index(["date", "opponent"])["attendance"].to_dict()
    to_fill = log_df["actual_attendance"].isna()
    if not to_fill.any():
        return 0
    filled_values = log_df.loc[to_fill].apply(
        lambda r: attendance_by_game.get((r["game_date"], r["opponent"])), axis=1)
    n_filled = int(filled_values.notna().sum())
    if n_filled:
        log_df.loc[to_fill, "actual_attendance"] = filled_values.values
        safe_to_csv(log_df, log_path, index=False)
    return n_filled


TRACKING_WINDOW_DAYS = 9  # start logging a game once it's this many days out or fewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None,
                         help="Backfill a missed day: reconstruct the run as if it happened on this "
                              "YYYY-MM-DD date (standings, days_out, and the training cutoff are accurate "
                              "for that date, but probable starters and the weather forecast come from "
                              "today's live API state, there's no way to query what either looked like "
                              "on a past date).")
    args = parser.parse_args()
    today = args.as_of or date.today().isoformat()
    if args.as_of:
        print(f"=== Mets home-game attendance prediction, BACKFILLED for {today} ===")
        print("    (standings/training cutoff are accurate for that date; probable starters and weather")
        print("     reflect today's live data, not necessarily what was known on that date)\n")
    else:
        print(f"=== Mets home-game attendance prediction, run date {today} ===\n")

    master_path = DATA_DIR / "mets_2026_master.csv"
    existing = safe_read_csv(master_path) if master_path.exists() else pd.DataFrame(columns=["game_pk"])

    print("Refreshing 2026 home game data...")
    games_2026 = get_2026_home_games(known_game_pks=frozenset(existing["game_pk"]))
    print(f"  {len(games_2026)} total 2026 home games ({(games_2026['status'] == 'Final').sum()} Final, "
          f"{(games_2026['status'] == 'Scheduled').sum()} Scheduled)")

    upcoming = games_2026[games_2026["status"] == "Scheduled"].sort_values("date").copy()
    upcoming["days_out"] = (pd.to_datetime(upcoming["date"]) - pd.Timestamp(today)).dt.days
    tracked_games = upcoming[(upcoming["days_out"] >= 0) & (upcoming["days_out"] <= TRACKING_WINDOW_DAYS)]

    if tracked_games.empty:
        next_out = upcoming["days_out"].min() if not upcoming.empty else None
        print(f"\nNo home games within {TRACKING_WINDOW_DAYS} days yet"
              + (f" (next one is {next_out} days out)." if next_out is not None else ", none scheduled."))
    else:
        print(f"\nTracking {len(tracked_games)} home game(s) within {TRACKING_WINDOW_DAYS} days:")
        for _, g in tracked_games.iterrows():
            starter_note = ""
            if pd.isna(g["mets_starter_name"]) or pd.isna(g["opp_starter_name"]):
                starter_note = "  (starter(s) not yet announced)"
            print(f"  {g['date']} vs {g['opponent']} ({int(g['days_out'])}d out){starter_note}")

    completed_2026 = games_2026[(games_2026["status"] == "Final") & games_2026["attendance"].notna()
                                 & (games_2026["date"] <= today)]
    new_completed = completed_2026[~completed_2026["game_pk"].isin(existing["game_pk"])]
    print(f"\n{len(completed_2026)} completed games total, {len(existing)} already in mets_2026_master.csv, "
          f"{len(new_completed)} new since last run")

    team_id_map = get_team_id_map()
    print(f"Building features for {len(new_completed)} new completed games + {len(tracked_games)} tracked game(s)...")
    feature_rows = build_features(pd.concat([new_completed, tracked_games]), team_id_map, today,
                                   backfill=bool(args.as_of))

    new_completed_features = feature_rows[feature_rows["status"] == "Final"].copy()
    next_features = feature_rows[feature_rows["status"] == "Scheduled"].copy()

    completed_features = pd.concat([existing, new_completed_features], ignore_index=True) \
        if not new_completed_features.empty else existing
    safe_to_csv(completed_features, master_path, index=False)
    print(f"Saved refreshed mets_2026_master.csv ({len(completed_features)} completed games total)")

    log_path = DATA_DIR / "mets_2026_predictions_log.csv"
    n_backfilled = backfill_actual_attendance(completed_features, log_path)
    if n_backfilled:
        print(f"Backfilled actual_attendance for {n_backfilled} logged prediction(s) whose games have been played")

    if next_features.empty:
        print("\nNothing within the tracking window, nothing to predict this run.")
        from generate_prediction_report import build_report
        build_report()
        return

    print("\nTraining Gradient Boosting on 2022-2025 + all completed 2026 home games...")
    train_hist = safe_read_csv(DATA_DIR / "mets_master.csv").dropna(subset=["attendance"])
    train_full = pd.concat([train_hist, completed_features], ignore_index=True)

    model_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    train_rows = train_full.dropna(subset=[c for c in model_cols + [TARGET] if c in train_full.columns])
    print(f"  Training on {len(train_rows)} rows (dropped {len(train_full) - len(train_rows)} with missing features)")

    combined = pd.concat([train_rows[model_cols], next_features[model_cols]], ignore_index=True)
    combined_encoded = pd.get_dummies(combined, columns=CATEGORICAL_FEATURES, drop_first=True)
    X_train = combined_encoded.iloc[:len(train_rows)]
    X_next = combined_encoded.iloc[len(train_rows):]
    y_train = train_rows[TARGET].reset_index(drop=True)

    missing_next = X_next.isna().any(axis=1)
    if missing_next.any():
        missing_cols = X_next.columns[X_next.isna().any()].tolist()
        print(f"  Note: {missing_next.sum()} game(s) missing {missing_cols}, filling with training column means as a fallback")
        X_next = X_next.fillna(X_train.mean())

    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_next)

    next_features = next_features.reset_index(drop=True)
    log_rows = []
    print(f"\n{'=' * 60}")
    for i, prediction in enumerate(predictions):
        row = next_features.iloc[i]
        print(f"PREDICTION: {row['date']} vs {row['opponent']} ({int(row['days_out'])}d out): {prediction:,.0f}")
        log_rows.append({
            "run_date": today,
            "game_date": row["date"],
            "opponent": row["opponent"],
            "predicted_attendance": round(prediction),
            "mets_starter": row.get("mets_starter_name"),
            "mets_starter_era": row.get("mets_starter_era"),
            "opp_starter": row.get("opp_starter_name"),
            "opp_starter_era": row.get("opp_starter_era"),
            "avg_temp_f": row.get("avg_temp_f"),
            "avg_precip_mm": row.get("avg_precip_mm"),
            "is_promo": row.get("is_promo"),
            "days_out": int(row["days_out"]),
            "actual_attendance": None,
        })
    print(f"{'=' * 60}")

    log_df = pd.DataFrame(log_rows)
    if log_path.exists():
        safe_to_csv(log_df, log_path, mode="a", header=False, index=False)
    else:
        safe_to_csv(log_df, log_path, index=False)
    print(f"\nLogged {len(log_df)} prediction(s) to {log_path.name}, once each game is played, fill in "
          f"'actual_attendance' for that row to track accuracy.")

    from generate_prediction_report import build_report
    build_report()


if __name__ == "__main__":
    main()
