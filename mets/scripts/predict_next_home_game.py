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

# Local (non-iCloud) cache — launchd background runs can't reliably read/write
# files inside iCloud Drive (confirmed by testing: macOS file-coordination/TCC
# conflicts with unattended processes). Use sync_data.py to move files between
# here and the iCloud project folder when running interactively.
DATA_DIR = Path.home() / "Library" / "Application Support" / "mets-prediction" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path, retries=5, delay=1.0, **kwargs):
    """Pandas' C parser reading an iCloud-backed path directly can throw
    'Resource deadlock avoided' when running under launchd (no controlling session)
    — it conflicts with iCloud's file coordination in a way plain open() doesn't.
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
    "is_day_game",
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
            })
    df = pd.DataFrame(games)
    if df.empty:
        return df

    if df["game_date_utc"].notna().any():
        dt = pd.to_datetime(df["game_date_utc"], utc=True)
        hour_et = (dt.dt.hour - 4) % 24
        df["game_hour_et"] = hour_et + dt.dt.minute / 60
    df["is_day_game"] = (df["game_hour_et"] < 17).astype("Int64")

    missing_att = df[(df["status"] == "Final") & df["attendance"].isna() & ~df["game_pk"].isin(known_game_pks)]
    if len(missing_att) > 0:
        print(f"  Backfilling attendance for {len(missing_att)} new games via boxscore...")
        for idx, row in missing_att.iterrows():
            df.at[idx, "attendance"] = fetch_attendance(row["game_pk"])
    return df


def get_weather(target_date, forecast_needed):
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


def build_features(games_df, team_id_map, today_str):
    """Attach weather, standings, starter ERA, promotions to every row in games_df."""
    rows = []
    for _, row in games_df.iterrows():
        is_future = row["date"] > today_str
        weather = get_weather(row["date"], forecast_needed=is_future)
        standings_date = today_str if is_future else row["date"]
        standings = get_standings(standings_date)
        mets_s = standings.get(TEAM_ID, {})
        opp_id = team_id_map.get(row["opponent"])
        opp_s = standings.get(opp_id, {}) if opp_id else {}
        promo = load_promotions_for_date(row["date"])
        mets_era = get_starter_era_entering(row["mets_starter_id"], 2026, row["date"])
        opp_era = get_starter_era_entering(row["opp_starter_id"], 2026, row["date"])

        dt = pd.to_datetime(row["date"])
        rows.append({
            **row.to_dict(),
            **weather,
            **promo,
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


def main():
    today = date.today().isoformat()
    print(f"=== Mets next-home-game attendance prediction — run date {today} ===\n")

    master_path = DATA_DIR / "mets_2026_master.csv"
    existing = safe_read_csv(master_path) if master_path.exists() else pd.DataFrame(columns=["game_pk"])

    print("Refreshing 2026 home game data...")
    games_2026 = get_2026_home_games(known_game_pks=frozenset(existing["game_pk"]))
    print(f"  {len(games_2026)} total 2026 home games ({(games_2026['status'] == 'Final').sum()} Final, "
          f"{(games_2026['status'] == 'Scheduled').sum()} Scheduled)")

    upcoming = games_2026[games_2026["status"] == "Scheduled"].sort_values("date")
    if upcoming.empty:
        print("No upcoming Mets home games found. Nothing to predict.")
        return
    next_game = upcoming.iloc[[0]]
    print(f"\nNext Mets home game: {next_game.iloc[0]['date']} vs {next_game.iloc[0]['opponent']}")
    if pd.isna(next_game.iloc[0]["mets_starter_name"]):
        print("  (Mets starter not yet announced)")
    if pd.isna(next_game.iloc[0]["opp_starter_name"]):
        print("  (Opponent starter not yet announced)")

    completed_2026 = games_2026[(games_2026["status"] == "Final") & games_2026["attendance"].notna()]
    new_completed = completed_2026[~completed_2026["game_pk"].isin(existing["game_pk"])]
    print(f"\n{len(completed_2026)} completed games total, {len(existing)} already in mets_2026_master.csv, "
          f"{len(new_completed)} new since last run")

    team_id_map = get_team_id_map()
    print(f"Building features for {len(new_completed)} new completed games + the next game...")
    feature_rows = build_features(pd.concat([new_completed, next_game]), team_id_map, today)

    new_completed_features = feature_rows[feature_rows["status"] == "Final"].copy()
    next_features = feature_rows[feature_rows["status"] == "Scheduled"].copy()

    completed_features = pd.concat([existing, new_completed_features], ignore_index=True) \
        if not new_completed_features.empty else existing
    safe_to_csv(completed_features, master_path, index=False)
    print(f"Saved refreshed mets_2026_master.csv ({len(completed_features)} completed games total)")

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
        print(f"  Note: next game is missing {missing_cols} — filling with training column means as a fallback")
        X_next = X_next.fillna(X_train.mean())

    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    prediction = model.predict(X_next)[0]

    game_date = next_game.iloc[0]["date"]
    opponent = next_game.iloc[0]["opponent"]
    print(f"\n{'=' * 60}")
    print(f"PREDICTION: {game_date} vs {opponent}")
    print(f"Predicted attendance: {prediction:,.0f}")
    print(f"{'=' * 60}")

    log_path = DATA_DIR / "mets_2026_predictions_log.csv"
    log_row = pd.DataFrame([{
        "run_date": today,
        "game_date": game_date,
        "opponent": opponent,
        "predicted_attendance": round(prediction),
        "mets_starter": next_features.iloc[0].get("mets_starter_name"),
        "mets_starter_era": next_features.iloc[0].get("mets_starter_era"),
        "opp_starter": next_features.iloc[0].get("opp_starter_name"),
        "opp_starter_era": next_features.iloc[0].get("opp_starter_era"),
        "avg_temp_f": next_features.iloc[0].get("avg_temp_f"),
        "avg_precip_mm": next_features.iloc[0].get("avg_precip_mm"),
        "is_promo": next_features.iloc[0].get("is_promo"),
        "days_out": (pd.to_datetime(game_date) - pd.to_datetime(today)).days,
        "actual_attendance": None,
    }])
    if log_path.exists():
        safe_to_csv(log_row, log_path, mode="a", header=False, index=False)
    else:
        safe_to_csv(log_row, log_path, index=False)
    print(f"\nLogged to {log_path.name} — once the game is played, fill in 'actual_attendance' "
          f"for that row to track accuracy.")

    from generate_prediction_report import build_report
    build_report()


if __name__ == "__main__":
    main()
