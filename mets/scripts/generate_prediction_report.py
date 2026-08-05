#!/usr/bin/env python3
"""
Turn mets_2026_predictions_log.csv into a readable HTML report:
  - One row per game, one column per days_out value recorded so far, showing
    how the predicted attendance changed as the game got closer.
  - Once actual_attendance is filled in (after the game is played), an
    accuracy breakdown by days_out bucket: how much better is a prediction
    made 1 day out vs. 2 weeks out?

Run standalone any time: python generate_prediction_report.py
Also called automatically at the end of predict_next_home_game.py.
"""

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Local (non-iCloud) cache — see sync_data.py for why, and how to move files
# between here and the iCloud project folder.
DATA_DIR = Path.home() / "Library" / "Application Support" / "mets-prediction" / "data"
LOG_PATH = DATA_DIR / "mets_2026_predictions_log.csv"
REPORT_PATH = DATA_DIR / "prediction_accuracy_report.html"


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


def safe_write_text(path, text, retries=5, delay=1.0):
    last_err = None
    for attempt in range(retries):
        try:
            with open(path, "w") as f:
                f.write(text)
            return
        except OSError as e:
            last_err = e
            time.sleep(delay)
    raise last_err


def days_out_bucket(days_out):
    if pd.isna(days_out):
        return "unknown"
    if days_out >= 14:
        return "14+ days out"
    if days_out >= 7:
        return "7-13 days out"
    if days_out >= 3:
        return "3-6 days out"
    return "0-2 days out"


def build_report():
    if not LOG_PATH.exists():
        print(f"No log file yet at {LOG_PATH} — run predict_next_home_game.py first.")
        return

    log = safe_read_csv(LOG_PATH)
    log["game_label"] = log["game_date"] + " vs " + log["opponent"]

    # One row per game, one column per days_out, showing how the prediction moved
    pivot = log.pivot_table(index="game_label", columns="days_out",
                             values="predicted_attendance", aggfunc="last")
    pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)
    pivot.columns = [f"{int(c)}d out" for c in pivot.columns]

    actuals = log.groupby("game_label")["actual_attendance"].last()
    game_dates = log.groupby("game_label")["game_date"].last()
    pivot = pivot.join(actuals).join(game_dates)
    pivot = pivot.sort_values("game_date")
    pivot = pivot.drop(columns="game_date")
    pivot = pivot.rename(columns={"actual_attendance": "Actual"})

    # Accuracy by days-out bucket, wherever we have an actual to compare against
    scored = log.dropna(subset=["actual_attendance"]).copy()
    if not scored.empty:
        scored["abs_error"] = (scored["predicted_attendance"] - scored["actual_attendance"]).abs()
        scored["abs_pct_error"] = scored["abs_error"] / scored["actual_attendance"] * 100
        scored["bucket"] = scored["days_out"].apply(days_out_bucket)
        bucket_order = ["14+ days out", "7-13 days out", "3-6 days out", "0-2 days out"]
        accuracy = scored.groupby("bucket").agg(
            n=("abs_error", "count"),
            mae=("abs_error", "mean"),
            mape=("abs_pct_error", "mean"),
        ).reindex(bucket_order).dropna(how="all")
    else:
        accuracy = pd.DataFrame()

    html_parts = [
        "<html><head><meta charset='utf-8'><title>Mets 2026 Prediction Accuracy</title>",
        "<style>",
        "body { font-family: -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; }",
        "h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }",
        "table { border-collapse: collapse; margin-top: 0.5rem; }",
        "th, td { border: 1px solid #ddd; padding: 6px 12px; text-align: right; font-size: 0.9rem; }",
        "th { background: #002D72; color: white; text-align: center; }",
        "td:first-child, th:first-child { text-align: left; }",
        ".actual-col { background: #fff3e0; font-weight: bold; }",
        "caption { text-align: left; font-size: 0.85rem; color: #666; margin-bottom: 0.3rem; }",
        "</style></head><body>",
        "<h1>Mets 2026 Attendance Predictions — Tracked Over Time</h1>",
        f"<p>Generated from {len(log)} logged prediction run(s) across {log['game_label'].nunique()} game(s).</p>",
        "<h2>Prediction by Days Before Game</h2>",
        "<table><caption>Each column is the predicted attendance when the run happened that many days before the game. "
        "'Actual' fills in once MLB posts the real number.</caption>",
        "<tr><th>Game</th>" + "".join(f"<th>{c}</th>" for c in pivot.columns) + "</tr>",
    ]
    for game, row in pivot.iterrows():
        cells = []
        for col in pivot.columns:
            val = row[col]
            cls = " class='actual-col'" if col == "Actual" else ""
            cells.append(f"<td{cls}>{'' if pd.isna(val) else f'{val:,.0f}'}</td>")
        html_parts.append(f"<tr><td>{game}</td>{''.join(cells)}</tr>")
    html_parts.append("</table>")

    html_parts.append("<h2>Accuracy by Lead Time</h2>")
    if accuracy.empty:
        html_parts.append("<p>No games with a recorded actual attendance yet — fill in "
                           "'actual_attendance' in mets_2026_predictions_log.csv once games are played.</p>")
    else:
        html_parts.append("<table><caption>Mean absolute error and percent error, grouped by how "
                           "far ahead of the game the prediction was made.</caption>")
        html_parts.append("<tr><th>Lead Time</th><th>N</th><th>MAE</th><th>MAPE</th></tr>")
        for bucket, row in accuracy.iterrows():
            html_parts.append(
                f"<tr><td>{bucket}</td><td>{int(row['n'])}</td>"
                f"<td>{row['mae']:,.0f}</td><td>{row['mape']:.1f}%</td></tr>"
            )
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    safe_write_text(REPORT_PATH, "\n".join(html_parts))
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    build_report()
