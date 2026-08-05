#!/usr/bin/env python3
"""
Move files between the iCloud-synced project folder (mets/data/, used by the
notebooks and git) and the local, non-iCloud cache the daily prediction script
actually reads/writes from (~/Library/Application Support/mets-prediction/data/).

Why two locations: launchd background jobs cannot reliably read/write files
inside iCloud Drive (macOS's file-coordination + TCC privacy protections
conflict with unattended background processes — confirmed by testing, not a
guess). Interactive runs (you, in Terminal, or Claude in a session) don't have
this problem, so syncing has to happen interactively, not automatically.

  --from-project   pull mets_master.csv + mets_promotions.csv from the project
                    folder into the local cache. Run this whenever you update
                    those files via the notebooks.
  --to-project      push mets_2026_master.csv, mets_2026_predictions_log.csv,
                    and prediction_accuracy_report.html from the local cache
                    back into the project folder, e.g. before a git commit.
"""

import shutil
import sys
from pathlib import Path

PROJECT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_DATA_DIR = Path.home() / "Library" / "Application Support" / "mets-prediction" / "data"
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

FROM_PROJECT_FILES = ["mets_master.csv", "mets_promotions.csv"]
TO_PROJECT_FILES = ["mets_2026_master.csv", "mets_2026_predictions_log.csv", "prediction_accuracy_report.html"]


def sync_from_project():
    for fname in FROM_PROJECT_FILES:
        src, dst = PROJECT_DATA_DIR / fname, LOCAL_DATA_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {fname}: project -> local cache")
        else:
            print(f"  {fname}: not found in project folder, skipped")


def sync_to_project():
    for fname in TO_PROJECT_FILES:
        src, dst = LOCAL_DATA_DIR / fname, PROJECT_DATA_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {fname}: local cache -> project")
        else:
            print(f"  {fname}: not found in local cache, skipped")


if __name__ == "__main__":
    if "--to-project" in sys.argv:
        print("Syncing local cache -> project folder:")
        sync_to_project()
    else:
        print("Syncing project folder -> local cache:")
        sync_from_project()
