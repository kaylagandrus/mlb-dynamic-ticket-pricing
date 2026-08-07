#!/usr/bin/env python3
"""
Move files between the iCloud-synced project folder (mets/, used by the
notebooks and git) and the local, non-iCloud copies the daily prediction job
actually runs from (~/Library/Application Support/mets-prediction/).

Why two locations: launchd background jobs cannot reliably run scripts or
read/write data that live inside iCloud Drive. macOS's file-coordination and
TCC privacy protections conflict with unattended background processes at both
the exec/chdir level (starting the script at all) and the file-I/O level
(reading/writing data mid-run), confirmed by testing, not a guess. Interactive
runs (you, in Terminal, or Claude in a session) don't have this problem, so
syncing has to happen interactively, not automatically.

  --from-project   pull mets_master.csv + mets_promotions.csv (data), and
                    predict_next_home_game.py + generate_prediction_report.py
                    (scripts) from the project folder into the local copies.
                    Run this whenever you update those files via the notebooks
                    or edit the scripts.
  --to-project      push mets_2026_master.csv, mets_2026_predictions_log.csv,
                    and prediction_accuracy_report.html from the local cache
                    back into the project folder, e.g. before a git commit.
"""

import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DATA_DIR = PROJECT_DIR / "data"
PROJECT_SCRIPTS_DIR = PROJECT_DIR / "scripts"

LOCAL_DIR = Path.home() / "Library" / "Application Support" / "mets-prediction"
LOCAL_DATA_DIR = LOCAL_DIR / "data"
LOCAL_SCRIPTS_DIR = LOCAL_DIR / "scripts"
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

FROM_PROJECT_DATA_FILES = ["mets_master.csv", "mets_promotions.csv"]
FROM_PROJECT_SCRIPT_FILES = ["predict_next_home_game.py", "generate_prediction_report.py"]
TO_PROJECT_FILES = ["mets_2026_master.csv", "mets_2026_predictions_log.csv", "prediction_accuracy_report.html"]


def sync_from_project():
    for fname in FROM_PROJECT_DATA_FILES:
        src, dst = PROJECT_DATA_DIR / fname, LOCAL_DATA_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {fname}: project -> local data")
        else:
            print(f"  {fname}: not found in project data folder, skipped")

    for fname in FROM_PROJECT_SCRIPT_FILES:
        src, dst = PROJECT_SCRIPTS_DIR / fname, LOCAL_SCRIPTS_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {fname}: project -> local scripts")
        else:
            print(f"  {fname}: not found in project scripts folder, skipped")


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
        print("Syncing project folder -> local copies:")
        sync_from_project()
