# MLB Ticket Pricing / Attendance Prediction

Predicting game-day attendance for MLB home games, as a foundation for dynamic ticket pricing. Built end to end: raw data collection from public APIs, feature engineering, an interpretable model benchmarked honestly against naive baselines, a walk-forward backtest, and a fully automated daily prediction pipeline that tracks real 2026 games as the season happens.

## Why the Mets, not the Padres

I started with the Padres (`padres/`). After the initial EDA, I switched to the Mets (`mets/`) for a few reasons:
1. Padres attendance sells out too consistently, there's not much variance left to predict.
2. San Diego's weather has essentially no impact on attendance since it's excellent nearly every game.
3. Day of week barely moves Padres attendance either, the flexible/work-from-home culture in San Diego blunts the usual weekday effect even for day games.
4. The Padres' on-field performance has been fairly consistent across these seasons, leaving less to explain.
5. Mets attendance swings much more dramatically based on opponent, giving the model a lot more real signal to work with.

`padres/` is left in the repo as-is, it's real analysis, just not where the project continued. All the modeling, backtesting, and automation below is Mets-only.

## Data sources

- **MLB Stats API** (`statsapi.mlb.com`), free and official: schedule, box scores, standings, and pitcher game logs.
- **Open-Meteo** (archive + forecast endpoints): hourly temperature and precipitation around game time at Citi Field.
- **Promotions calendar** (`mets_promotions.csv`): hand-compiled from MLB.com press releases and fan sites, no official API exists for this. A reasonable approximation, not a guaranteed-complete record.

Every "as of the game" feature (standings, starter ERA, win/loss streak) is computed point-in-time, using only what would have actually been known before that game was played, not season-end stats.

## Notebooks (`mets/notebooks/`)

| Notebook | What it does |
|---|---|
| `01_data_collection.ipynb` | Pull every Mets home game, 2022-2025 (training) and 2026 (test/live), from the MLB Stats API. |
| `02_weather_and_database.ipynb` | Pull weather for every game date, build a DuckDB database, join in promotions. |
| `02b_standings_and_pitchers.ipynb` | Add team record, streak, games back, and both starting pitchers' ERA entering the game. |
| `02c_broadcast_gametime_and_2026_master.ipynb` | Add national broadcast tier and game start time (day/twilight/night), assemble the final combined dataset for both training and the 2026 backtest. |
| `03_eda.ipynb` | Exploratory analysis: attendance by opponent, weather, promotions, day of week, broadcast tier. |
| `04_modeling.ipynb` | Linear regression (interpretability, chosen deliberately over a black-box model) plus Random Forest, Gradient Boosting, and an ensemble, benchmarked against four naive baselines using an honest R² (vs. the 2022-2025 historical mean, not the test set's own mean). |
| `05_expanding_window_backtest.ipynb` | Simulates a daily-refresh model: retrain on 2022-2025 plus every 2026 game played so far, predict the next one, walk forward game by game. This is what the automation below actually does. |
| `06_prediction_tracking.ipynb` | Reads the live automation's prediction log and shows how each prediction evolved as game day approached. |

## Honest baselines, not sklearn's default

Standard R² compares a model against the test set's own mean, which quietly assumes you already know the answer. Every R² in this project instead uses the 2022-2025 historical mean as a fixed reference point, the best of four candidate naive baselines tested in `04_modeling.ipynb` (historical mean, last year's average, same matchup last year, same weekday/month average). R²=0 means "tied with just guessing the historical average," negative means worse than that.

**The honest finding:** on a static, train-once model, none of the approaches (linear, Random Forest, Gradient Boosting, ensemble) beat the historical mean baseline against the full 2026 backtest. With only ~280 training rows and real season-to-season volatility, that's a legitimate result, not a bug. Retraining daily on an expanding window (`05_expanding_window_backtest.ipynb`) does change this: Gradient Boosting, Random Forest, and the ensemble all come back with genuinely positive R² under that approach. That result has already moved once as more 2026 games came in (MAPE went from 4.8% to 7.6% as the test set grew), which is itself the finding: with this little data, "beats the baseline" is a real but fragile, moving target.

## Daily automated predictions

`mets/scripts/predict_next_home_game.py` runs once a day via `launchd` and tracks every Mets home game within 9 days of game day:
- Refreshes the 2026 dataset with any newly completed games.
- Builds live features for each tracked game (weather forecast if the game hasn't happened, current standings, probable starter ERA if announced, promotions, broadcast info).
- Retrains Gradient Boosting on 2022-2025 plus every completed 2026 game, and predicts each tracked game.
- Logs every prediction to `mets_2026_predictions_log.csv`, so you can see how the same game's prediction moves over the days leading up to it as more information (forecast, probable starter) firms up.

Because the data and code live in iCloud Drive, and macOS blocks unattended background processes from reading/writing iCloud-synced files, the automation runs from a local copy in `~/Library/Application Support/mets-prediction/`. `mets/scripts/sync_data.py` moves data and scripts between there and this repo:

```bash
python mets/scripts/sync_data.py --from-project   # push latest data/scripts to the local deployment
python mets/scripts/sync_data.py --to-project     # pull the latest predictions log back here
```

## Running it

Requires a conda environment with `pandas`, `numpy`, `requests`, `duckdb`, `scikit-learn`, and `matplotlib` (this project used a `sports` env). Run the notebooks in order, 01 through 06, each one depends on the CSVs the previous ones produce.

**To run this for a different team:** everything is keyed off the team ID and the home ballpark's coordinates, both of which are otherwise generic. Update:
- `TEAM_ID` at the top of `01_data_collection.ipynb`, `02b_standings_and_pitchers.ipynb`, and `02c_broadcast_gametime_and_2026_master.ipynb` (find a team's ID via `https://statsapi.mlb.com/api/v1/teams?sportId=1`).
- The `lat`/`lon` default args in `02_weather_and_database.ipynb`'s `get_weather()`, to the new ballpark's coordinates.
- `TEAM_ID` and `LAT, LON` at the top of `mets/scripts/predict_next_home_game.py`, if you also want the daily automation running for that team.

## Status

Actively tracking the 2026 season. Plan is to revisit `04` and `05` once the season concludes and the backtest is final, and keep an eye on `06` as daily predictions come in over the remaining home games. This is attendance prediction only for now, a dynamic pricing layer on top of it is a planned next step, not built yet, held off on until the model has a consistently accurate MAPE. Will come back after the season concludes and see if it's good enough to use for pricing analytics.
