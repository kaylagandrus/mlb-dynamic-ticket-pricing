# Building an MLB Attendance Prediction Model: A Case Study

The goal: predict game-day attendance for MLB home games accurately enough to eventually inform dynamic ticket pricing. What follows is the actual decision-making process, including the parts that didn't work.

## Starting with the Padres, then switching to the Mets

I started this analysis on the San Diego Padres. After finishing the initial EDA, I switched to the Mets instead:

- Padres attendance sells out too consistently, there wasn't much variance left to predict.
- San Diego's weather has almost no impact on attendance since it's excellent nearly every game, one of the inputs I most wanted to use had nothing to say.
- Day of week barely moved Padres attendance either, the flexible/remote-work culture in San Diego blunts the usual weekday effect even for day games.
- The Padres' on-field performance was fairly consistent across the seasons I had data for, leaving less to explain.
- Mets attendance swings dramatically by opponent, giving the model real signal to work with.

Lesson: check whether your target variable actually has variance to explain before building anything. `padres/` stays in the repo as real, honest exploratory work, just not where the project continued.

## Building the pipeline

Data sources: the MLB Stats API (schedule, box scores, standings, pitcher game logs, all free and official), Open-Meteo for weather, and a hand-compiled promotions calendar (no official API exists for that one).

The decision that mattered most here: every "as of the game" feature (team record, streak, games back, starting pitcher ERA) is computed **point-in-time**, using only what would have actually been knowable before that specific game was played, not season-end stats. It's more API calls and more code, but a model trained on stats that include the future isn't predicting anything real.

## Modeling: interpretability over accuracy, and a different way to measure baselines

I used linear regression as the primary model, not because it's the most accurate (it usually wasn't), but because it mirrors how real MLB analytics groups actually work: a coefficient that says "a promo night adds ~X fans, holding everything else constant" is something you can explain to a front office; a random forest's feature importances aren't. Random Forest, Gradient Boosting, and an ensemble were tested alongside it to see if they yielded better results.

The bigger decision: how to grade any of these models. sklearn's default R² compares a model against the *test set's own mean*, which quietly assumes you already know the answer, not something you'd actually have before a season starts. I built four honest naive baselines instead (historical mean, last year's average, same matchup last year, same weekday/month average) and used the best of those, the historical mean, as the fixed reference point for every R² in the project. R²=0 means "tied with just guessing the average." Negative means worse than that.

## What I found (including the part that didn't work)

On a static, train-once model: **none of the approaches beat the historical mean baseline** against the full 2026 backtest. With ~280 training rows and real season-to-season volatility, that's an honest result, not a bug, and I'd rather report it than dress it up.

What did work: retraining daily on an expanding window (2022-2025 plus every 2026 game played so far) instead of fitting once in the offseason. Under that approach, Gradient Boosting, Random Forest, and the ensemble all came back with genuinely positive R², beating the baseline for real. That result already shifted once, MAPE went from 4.8% to 7.6% as more 2026 games came in, which is itself a finding: with this little data, "beats the baseline" is a real but fragile, moving target, not a permanent verdict.

## Debugging notes worth keeping

- **Duplicate rows from postponed games.** Games that got postponed and replayed later showed up twice in the pitcher schedule pull, once for the original date and once for the makeup date, each with a different probable-pitcher snapshot. Inflated my row count by 14 rows before I caught it by checking for duplicate `game_pk`s.
- **Doubleheader game-2s.** A cluster of badly overpredicted games turned out to all be nightcaps of a doubleheader, which draw noticeably smaller crowds. The MLB API already returns a field for this (`gameNumber`), I just hadn't asked for it. Added as `is_game2` across the whole pipeline once I found the pattern.
- **The daily automation wouldn't run unattended.** launchd jobs reading/writing files in iCloud Drive failed with `Resource deadlock avoided`, even from a native `cp` call, not a pandas issue. Root cause: macOS blocks unattended background processes from touching iCloud-synced files at all. Fix was moving both the data and the scripts themselves to a local, non-iCloud folder, with a small sync script to move things back and forth when working interactively.

## The daily prediction system

A script now runs every morning via `launchd`, tracking every Mets home game within 9 days of game day: refreshes the dataset with newly completed games, builds live features (weather forecast, current standings, probable starter ERA if announced), retrains Gradient Boosting on everything available, and logs a prediction for each tracked game. Because it logs every run, I can see how the same game's prediction shifts as game day approaches and more information firms up, which is the actual question worth asking of a system like this.

## What's next

This is attendance prediction only for now. A dynamic pricing layer on top is the planned next step, but I'm holding off until the model has a consistently accurate MAPE, not just an occasionally sufficient R² on a small backtest. Plan is to revisit once the 2026 season concludes and see if it's good enough to actually inform pricing.

Full code and notebooks: see the [README](README.md).
