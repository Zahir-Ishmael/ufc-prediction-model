UFC prediction model — existing research/backtest package

Contains the historical datasets and the Python backtest scripts, including the momentum model.

IMPORTANT: This is NOT yet an automatically updating model. The scripts use absolute
/mnt/data/ paths and were built for the ChatGPT analysis environment. Uploading this ZIP
to GitHub alone will not create scheduled updates or live predictions. The odds file
is a historical snapshot, not a live odds feed.

To run the momentum backtest in a Python environment with dependencies installed:
  python ufc_momentum_backtest.py
The chain of imported scripts is:
  ufc_momentum_backtest.py -> ufc_closing_odds_backtest.py ->
  ufc_advanced_features_backtest.py -> ufc_refined_styles_backtest.py ->
  ufc_odds_backtest.py -> train_ufc_models_improved.py

Dependencies: pandas, numpy, scikit-learn.

For GitHub, unzip first and upload the contents, not the ZIP itself, if you want
GitHub Actions to access the individual scripts.
