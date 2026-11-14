# UFC prediction — unified historical pipeline

## Included
- `update_ufc_data.py` downloads the latest *available* 2015-present archive from `nlbair/ufc-data`, validates IDs and prevents destructive data loss.
- `build_features.py` rebuilds pre-fight fighter histories (win rates, experience, Elo, striking, wrestling, submission activity, defensive stats, style proxies, momentum, age/reach, opponent quality, opening/closing no-vig odds) from that one archive. No results from the current fight are used to construct its inputs.
- `train_model.py` selects a logistic model on 2023–2024 validation, tests on 2025+, then refits a model on all available completed fights for future inference.
- `.github/workflows/update.yml` runs every Monday and can be manually triggered in GitHub Actions.

## GitHub setup
Extract the ZIP. Upload **all extracted files and folders**, preserving `.github/workflows/update.yml`, to the **repository root**. Existing files with the same names may be replaced. In GitHub Settings → Actions → General, enable Actions and allow read/write workflow permissions if your repository requires it. Go to Actions → Update UFC data and rebuild model → Run workflow.

## Caveats
- **This is not yet an automatic upcoming-card predictor**. It creates a refreshed trained model and historical validation results. Future matchup inference and card/odds ingestion need separate integration.
- Source updates can lag behind real UFC events. The source's latest event date is printed in `data/update_status.json`.
- Nationality, hometown, gym and travel features are **excluded**: historical snapshots aren't trustworthy enough to treat as pre-fight facts.
- Opening odds are useful for early forecasts; closing odds only become available close to fight time. We do not claim Thursday forecasts use closing odds.
- Style cutoffs are heuristic; derived feature definitions differ from previous backtests. Reported historical 69.36% is **not** an expected result for this rebuilt pipeline.
- Raw source contains completed fight stats; those are used only to update fighter history **after** each date, never as predictors of that same fight.
- The GitHub Actions run may require GitHub repository permissions to commit its outputs. On failure, inspect the workflow logs rather than assuming an update succeeded.
