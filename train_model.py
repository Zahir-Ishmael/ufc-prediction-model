#!/usr/bin/env python3
"""Chronological validation and latest-model fitting; does NOT create future card predictions."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,brier_score_loss,log_loss
import joblib
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
d=pd.read_csv(DATA/'prefight_features.csv')
d['date']=pd.to_datetime(d['date']); d=d.sort_values('date')
features=[c for c in d if c.startswith(('diff_','style_')) or c in ('wrestling_matchup','striking_matchup')]
# Hold out 2025+ for honest evaluation; 2023-24 validation used only for model selection.
train=d.date<'2023-01-01'; valid=(d.date>='2023-01-01')&(d.date<'2025-01-01'); test=d.date>='2025-01-01'
variants={'stats_only':features,'stats_plus_open':features+['open_prob'],'stats_plus_close':features+['close_prob'],'close_only':['close_prob']}
results={};best=None
for name,cols in variants.items():
    # Closing/opening odds may be absent: evaluate on observed odds rows only.
    needed=[x for x in cols if x.endswith('_prob')]
    mask=d[needed].notna().all(axis=1) if needed else pd.Series(True,index=d.index)
    tr=train&mask;va=valid&mask;te=test&mask
    if min(tr.sum(),va.sum(),te.sum())<50:continue
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=False),StandardScaler(),LogisticRegression(max_iter=3000))
    model.fit(d.loc[tr,cols],d.loc[tr,'y'])
    pv=model.predict_proba(d.loc[va,cols])[:,1]
    validation_brier=float(brier_score_loss(d.loc[va,'y'],pv))
    # Choose only on validation period, never on test.
    if best is None or validation_brier<best[0]:best=(validation_brier,name,cols)
    results[name]={'validation_n':int(va.sum()),'validation_accuracy':round(float(accuracy_score(d.loc[va,'y'],pv>=0.5)),4),'validation_brier':round(validation_brier,4)}
if best is None:raise RuntimeError('Insufficient training/validation data')
_,name,cols=best
mask=d[[x for x in cols if x.endswith('_prob')]].notna().all(axis=1) if any(x.endswith('_prob') for x in cols) else pd.Series(True,index=d.index)
fit=(train|valid)&mask;te=test&mask
model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=3000))
model.fit(d.loc[fit,cols],d.loc[fit,'y']);p=model.predict_proba(d.loc[te,cols])[:,1]
report={'selection_rule':'Lowest Brier score on 2023–2024 validation','selected_model':name,'train_through':'2024-12-31','test_period':'2025 onward','test_n':int(te.sum()),'test_accuracy':round(float(accuracy_score(d.loc[te,'y'],p>=0.5)),4),'test_auc':round(float(roc_auc_score(d.loc[te,'y'],p)),4),'test_brier':round(float(brier_score_loss(d.loc[te,'y'],p)),4),'validation':results,'caveat':'This is a NEW model with new feature definitions, not a direct replication of previous 69.36% results. The stored production model is refit on all completed fights after evaluation.'}
(DATA/'validation_results.json').write_text(json.dumps(report,indent=2))
# Refit after the test is evaluated, for future use only; no test claim from this refit.
allrows=mask
model.fit(d.loc[allrows,cols],d.loc[allrows,'y']);joblib.dump({'pipeline':model,'features':cols,'selected_variant':name,'last_fight_date':str(d.date.max().date())},DATA/'latest_model.joblib')
print(json.dumps(report,indent=2))
