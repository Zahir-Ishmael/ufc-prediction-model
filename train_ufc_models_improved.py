import pandas as pd, numpy as np, json
from collections import defaultdict,deque
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,log_loss
P='/mnt/data/'
d=pd.read_csv(P+'checkpoint8_transformed_fighter_appearances.csv',parse_dates=['event_date']).sort_values(['event_date','fight_url','side']).reset_index(drop=True)
f=pd.read_csv(P+'fighters_clean.csv',usecols=['full_name','dob_parsed']); f['dob_parsed']=pd.to_datetime(f.dob_parsed,errors='coerce'); f=f.dropna(subset=['full_name']).drop_duplicates('full_name',keep=False);dob=dict(zip(f.full_name,f.dob_parsed))
# Compute features simultaneously per event date to prevent same-date lookahead.
history=defaultdict(list); rows=[]
for date,group in d.groupby('event_date',sort=True):
    pending=[]
    for _,r in group.iterrows():
        h=history[r.fighter]; oh=history[r.opponent]
        recent=h[-3:]
        birth=dob.get(r.fighter,pd.NaT)
        age=(date-birth).days/365.25 if pd.notna(birth) else np.nan
        # Historical opponent strength at the time each past fight was fought, not future record.
        opponent_strength=np.mean([x['opp_pre_winrate'] for x in h if pd.notna(x['opp_pre_winrate'])]) if any(pd.notna(x['opp_pre_winrate']) for x in h) else np.nan
        feat={'fight_url':r.fight_url,'side':int(r.side),'date':date,'win':int(r.win),'fighter':r.fighter,
          'age':age,'experience':len(h),'prior_winrate':np.mean([x['win'] for x in h]) if h else np.nan,
          'recent3_winrate':np.mean([x['win'] for x in recent]) if recent else np.nan,
          'prior_opp_strength':opponent_strength,
          'recent3_strikes':np.mean([x['strikes_rate'] for x in recent]) if recent else np.nan,
          'recent3_takedowns':np.mean([x['takedowns_rate'] for x in recent]) if recent else np.nan,
          'recent3_control':np.mean([x['control_rate'] for x in recent]) if recent else np.nan,
          'prior_strikes':r.prior_strikes_rate,'prior_takedowns':r.prior_takedowns_rate,
          'prior_submissions':r.prior_submissions_rate,'prior_control':r.prior_control_seconds_rate,
          'layoff_log':np.log1p(max(0,r.layoff_days)) if pd.notna(r.layoff_days) else np.nan}
        rows.append(feat)
        pending.append((r.fighter,{'win':int(r.win),'opp_pre_winrate':np.mean([x['win'] for x in oh]) if oh else np.nan,
            'strikes_rate':r.strikes_per_min,'takedowns_rate':r.takedowns_per_min,'control_rate':r.control_seconds_per_min}))
    for fighter,entry in pending:history[fighter].append(entry)
features=pd.DataFrame(rows)
a=features[features.side==1].set_index('fight_url');b=features[features.side==2].set_index('fight_url');ix=a.index.intersection(b.index);a=a.loc[ix];b=b.loc[ix]
X=pd.DataFrame(index=ix); X['date']=a.date;X['y']=a.win
cols=['age','experience','prior_winrate','recent3_winrate','prior_opp_strength','recent3_strikes','recent3_takedowns','recent3_control','prior_strikes','prior_takedowns','prior_submissions','prior_control','layoff_log']
for c in cols:X['diff_'+c]=a[c].to_numpy()-b[c].to_numpy()
# Train-only thresholds for style proxy, applied to pre-fight historical rates.
train_mask=X.date<'2023-01-01'; tr=features[features.date<'2023-01-01']
med={c:tr[c].median() for c in ['prior_strikes','prior_takedowns','prior_control']}
def style(df):
  known=df[['prior_strikes','prior_takedowns','prior_control']].notna().all(axis=1)
  s=df.prior_strikes>med['prior_strikes'];g=(df.prior_takedowns>med['prior_takedowns'])|(df.prior_control>med['prior_control'])
  return np.select([~known,s&g,s, g],['unknown','mixed','striking','grappling'],default='low_activity')
sa=style(a);sb=style(b)
for v in ['striking','grappling','mixed','low_activity']:X['style_'+v]=(sa==v).astype(int)-(sb==v).astype(int)
train=X.date<'2023-01-01';test=~train
sets={'historical': [c for c in X if c.startswith('diff_') and c not in ['diff_age','diff_prior_winrate','diff_recent3_winrate','diff_prior_opp_strength','diff_recent3_strikes','diff_recent3_takedowns','diff_recent3_control']],
      'improved': [c for c in X if c.startswith('diff_')],
      'improved_plus_style':[c for c in X if c.startswith('diff_') or c.startswith('style_')]}
results=[]
for featureset in ['historical','improved','improved_plus_style']:
  cols=sets[featureset]
  for model in ['logistic','forest']:
    clf=make_pipeline(SimpleImputer(strategy='constant',fill_value=0),StandardScaler(),LogisticRegression(max_iter=2000)) if model=='logistic' else make_pipeline(SimpleImputer(strategy='constant',fill_value=0),RandomForestClassifier(n_estimators=250,min_samples_leaf=12,max_features='sqrt',random_state=42,n_jobs=-1))
    clf.fit(X.loc[train,cols],X.loc[train,'y']); p=clf.predict_proba(X.loc[test,cols])[:,1]
    results.append({'features':featureset,'model':model,'accuracy':round(accuracy_score(X.loc[test,'y'],p>=0.5),4),'auc':round(roc_auc_score(X.loc[test,'y'],p),4),'log_loss':round(log_loss(X.loc[test,'y'],p),4)})
output={'train_fights':int(train.sum()),'test_fights':int(test.sum()),'first_fighter_baseline_accuracy':round(float(X.loc[test,'y'].mean()),4),'results':results,'notes':'Age from unique name-matched DOB; ambiguous names excluded from DOB mapping. Opponent strength uses only opponents prior to each historical fight. Date batches avoid same-day lookahead. Style proxy is not independently verified.'}
with open(P+'ufc_improved_model_results.json','w') as f:json.dump(output,f,indent=2)
print(json.dumps(output,indent=2))
