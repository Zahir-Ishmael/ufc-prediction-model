import runpy, pandas as pd, numpy as np, json, unicodedata, re
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,log_loss,brier_score_loss
p='/mnt/data/'
# Rebuild the original, leakage-safe pre-fight feature matrix.
s=runpy.run_path(p+'train_ufc_models_improved.py');X=s['X'].copy();fights=pd.read_csv(p+'fights_clean.csv',usecols=['fight_url','event_date','fighter_1','fighter_2']);odds=pd.read_csv(p+'ufc_all_fights_wide.csv',usecols=['event_date','f1_name','f2_name','result_f1','f1_open_prob_novig','f1_close_prob_novig','has_odds'])
def norm(v):
 if pd.isna(v):return ''
 return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',str(v)).encode('ascii','ignore').decode().lower())
def key(date,a,b):return (str(date)[:10],)+tuple(sorted([norm(a),norm(b)]))
odds=odds[(odds.event_date>='2015-01-01')&(odds.event_date<='2025-12-31')].copy();odds['key']=[key(*t) for t in odds[['event_date','f1_name','f2_name']].itertuples(index=False,name=None)]
odds=odds.drop_duplicates('key',keep=False).set_index('key')
fights['key']=[key(*t) for t in fights[['event_date','fighter_1','fighter_2']].itertuples(index=False,name=None)];fights=fights.drop_duplicates('fight_url').set_index('fight_url')
X['key']=fights.reindex(X.index)['key'];X['fighter_1']=fights.reindex(X.index)['fighter_1'];X['fighter_2']=fights.reindex(X.index)['fighter_2']
match=odds.reindex(X['key'].tolist()).reset_index(drop=True); match.index=X.index
flip=np.array([norm(a)!=norm(b) for a,b in zip(X.fighter_1,match.f1_name)])
prob=pd.to_numeric(match.f1_open_prob_novig,errors='coerce').to_numpy();prob=np.where(flip,1-prob,prob)
X['odds_p']=prob;X['odds_logit']=np.log(np.clip(prob,0.001,0.999)/(1-np.clip(prob,0.001,0.999)))
valid=X.odds_p.notna() & X.odds_p.between(0,1) & match.result_f1.isin(['W','L']).to_numpy()
train=(X.date<'2023-01-01') & valid;test=(X.date>='2023-01-01') & valid
cols=s['sets']['improved_plus_style'];y=X.y.astype(int)
records=[];preds=pd.DataFrame({'date':X.loc[test,'date'],'fighter_1':X.loc[test,'fighter_1'],'fighter_2':X.loc[test,'fighter_2'],'actual_f1_win':y[test]})
def metric(label,pr):
 pred=(pr>=.5).astype(int);records.append({'model':label,'accuracy':round(accuracy_score(y[test],pred),4),'roc_auc':round(roc_auc_score(y[test],pr),4),'log_loss':round(log_loss(y[test],pr),4),'brier':round(brier_score_loss(y[test],pr),4),'correct':int((pred==y[test]).sum())});preds[label+'_prob']=pr
metric('betting_odds_only',X.loc[test,'odds_p'].to_numpy())
for name,features in [('historical_model',cols),('combined_model',cols+['odds_logit'])]:
 clf=make_pipeline(SimpleImputer(strategy='constant',fill_value=0),StandardScaler(),LogisticRegression(max_iter=2000));clf.fit(X.loc[train,features],y[train]);metric(name,clf.predict_proba(X.loc[test,features])[:,1])
preds.to_csv(p+'ufc_odds_backtest_predictions.csv',index=False)
output={'source':'nlbair/ufc-data uploaded wide CSV; opening odds (no-vig)', 'original_model_fights':len(X),'matched_train':int(train.sum()),'matched_test':int(test.sum()),'all_test_fights':int((X.date>='2023-01-01').sum()),'results':records,'notes':'Only exactly matched same-date fighter pairs; draws/no contests and missing odds excluded. Odds model trained on same matched fights as combined model. First fighter orientation aligned by fighter name. Observational retrospective evaluation.'}
with open(p+'ufc_odds_backtest_results.json','w') as f:json.dump(output,f,indent=2)
print(json.dumps(output,indent=2))
