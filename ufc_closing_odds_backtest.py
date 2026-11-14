import runpy, numpy as np, pandas as pd, json
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
P='/mnt/data/'
s=runpy.run_path(P+'ufc_advanced_features_backtest.py')
X=s['X'].copy(); train=s['train'];test=s['test'];y=s['y'];base=s['base'];advanced=s['advanced'];old=base+advanced
odds=pd.read_csv(P+'ufc_all_fights_wide.csv',usecols=['event_date','f1_name','f2_name','result_f1','f1_close_prob_novig'])
key=s['s']['s']['key'];norm=s['s']['s']['norm']
odds['key']=[key(*v) for v in odds[['event_date','f1_name','f2_name']].itertuples(index=False,name=None)]
odds=odds.drop_duplicates('key',keep=False).set_index('key')
match=odds.reindex(X['key'].tolist()).reset_index(drop=True);match.index=X.index
flip=np.array([norm(a)!=norm(b) for a,b in zip(X.fighter_1,match.f1_name)])
p=pd.to_numeric(match.f1_close_prob_novig,errors='coerce').to_numpy();p=np.where(flip,1-p,p)
X['closing_p']=p;X['closing_logit']=np.log(np.clip(p,.001,.999)/(1-np.clip(p,.001,.999)))
X['line_movement']=X.closing_logit-X.odds_logit
valid=train|test
valid=valid & X.closing_p.between(0,1) & match.result_f1.isin(['W','L']).to_numpy()
tr=train&valid;te=test&valid
pred=pd.DataFrame({'date':X.loc[te,'date'],'fighter_1':X.loc[te,'fighter_1'],'fighter_2':X.loc[te,'fighter_2'],'actual_f1_win':y[te], 'opening_market_prob':X.loc[te,'odds_p'],'closing_market_prob':X.loc[te,'closing_p']})
results=[]
def evaluate(name,probs):
    probs=np.asarray(probs);actual=y[te].to_numpy();guess=(probs>=.5).astype(int)
    results.append({'model':name,'n':len(actual),'correct':int((guess==actual).sum()),'accuracy':round(accuracy_score(actual,guess),4),'roc_auc':round(roc_auc_score(actual,probs),4),'log_loss':round(log_loss(actual,probs),4),'brier':round(brier_score_loss(actual,probs),4)})
    pred[name+'_prob']=probs
evaluate('opening_market',X.loc[te,'odds_p'])
evaluate('closing_market',X.loc[te,'closing_p'])
for name,cols in [('previous_advanced_open',old),('advanced_closing', [c for c in old if c!='odds_logit']+['closing_logit']),('advanced_open_and_close',old+['closing_logit'])]:
    clf=make_pipeline(SimpleImputer(strategy='constant',fill_value=0),StandardScaler(),LogisticRegression(max_iter=3000))
    clf.fit(X.loc[tr,cols],y[tr]);evaluate(name,clf.predict_proba(X.loc[te,cols])[:,1])
pred.to_csv(P+'ufc_closing_odds_predictions.csv',index=False)
res={'matched_train':int(tr.sum()),'matched_test':int(te.sum()),'previous_test':int(test.sum()),'results':results,'note':'Retrospective closing odds as supplied in source, same matched sample, training before 2023; closing prices may not be available until near fight start. Closing probabilities taken from source no-vig columns.'}
with open(P+'ufc_closing_odds_results.json','w') as f:json.dump(res,f,indent=2)
print('CLOSING_RESULT',json.dumps(res,indent=2))
