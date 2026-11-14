import runpy, numpy as np, pandas as pd, json
from collections import defaultdict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,log_loss,brier_score_loss
P='/mnt/data/'
s=runpy.run_path(P+'ufc_closing_odds_backtest.py')
X=s['X'].copy();train=s['tr'];test=s['te'];y=s['y'];base=s['old'];norm=s['norm']
d=pd.read_csv(P+'checkpoint8_transformed_fighter_appearances.csv',parse_dates=['event_date'])
d=d.sort_values(['event_date','fight_url','side'])
hist=defaultdict(list);rows=[]
for date,g in d.groupby('event_date',sort=True):
    pending=[]
    for r in g.itertuples(index=False):
        name=norm(r.fighter);h=hist[name];last=h[-5:];last3=h[-3:]
        # W/L streak: positive wins, negative losses
        streak=0
        for v in reversed(h):
            sign=1 if v['win'] else -1
            if streak and (streak>0)!=(sign>0):break
            streak+=sign
        def avg(vals,field):
            q=[v[field] for v in vals if pd.notna(v[field])]
            return float(np.mean(q)) if q else np.nan
        # Compare recent 3 to earlier 3 (if available)
        earlier=h[-6:-3]
        def trend(field):
            x=avg(last3,field);z=avg(earlier,field)
            return x-z if pd.notna(x) and pd.notna(z) else np.nan
        feat={'date':date,'name':name,'streak':streak,'last5_weighted_wins':sum((i+1)*v['win'] for i,v in enumerate(last))/sum(range(1,len(last)+1)) if last else np.nan,
              'last3_finish_rate':avg(last3,'finish_win'), 'strike_trend':trend('strikes'), 'takedown_trend':trend('takedowns'), 'control_trend':trend('control'),
              'last3_opp_winrate':avg(last3,'opponent_prior_winrate'),'last3_fights_count':len(last3)}
        rows.append(feat)
        oh=hist[norm(r.opponent)]
        prior_opp=avg(oh,'win')
        finish=str(r.finish_type).lower() if pd.notna(r.finish_type) else ''
        is_finish=int(r.win==1 and any(t in finish for t in ['ko','tko','submission','sub']))
        pending.append((name,{'win':int(r.win),'finish_win':is_finish,'strikes':r.strikes_per_min,'takedowns':r.takedowns_per_min,'control':r.control_seconds_per_min,'opponent_prior_winrate':prior_opp}))
    for name,z in pending:hist[name].append(z)
features=pd.DataFrame(rows).drop_duplicates(['date','name'],keep=False).set_index(['date','name'])
left=pd.MultiIndex.from_arrays([pd.to_datetime(X.date),[norm(v) for v in X.fighter_1]])
right=pd.MultiIndex.from_arrays([pd.to_datetime(X.date),[norm(v) for v in X.fighter_2]])
a=features.reindex(left).reset_index(drop=True);b=features.reindex(right).reset_index(drop=True)
a.index=X.index;b.index=X.index
momentum=[]
for c in ['streak','last5_weighted_wins','last3_finish_rate','strike_trend','takedown_trend','control_trend','last3_opp_winrate','last3_fights_count']:
    name='momentum_'+c
    X[name]=pd.to_numeric(a[c],errors='coerce')-pd.to_numeric(b[c],errors='coerce')
    momentum.append(name)
results=[];pred=pd.DataFrame({'date':X.loc[test,'date'],'fighter_1':X.loc[test,'fighter_1'],'fighter_2':X.loc[test,'fighter_2'],'actual_f1_win':y[test],'closing_market_prob':X.loc[test,'closing_p']})
def evaluate(name,p):
    yy=y[test].to_numpy();p=np.asarray(p);results.append({'model':name,'n':len(yy),'correct':int(((p>=.5)==yy).sum()),'accuracy':round(accuracy_score(yy,p>=.5),4),'roc_auc':round(roc_auc_score(yy,p),4),'log_loss':round(log_loss(yy,p),4),'brier':round(brier_score_loss(yy,p),4)});pred[name+'_prob']=p

evaluate('closing_market',X.loc[test,'closing_p'])
for name,cols in [('advanced_closing',[c for c in base if c!='odds_logit']+['closing_logit']),('advanced_closing_momentum',[c for c in base if c!='odds_logit']+['closing_logit']+momentum),('advanced_open_momentum',base+momentum)]:
    model=make_pipeline(SimpleImputer(strategy='constant',fill_value=0),StandardScaler(),LogisticRegression(max_iter=3000))
    model.fit(X.loc[train,cols],y[train]);evaluate(name,model.predict_proba(X.loc[test,cols])[:,1])
for c in momentum:pred[c]=X.loc[test,c]
pred.to_csv(P+'ufc_momentum_predictions.csv',index=False)
res={'train_fights':int(train.sum()),'test_fights':int(test.sum()),'features':momentum,'coverage':{c:int(X.loc[test,c].notna().sum()) for c in momentum},'results':results,'notes':'Momentum calculated from fighter appearances strictly before each event date, updated in date batches. Last-5 weighted wins, signed streak, last-3 finish rate, trends comparing last three versus preceding three, recent opponent prior win rates. Closing odds are only available near fight time. No test-set tuning.'}
with open(P+'ufc_momentum_results.json','w') as f:json.dump(res,f,indent=2)
print('MOMENTUM_RESULT',json.dumps(res,indent=2))
