import runpy, pandas as pd, numpy as np, json
from collections import defaultdict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,log_loss,brier_score_loss
P='/mnt/data/'
s=runpy.run_path(P+'ufc_refined_styles_backtest.py')
X=s['X'].copy(); a=s['a'];b=s['b'];train=s['train'];test=s['test'];y=s['y'];sa=s['sa'];sb=s['sb'];base=s['stat']+s['refined']+['odds_logit']
wide=pd.read_csv(P+'ufc_all_fights_wide.csv',low_memory=False)
wide['event_date']=pd.to_datetime(wide.event_date,errors='coerce')
# Link each wide-row to an original fight via same-day normalized pair; only unique matches.
key=s['s']['key'];norm=s['s']['norm']
wide['match_key']=[key(d,f,g) for d,f,g in wide[['event_date','f1_name','f2_name']].itertuples(index=False,name=None)]
wide=wide.drop_duplicates('match_key',keep=False).sort_values('event_date')
# Calculate historical pre-fight features using date batches, preventing same-day lookahead.
hist=defaultdict(lambda: {'elo':1500.,'n':0,'opp_sig_att':0.,'opp_sig_landed':0.,'opp_td_att':0.,'opp_td_landed':0.})
records=[]
for dt,group in wide.groupby('event_date',sort=True):
    updates=[]
    for _,r in group.iterrows():
        n1,n2=norm(r.f1_name),norm(r.f2_name)
        if not n1 or not n2:continue
        h1,h2=hist[n1],hist[n2]
        def defense(h,att,land):
            return 1-land/att if att>0 else np.nan
        z={'match_key':r.match_key,'f1_name':r.f1_name,'f2_name':r.f2_name,
           'elo_diff':h1['elo']-h2['elo'],
           'str_def_diff':defense(h1,h1['opp_sig_att'],h1['opp_sig_landed'])-defense(h2,h2['opp_sig_att'],h2['opp_sig_landed']),
           'td_def_diff':defense(h1,h1['opp_td_att'],h1['opp_td_landed'])-defense(h2,h2['opp_td_att'],h2['opp_td_landed']),
           'reach_diff':pd.to_numeric(r.f1_reach_cm,errors='coerce')-pd.to_numeric(r.f2_reach_cm,errors='coerce')}
        records.append(z)
        if r.result_f1 not in ['W','L']:continue
        def num(v):
            t=pd.to_numeric(v,errors='coerce');return float(t) if pd.notna(t) else 0.
        updates.append((n1,n2,int(r.result_f1=='W'),
                        (num(r.f2_tot_sig_str_att),num(r.f2_tot_sig_str_landed),num(r.f2_tot_td_att),num(r.f2_tot_td_landed)),
                        (num(r.f1_tot_sig_str_att),num(r.f1_tot_sig_str_landed),num(r.f1_tot_td_att),num(r.f1_tot_td_landed))))
    for n1,n2,win,opp1,opp2 in updates:
        h1,h2=hist[n1],hist[n2]
        e1=1/(1+10**((h2['elo']-h1['elo'])/400));change=24*(win-e1)
        h1['elo']+=change;h2['elo']-=change
        for h,vals in [(h1,opp1),(h2,opp2)]:
            h['n']+=1
            for c,v in zip(['opp_sig_att','opp_sig_landed','opp_td_att','opp_td_landed'],vals):h[c]+=v
features=pd.DataFrame(records).drop_duplicates('match_key',keep=False).set_index('match_key')
joined=features.reindex(X['key'].tolist());joined.index=X.index
flip=np.array([norm(x)!=norm(z) for x,z in zip(X.fighter_1,joined.f1_name)])
for col in ['elo_diff','str_def_diff','td_def_diff','reach_diff']:
    X['adv_'+col]=np.where(flip,-pd.to_numeric(joined[col],errors='coerce'),pd.to_numeric(joined[col],errors='coerce'))
# Matchup interactions: offensive pre-fight takedown/striking rates vs opponent's pre-fight defense.
# Missing historical defense is imputed by pipeline; interactions are missing if component unavailable.
for label,rate,defense in [('wrestling','prior_takedowns','adv_td_def_diff'),('striking','prior_strikes','adv_str_def_diff')]:
    # Difference in attack rate multiplied by opposing defense gap: symmetric fighter orientation.
    X['adv_'+label+'_matchup']=(a[rate].to_numpy()+b[rate].to_numpy())*X[defense].to_numpy()
advanced=[c for c in X if c.startswith('adv_')]
configs={'previous_refined_odds':base,'advanced_refined_odds':base+advanced,'advanced_without_odds':[c for c in base if c!='odds_logit']+advanced}
out=[];pred=pd.DataFrame({'date':X.loc[test,'date'],'fighter_1':X.loc[test,'fighter_1'],'fighter_2':X.loc[test,'fighter_2'],'actual_f1_win':y[test]})
for name,cols in configs.items():
    clf=make_pipeline(SimpleImputer(strategy='constant',fill_value=0),StandardScaler(),LogisticRegression(max_iter=3000))
    clf.fit(X.loc[train,cols],y[train]);pr=clf.predict_proba(X.loc[test,cols])[:,1]
    pred[name+'_prob']=pr
    out.append({'model':name,'correct':int(((pr>=.5)==y[test]).sum()),'fights':int(test.sum()),'accuracy':round(accuracy_score(y[test],pr>=.5),4),'roc_auc':round(roc_auc_score(y[test],pr),4),'log_loss':round(log_loss(y[test],pr),4),'brier':round(brier_score_loss(y[test],pr),4)})
pred.to_csv(P+'ufc_advanced_features_predictions.csv',index=False)
result={'features':advanced,'matched_features':int(X.loc[test,'adv_elo_diff'].notna().sum()),'train_fights':int(train.sum()),'test_fights':int(test.sum()),'results':out,'notes':'Date-batched pre-fight Elo (K=24, initial 1500), cumulative opponent striking and takedown defenses, fighter reach, matchup interactions. Matched by unique same-date fighter pair. Odds from opening lines. No tuning on test data.'}
with open(P+'ufc_advanced_features_results.json','w') as f:json.dump(result,f,indent=2)
print('ADVANCED_FINAL',json.dumps(result,indent=2))
