import runpy, numpy as np, pandas as pd, json
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,log_loss,brier_score_loss
P='/mnt/data/'
s=runpy.run_path(P+'ufc_odds_backtest.py')
X=s['X'].copy();a=s['s']['a'];b=s['s']['b'];valid=s['valid'];train=s['train'];test=s['test'];y=s['y'];base=s['cols']
# Use training-period fighter histories only to determine style thresholds.
tr=pd.concat([a.loc[train],b.loc[train]])
metrics=['prior_strikes','prior_takedowns','prior_submissions','prior_control']
med={c:float(tr[c].median()) for c in metrics}
# Style categories: prioritize genuine grappling activity, differentiate by takedowns/control vs submission attempts.
def styles(z):
    labels=[]
    for _,r in z.iterrows():
        v={c:r[c] for c in metrics}
        if any(pd.isna(q) for q in v.values()): labels.append('unknown');continue
        striking=v['prior_strikes']>med['prior_strikes']
        wrestling=(v['prior_takedowns']>med['prior_takedowns']) or (v['prior_control']>med['prior_control'])
        submissions=v['prior_submissions']>med['prior_submissions']
        if striking and (wrestling or submissions):label='mixed'
        elif wrestling and submissions:
            # Larger relative departure from the historical training median decides primary grappling mode.
            wrest_strength=max(v['prior_takedowns']/(med['prior_takedowns']+1e-8),v['prior_control']/(med['prior_control']+1e-8))
            sub_strength=v['prior_submissions']/(med['prior_submissions']+1e-8)
            label='submission_grappler' if sub_strength>wrest_strength else 'wrestler'
        elif wrestling:label='wrestler'
        elif submissions:label='submission_grappler'
        elif striking:label='striker'
        else:label='low_activity'
        labels.append(label)
    return pd.Series(labels,index=z.index)
sa=styles(a);sb=styles(b)
for label in ['striker','wrestler','submission_grappler','mixed','low_activity']:
    X['refined_'+label]=(sa.eq(label).astype(int)-sb.eq(label).astype(int)).to_numpy()
refined=[c for c in X if c.startswith('refined_')]
# Replace original style proxies, preserve all pre-fight historical metrics and opening odds.
stat=[c for c in base if c.startswith('diff_')]
configs={'existing_stats_style':base,'refined_stats_style':stat+refined,'existing_combined_odds':base+['odds_logit'],'refined_combined_odds':stat+refined+['odds_logit']}
rows=[];pred=pd.DataFrame({'date':X.loc[test,'date'],'fighter_1':X.loc[test,'fighter_1'],'fighter_2':X.loc[test,'fighter_2'],'actual_f1_win':y[test], 'fighter_1_style':sa[test], 'fighter_2_style':sb[test]})
for name,cols in configs.items():
    model=make_pipeline(SimpleImputer(strategy='constant',fill_value=0),StandardScaler(),LogisticRegression(max_iter=2000))
    model.fit(X.loc[train,cols],y[train]);p=model.predict_proba(X.loc[test,cols])[:,1]
    pred[name+'_prob']=p
    rows.append({'model':name,'fights':int(test.sum()),'correct':int(((p>=.5)==y[test]).sum()),'accuracy':round(accuracy_score(y[test],p>=.5),4),'roc_auc':round(roc_auc_score(y[test],p),4),'log_loss':round(log_loss(y[test],p),4),'brier':round(brier_score_loss(y[test],p),4)})
pred.to_csv(P+'ufc_refined_styles_predictions.csv',index=False)
counts=pd.concat([sa[test],sb[test]]).value_counts().to_dict()
output={'test_fights':int(test.sum()),'train_fights':int(train.sum()),'training_medians':med,'fighter_appearance_style_counts_test':counts,'results':rows,'notes':'Pre-fight historical rates only; thresholds learned from pre-2023 train subset; styles are heuristics, not verified disciplines. Same matched odds subset used for all models.'}
with open(P+'ufc_refined_styles_results.json','w') as f:json.dump(output,f,indent=2)
print('REFINED_RESULTS',json.dumps(output,indent=2))
