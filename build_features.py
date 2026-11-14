#!/usr/bin/env python3
"""Rebuild leakage-safe pre-fight UFC features from the master archive (2015 onward).
The first fight on each date sees only fights on earlier dates.
"""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
SRC=DATA/'ufc_fights_2015_present.csv'
if not SRC.exists(): SRC=DATA/'ufc_all_fights_wide.csv'
OUT=DATA/'prefight_features.csv'

def num(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError,ValueError): return np.nan

def mean(vals):
    v=[x for x in vals if pd.notna(x)]
    return float(np.mean(v)) if v else np.nan

def ratio(n,d):
    return n/d if pd.notna(n) and pd.notna(d) and d>0 else np.nan

def fighter_snapshot(r,s,h,ratings):
    recent=h[-5:]; last3=h[-3:]
    vals={
      'age':num(r[f'{s}_age']), 'reach':num(r[f'{s}_reach_cm']),
      'experience':len(h),'elo':ratings.get(str(r[f'{s}_id']),1500.),
      'career_winrate':mean([e['win'] for e in h]),
      'recent3_winrate':mean([e['win'] for e in last3]),
      'recent5_weighted_winrate':ratio(sum((i+1)*e['win'] for i,e in enumerate(recent)),sum(range(1,len(recent)+1))) if recent else np.nan,
      'opponent_elo':mean([e['opponent_elo'] for e in h]),
      'strikes_per_min':mean([e['strikes_per_min'] for e in h]),
      'takedowns_per_min':mean([e['takedowns_per_min'] for e in h]),
      'submissions_per_min':mean([e['submissions_per_min'] for e in h]),
      'control_per_min':mean([e['control_per_min'] for e in h]),
      'striking_defense':mean([e['striking_defense'] for e in h]),
      'takedown_defense':mean([e['takedown_defense'] for e in h]),
      'recent_striking':mean([e['strikes_per_min'] for e in last3]),
      'recent_control':mean([e['control_per_min'] for e in last3]),
      'recent_finish_rate':mean([e['finish_win'] for e in recent]),
      'recent_opponent_elo':mean([e['opponent_elo'] for e in recent]),
      'recent_fight_count':len(recent),
      'layoff_days':(r['event_date']-h[-1]['date']).days if h else np.nan,
    }
    if h:
        streak=0; last=h[-1]['win']
        for e in reversed(h):
            if e['win']!=last:break
            streak+=1
        vals['streak']=streak if last else -streak
    else:vals['streak']=0
    for field in ['strikes_per_min','takedowns_per_min','control_per_min']:
        vals['trend_'+field]=mean([e[field] for e in h[-3:]])-mean([e[field] for e in h[-6:-3]]) if len(h)>=6 else np.nan
    # Style is a pre-fight proxy, not a verified martial arts background.
    if len(h)<2: style='unknown'
    else:
        striking=vals['strikes_per_min']>=3.0 if pd.notna(vals['strikes_per_min']) else False
        wrestling=(vals['takedowns_per_min']>=0.35 if pd.notna(vals['takedowns_per_min']) else False) or (vals['control_per_min']>=0.65 if pd.notna(vals['control_per_min']) else False)
        submission=vals['submissions_per_min']>=0.15 if pd.notna(vals['submissions_per_min']) else False
        style='mixed' if striking and (wrestling or submission) else 'submission' if submission else 'wrestler' if wrestling else 'striker' if striking else 'low_activity'
    vals['style']=style
    return vals

def entry(r,s,other,win,opponent_elo):
    duration=num(r.get('fight_duration_sec'))
    minutes=duration/60 if pd.notna(duration) and duration>0 else np.nan
    def v(name):return num(r.get(f'{s}_tot_{name}'))
    def opp(name):return num(r.get(f'{other}_tot_{name}'))
    method=str(r.get('finish_type','')).lower()
    return {'date':r['event_date'],'win':int(win),'opponent_elo':opponent_elo,
      'strikes_per_min':ratio(v('sig_str_landed'),minutes),
      'takedowns_per_min':ratio(v('td_landed'),minutes),
      'submissions_per_min':ratio(v('sub_att'),minutes),
      'control_per_min':ratio(v('ctrl_sec'),duration) * 60 if pd.notna(duration) and duration>0 else np.nan,
      'striking_defense':1-ratio(opp('sig_str_landed'),opp('sig_str_att')) if pd.notna(ratio(opp('sig_str_landed'),opp('sig_str_att'))) else np.nan,
      'takedown_defense':1-ratio(opp('td_landed'),opp('td_att')) if pd.notna(ratio(opp('td_landed'),opp('td_att'))) else np.nan,
      'finish_win':int(bool(win) and ('ko' in method or 'sub' in method or 'tko' in method))}

def main():
    d=pd.read_csv(SRC,low_memory=False); required=['fight_id','event_date','f1_id','f2_id','f1_name','f2_name','winner','f1_age','f2_age','f1_reach_cm','f2_reach_cm']
    missing=[c for c in required if c not in d]
    if missing:raise ValueError(f'Missing columns: {missing}')
    d['event_date']=pd.to_datetime(d.event_date,errors='coerce');d=d[(d.event_date>=pd.Timestamp('2015-01-01')) & (d.event_date<=pd.Timestamp.today())]
    d=d.dropna(subset=['fight_id','event_date','f1_id','f2_id']).drop_duplicates('fight_id').sort_values(['event_date','fight_id'])
    history=defaultdict(list);ratings={};rows=[];counts={'winner_unknown':0,'no_odds':0}
    for date,group in d.groupby('event_date',sort=True):
        pending=[]
        for _,r in group.iterrows():
            id1,id2=str(r.f1_id),str(r.f2_id)
            s1=fighter_snapshot(r,'f1',history[id1],ratings);s2=fighter_snapshot(r,'f2',history[id2],ratings)
            # Winner source varies: compare actual stored winner to both name and fighter-id.
            w=str(r.winner).strip().lower(); w=w[:-2] if w.endswith('.0') else w; n1=str(r.f1_name).strip().lower();n2=str(r.f2_name).strip().lower()
            if w in [n1,id1.lower(),'f1','fighter 1','1']: y=1
            elif w in [n2,id2.lower(),'f2','fighter 2','2']:y=0
            else:
                counts['winner_unknown']+=1;continue
            row={'fight_id':r.fight_id,'date':date.strftime('%Y-%m-%d'),'fighter_1':r.f1_name,'fighter_2':r.f2_name,'f1_id':id1,'f2_id':id2,'y':y,
                 'event_country':r.get('event_country'),'event_state':r.get('event_state'),'event_city':r.get('event_city')}
            for k in s1:
                if k=='style':
                    row['f1_style']=s1[k];row['f2_style']=s2[k]
                else:row['diff_'+k]=s1[k]-s2[k] if pd.notna(s1[k]) and pd.notna(s2[k]) else np.nan
            for sty in ['striker','wrestler','submission','mixed','low_activity']:
                row['style_'+sty]=int(s1['style']==sty)-int(s2['style']==sty)
            row['wrestling_matchup']=s1['takedowns_per_min']*(1-s2['takedown_defense'])-s2['takedowns_per_min']*(1-s1['takedown_defense']) if all(pd.notna(z) for z in [s1['takedowns_per_min'],s2['takedowns_per_min'],s1['takedown_defense'],s2['takedown_defense']]) else np.nan
            row['striking_matchup']=s1['strikes_per_min']*(1-s2['striking_defense'])-s2['strikes_per_min']*(1-s1['striking_defense']) if all(pd.notna(z) for z in [s1['strikes_per_min'],s2['strikes_per_min'],s1['striking_defense'],s2['striking_defense']]) else np.nan
            for kind in ['open','close']:
                col=f'f1_{kind}_prob_novig';row[f'{kind}_prob']=num(r.get(col))
                if pd.notna(row[f'{kind}_prob']) and not 0<row[f'{kind}_prob']<1:row[f'{kind}_prob']=np.nan
            if pd.isna(row['open_prob']) or pd.isna(row['close_prob']):counts['no_odds']+=1
            rows.append(row)
            elo1=ratings.get(id1,1500.);elo2=ratings.get(id2,1500.)
            exp=1/(1+10**((elo2-elo1)/400));delta=24*(y-exp)
            pending.extend([(id1,entry(r,'f1','f2',y,elo2),elo1+delta),(id2,entry(r,'f2','f1',1-y,elo1),elo2-delta)])
        # Delay all updates until after the event date to prevent same-day leakage.
        for fid,e,new_rating in pending:history[fid].append(e);ratings[fid]=new_rating
    result=pd.DataFrame(rows)
    result.to_csv(OUT,index=False)
    report={'source_file':str(SRC.name),'fight_count':len(result),'first_date':str(result.date.min()),'last_date':str(result.date.max()),'unknown_winner_excluded':counts['winner_unknown'],'missing_either_odds':counts['no_odds'],'features':list(result.columns),'warning':'No nationality/hometown/training-location features included: snapshots are not historically verified. Closing odds usable only immediately pre-fight.'}
    (DATA/'feature_status.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='features'},indent=2))
if __name__=='__main__':main()
