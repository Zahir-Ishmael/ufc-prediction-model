#!/usr/bin/env python3
"""Refresh the UFC fight archive from nlbair/ufc-data. Python standard library only."""
import csv
import io
import json
import os
import pathlib
import tempfile
import urllib.request
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'ufc_fights_2015_present.csv'
STATUS = ROOT / 'data' / 'update_status.json'
URL = 'https://raw.githubusercontent.com/nlbair/ufc-data/main/data/ufc_all_fights_wide.csv'
START = date(2015, 1, 1)

def main():
    req = urllib.request.Request(URL, headers={'User-Agent': 'UFC-model-data-updater/1.0'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    if len(raw) < 100000:
        raise RuntimeError('Unexpectedly small source file; refusing to overwrite existing data')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    required = {'fight_id', 'event_date', 'f1_name', 'f2_name', 'winner'}
    if not required.issubset(reader.fieldnames or []):
        raise RuntimeError('Source schema changed; missing: ' + ', '.join(sorted(required - set(reader.fieldnames or []))))
    fields = reader.fieldnames
    fights = {}
    for row in reader:
        try:
            fight_date = date.fromisoformat(row['event_date'][:10])
        except (ValueError, TypeError):
            continue
        if fight_date < START or fight_date > date.today():
            continue
        key = row['fight_id']
        if not key or not row['f1_name'] or not row['f2_name']:
            continue
        fights[key] = row
    if len(fights) < 5000:
        raise RuntimeError(f'Only {len(fights)} valid fights; refusing to overwrite')
    rows = sorted(fights.values(), key=lambda r: (r['event_date'], r['fight_id']))
    if OUT.exists():
        with OUT.open(newline='', encoding='utf-8') as f:
            old = list(csv.DictReader(f))
        old_ids = {r['fight_id'] for r in old}
        new_ids = set(fights)
        if len(rows) < len(old) or not old_ids.issubset(new_ids):
            raise RuntimeError('Source dropped previously saved fights; refusing to overwrite')
        added = len(new_ids - old_ids)
    else:
        added = len(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', newline='', encoding='utf-8', dir=OUT.parent, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
        temp_name = tmp.name
    os.replace(temp_name, OUT)
    report = {'source': URL, 'checked_on_utc': date.today().isoformat(), 'first_fight_date': rows[0]['event_date'], 'latest_fight_date': rows[-1]['event_date'], 'fight_count': len(rows), 'new_fights': added, 'note': 'Only events present in upstream source; latest source date may lag behind today.'}
    STATUS.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
