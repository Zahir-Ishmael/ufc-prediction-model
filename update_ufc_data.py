
import csv
import json
import os
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

API = "https://mcp.fighterapi.com/v1/"
KEY = os.environ.get("FIGHTERAPI_KEY")

if not KEY:
    raise RuntimeError("FIGHTERAPI_KEY is missing from GitHub Secrets.")

RESULTS = DATA / "fighterapi_results.csv"

FIELDS = [
    "bout_id", "event_id", "event_date",
    "fighter_a", "fighter_b", "winner",
    "method", "end_round", "weight_class"
]


def request(endpoint, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API + endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": KEY
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def main():
    existing = {}

    if RESULTS.exists():
        with RESULTS.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["bout_id"]] = row

    # Check current and previous seasons.
    years = [date.today().year - 1, date.today().year]

    new_count = 0

    for year in years:
        offset = 0

        while True:
            rresponse = request("list_events", {
    "season": year,
    "limit": 50,
    "offset": offset
})

print(f"Year {year}: {len(response.get('events', []))} events returned")
           

            events = response.get("events", [])

            for event in events:
                event_id = str(event["event_id"])
                event_date = event.get("event_date", "")

                if not event_date or event_date > date.today().isoformat():
                    continue

                # Skip events whose bouts are already stored.
                if any(
                    row["event_id"] == event_id
                    for row in existing.values()
                ):
                    continue

                detail = request("get_event", {
                    "event_id": event_id
                })

                for bout in detail.get("bouts", []):
                    bout_id = str(bout.get("bout_id", ""))

                    if not bout_id or not bout.get("winner"):
                        continue

                    if bout_id in existing:
                        continue

                    existing[bout_id] = {
                        "bout_id": bout_id,
                        "event_id": event_id,
                        "event_date": event_date,
                        "fighter_a": bout.get("fighter_a_name", ""),
                        "fighter_b": bout.get("fighter_b_name", ""),
                        "winner": bout.get("winner", ""),
                        "method": bout.get("method", ""),
                        "end_round": bout.get("end_round", ""),
                        "weight_class": bout.get("weight_class", "")
                    }

                    new_count += 1

            next_offset = response.get("next_offset")
            if next_offset is None or not events:
                break

            offset = int(next_offset)

    with RESULTS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing.values())

    print(json.dumps({
        "new_fights": new_count,
        "total_api_fights": len(existing),
        "saved_to": str(RESULTS)
    }, indent=2))


if __name__ == "__main__":
    main()
