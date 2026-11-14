
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
    raise RuntimeError(
        "FIGHTERAPI_KEY is missing from GitHub Secrets."
    )

RESULTS = DATA / "fighterapi_results.csv"

FIELDS = [
    "bout_id",
    "event_id",
    "event_date",
    "fighter_a",
    "fighter_b",
    "winner",
    "method",
    "end_round",
    "weight_class",
]


def request(endpoint, payload):
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        API + endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": KEY,
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def main():
    existing = {}

    if RESULTS.exists():
        with RESULTS.open(
            newline="", encoding="utf-8"
        ) as f:
            for row in csv.DictReader(f):
                if row.get("bout_id"):
                    existing[row["bout_id"]] = row

    # Check current and previous years.
    years = [
        date.today().year - 1,
        date.today().year,
    ]

    new_count = 0
    checked_events = 0

    for year in years:
        offset = 0

        while True:
            response = request(
                "list_events",
                {
                    "season": year,
                    "status": "final",
                    "league": "mma",
                    "limit": 50,
                    "offset": offset,
                },
            )

            events = response.get("events", [])

            print(
                f"Year {year}, offset {offset}: "
                f"{len(events)} events returned"
            )

            for event in events:
                event_id = str(event.get("event_id", ""))
                event_date = event.get("event_date", "")

                if not event_id or not event_date:
                    continue

                if event_date > date.today().isoformat():
                    continue

                checked_events += 1

                # Fetch event details and fight results.
                detail = request(
                    "get_event",
                    {"event_id": event_id},
                )

                for bout in detail.get("bouts", []):
                    bout_id = str(bout.get("bout_id", ""))

                    if not bout_id:
                        continue

                    if not bout.get("winner"):
                        continue

                    if bout_id in existing:
                        continue

                    existing[bout_id] = {
                        "bout_id": bout_id,
                        "event_id": event_id,
                        "event_date": event_date,
                        "fighter_a": bout.get(
                            "fighter_a_name", ""
                        ),
                        "fighter_b": bout.get(
                            "fighter_b_name", ""
                        ),
                        "winner": bout.get("winner", ""),
                        "method": bout.get("method", ""),
                        "end_round": bout.get(
                            "end_round", ""
                        ),
                        "weight_class": bout.get(
                            "weight_class", ""
                        ),
                    }

                    new_count += 1

            next_offset = response.get("next_offset")

            if next_offset is None or not events:
                break

            next_offset = int(next_offset)

            if next_offset <= offset:
                raise RuntimeError(
                    "API pagination did not advance."
                )

            offset = next_offset

    # Save updated results.
    with RESULTS.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDS
        )
        writer.writeheader()

        writer.writerows(
            sorted(
                existing.values(),
                key=lambda row: (
                    row["event_date"],
                    row["bout_id"],
                ),
            )
        )

    print(
        json.dumps(
            {
                "new_fights": new_count,
                "total_api_fights": len(existing),
                "events_checked": checked_events,
                "saved_to": str(RESULTS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
