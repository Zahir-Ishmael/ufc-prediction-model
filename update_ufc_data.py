
import csv
import json
import os
import urllib.error
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

    try:
        with urllib.request.urlopen(
            req, timeout=45
        ) as response:
            result = json.load(response)

    except urllib.error.HTTPError as error:
        message = error.read().decode(
            "utf-8", errors="replace"
        )
        raise RuntimeError(
            f"FighterAPI HTTP {error.code}: {message}"
        ) from error

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected API response: {type(result)}"
        )

    if result.get("error"):
        raise RuntimeError(
            f"FighterAPI error: {result['error']}"
        )

    return result


def main():
    existing = {}

    if RESULTS.exists():
        with RESULTS.open(
            newline="", encoding="utf-8"
        ) as f:
            for row in csv.DictReader(f):
                bout_id = row.get("bout_id")

                if bout_id:
                    existing[bout_id] = row

    current_year = date.today().year

    # Initial backfill: 2015 through current year.
    # Later runs: current and previous year.
    if existing:
        years = [
            current_year - 1,
            current_year,
        ]
    else:
        years = range(2015, current_year + 1)

    new_count = 0
    checked_events = 0
    returned_events = 0

    for year in years:
        offset = 0

        while True:
            response = request(
                "list_events",
                {
                    "season": year,
                    "league": "mma",
                    "limit": 50,
                    "offset": offset,
                },
            )

            events = response.get("events")

            if not isinstance(events, list):
                raise RuntimeError(
                    f"Unexpected list_events response "
                    f"for {year}: "
                    f"{json.dumps(response)[:1500]}"
                )

            print(
                f"Year {year}, offset {offset}: "
                f"{len(events)} events returned"
            )

            returned_events += len(events)

            # Show diagnostic information if empty.
            if not events:
                print(
                    "API response keys:",
                    list(response.keys()),
                )
                print(
                    "API response preview:",
                    json.dumps(response)[:1500],
                )

            for event in events:
                event_id = str(
                    event.get("event_id") or ""
                )
                event_date = str(
                    event.get("event_date") or ""
                )

                if not event_id or not event_date:
                    continue

                if event_date[:10] > date.today().isoformat():
                    continue

                if event.get("status") != "final":
                    continue

                # Skip events already saved.
                if any(
                    row.get("event_id") == event_id
                    for row in existing.values()
                ):
                    continue

                checked_events += 1

                detail = request(
                    "get_event",
                    {"event_id": event_id},
                )

                bouts = detail.get("bouts", [])

                if not isinstance(bouts, list):
                    raise RuntimeError(
                        f"Unexpected bouts for {event_id}"
                    )

                for bout in bouts:
                    bout_id = str(
                        bout.get("bout_id") or ""
                    )

                    if not bout_id:
                        continue

                    if not bout.get("winner"):
                        continue

                    if bout_id in existing:
                        continue

                    existing[bout_id] = {
                        "bout_id": bout_id,
                        "event_id": event_id,
                        "event_date": event_date[:10],
                        "fighter_a": bout.get(
                            "fighter_a_name", ""
                        ),
                        "fighter_b": bout.get(
                            "fighter_b_name", ""
                        ),
                        "winner": bout.get(
                            "winner", ""
                        ),
                        "method": bout.get(
                            "method", ""
                        ),
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

    # Do not overwrite with an empty dataset.
    if not existing:
        raise RuntimeError(
            "FighterAPI returned no completed fights. "
            "Check the API response logs. "
            "No results file was overwritten."
        )

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
                "events_returned": returned_events,
                "events_checked": checked_events,
                "saved_to": str(RESULTS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
