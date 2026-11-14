
import csv
import json
import os
import time
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

    for attempt in range(3):
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

            if not isinstance(result, dict):
                raise RuntimeError(
                    f"Invalid response from {endpoint}: {result}"
                )

            if result.get("error"):
                raise RuntimeError(
                    f"FighterAPI error: {result['error']}"
                )

            return result.get("data", result)

        except urllib.error.HTTPError as exc:
            details = exc.read().decode(
                "utf-8", errors="replace"
            )

            if exc.code in (429, 500, 502, 503, 504):
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue

            raise RuntimeError(
                f"FighterAPI HTTP {exc.code}: {details[:500]}"
            ) from exc

        except urllib.error.URLError:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise

    raise RuntimeError(
        f"Request failed after retries: {endpoint}"
    )


def load_existing():
    existing = {}

    if RESULTS.exists():
        with RESULTS.open(
            newline="", encoding="utf-8"
        ) as f:
            for row in csv.DictReader(f):
                bout_id = row.get("bout_id")
                if bout_id:
                    existing[bout_id] = row

    return existing


def get_events(year):
    events = []
    offset = 0
    seen_offsets = set()

    while True:
        if offset in seen_offsets:
            raise RuntimeError(
                f"Repeated pagination offset for {year}"
            )

        seen_offsets.add(offset)

        response = request(
            "list_events",
            {
                "season": year,
                "league": "mma",
                "limit": 50,
                "offset": offset,
            },
        )

        if not isinstance(response, dict):
            raise RuntimeError(
                f"Unexpected list_events response: {response}"
            )

        page = response.get("events", [])

        if not isinstance(page, list):
            raise RuntimeError(
                f"Invalid events list for {year}: {response}"
            )

        print(
            f"Year {year}, offset {offset}: "
            f"{len(page)} events returned",
            flush=True,
        )

        events.extend(page)

        next_offset = response.get("next_offset")

        if next_offset is None or not page:
            break

        next_offset = int(next_offset)

        if next_offset == offset:
            raise RuntimeError(
                f"Pagination did not advance for {year}"
            )

        offset = next_offset

    return events


def get_event_bouts(event_id):
    response = request(
        "get_event",
        {"event_id": event_id},
    )

    if not isinstance(response, dict):
        raise RuntimeError(
            f"Unexpected get_event response: {response}"
        )

    bouts = response.get("bouts", [])

    if not isinstance(bouts, list):
        raise RuntimeError(
            f"Invalid bouts for event {event_id}"
        )

    return bouts


def main():
    existing = load_existing()
    today = date.today().isoformat()

    new_count = 0
    events_checked = 0
    events_with_results = set(
        row.get("event_id")
        for row in existing.values()
        if row.get("event_id")
    )

    for year in range(2015, date.today().year + 1):
        print(f"\nChecking UFC events for {year}...")

        events = get_events(year)

        for event in events:
            event_id = str(event.get("event_id", ""))
            event_date = event.get("event_date", "")
            status = str(
                event.get("status", "")
            ).lower()

            if not event_id or not event_date:
                continue

            if event_date < "2015-01-01":
                continue

            if event_date > today:
                continue

            if status != "final":
                continue

            if event_id in events_with_results:
                continue

            events_checked += 1

            bouts = get_event_bouts(event_id)

            added_for_event = 0

            for bout in bouts:
                bout_id = str(bout.get("bout_id", ""))

                if not bout_id:
                    continue

                if bout_id in existing:
                    continue

                winner = bout.get("winner")

                if not winner:
                    continue

                fighter_a = (
                    bout.get("fighter_a_name")
                    or bout.get("fighter_a")
                    or ""
                )

                fighter_b = (
                    bout.get("fighter_b_name")
                    or bout.get("fighter_b")
                    or ""
                )

                existing[bout_id] = {
                    "bout_id": bout_id,
                    "event_id": event_id,
                    "event_date": event_date,
                    "fighter_a": str(fighter_a),
                    "fighter_b": str(fighter_b),
                    "winner": str(winner),
                    "method": bout.get("method") or "",
                    "end_round": bout.get("end_round") or "",
                    "weight_class": (
                        bout.get("weight_class") or ""
                    ),
                }

                new_count += 1
                added_for_event += 1

            if added_for_event:
                events_with_results.add(event_id)

            print(
                f"Event {event_id}: "
                f"{added_for_event} new fights",
                flush=True,
            )

    sorted_rows = sorted(
        existing.values(),
        key=lambda row: (
            row.get("event_date", ""),
            row.get("bout_id", ""),
        ),
    )

    with RESULTS.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDS
        )
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(
        json.dumps(
            {
                "new_fights": new_count,
                "total_api_fights": len(existing),
                "events_checked": events_checked,
                "saved_to": str(RESULTS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
