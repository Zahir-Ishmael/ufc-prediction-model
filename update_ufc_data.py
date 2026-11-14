
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

RESULTS = DATA / "fighterapi_results.csv"
STATE = DATA / "fighterapi_completed_events.json"

# The existing master dataset covers fights through Oct 10, 2026.
# Begin API synchronization here, with a small overlap.
START_DATE = "2026-10-01"

# Limit credit consumption on each GitHub Actions run.
MAX_CREDITS_PER_RUN = 30

# Conservative request pacing, including low-balance accounts.
REQUEST_DELAY = 7
MAX_RETRIES = 3

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

credits_used = 0


def request(endpoint, payload, cost):
    global credits_used

    if credits_used + cost > MAX_CREDITS_PER_RUN:
        print("Credit budget reached for this run.")
        return None

    body = json.dumps(payload).encode("utf-8")

    for attempt in range(MAX_RETRIES):
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
            time.sleep(REQUEST_DELAY)

            with urllib.request.urlopen(
                req, timeout=45
            ) as response:
                result = json.load(response)

            if not isinstance(result, dict):
                raise RuntimeError(
                    "Unexpected FighterAPI response type."
                )

            if result.get("error"):
                raise RuntimeError(
                    str(result["error"])
                )

            data = result.get("data", result)

            if not isinstance(data, dict):
                raise RuntimeError(
                    "Unexpected FighterAPI data format."
                )

            credits_used += cost
            return data

        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = exc.headers.get(
                    "Retry-After", ""
                )

                try:
                    wait = max(10, min(120, int(retry_after)))
                except (ValueError, TypeError):
                    wait = 30 * (attempt + 1)

                print(
                    f"Rate limited. Waiting {wait} seconds.",
                    flush=True,
                )

                time.sleep(wait)

                if attempt == MAX_RETRIES - 1:
                    print(
                        "Rate limit persists. "
                        "Saving progress and stopping."
                    )
                    return None

                continue

            if exc.code == 402:
                print(
                    "Insufficient API credits. "
                    "Stopping safely."
                )
                return None

            details = exc.read().decode(
                "utf-8", errors="replace"
            )

            raise RuntimeError(
                f"HTTP {exc.code}: {details[:300]}"
            ) from exc

        except urllib.error.URLError as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Network failure: {exc}"
                ) from exc

            time.sleep(10 * (attempt + 1))

    return None


def load_results():
    rows = {}

    if RESULTS.exists():
        with RESULTS.open(
            newline="", encoding="utf-8"
        ) as f:
            for row in csv.DictReader(f):
                bout_id = row.get("bout_id")
                if bout_id:
                    rows[bout_id] = row

    return rows


def load_completed():
    if not STATE.exists():
        return set()

    with STATE.open(encoding="utf-8") as f:
        return set(json.load(f))


def save_progress(rows, completed):
    # Write temporary files first to reduce corruption risk.
    temp_csv = RESULTS.with_suffix(".tmp")
    temp_state = STATE.with_suffix(".tmp")

    with temp_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDS
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                rows.values(),
                key=lambda r: (
                    r.get("event_date", ""),
                    r.get("bout_id", ""),
                ),
            )
        )

    with temp_state.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            sorted(completed), f, indent=2
        )

    os.replace(temp_csv, RESULTS)
    os.replace(temp_state, STATE)


def get_events(year):
    all_events = []
    offset = 0
    seen_offsets = set()

    while offset not in seen_offsets:
        seen_offsets.add(offset)

        data = request(
            "list_events",
            {
                "season": year,
                "league": "mma",
                "limit": 50,
                "offset": offset,
            },
            cost=1,
        )

        if data is None:
            return None

        events = data.get("events", [])

        if not isinstance(events, list):
            raise RuntimeError(
                "Invalid events response."
            )

        all_events.extend(events)

        next_offset = data.get("next_offset")

        if next_offset is None or not events:
            break

        offset = int(next_offset)

    return all_events


def main():
    if not KEY:
        raise RuntimeError(
            "Missing FIGHTERAPI_KEY secret."
        )

    rows = load_results()
    completed = load_completed()

    today = date.today().isoformat()
    new_fights = 0
    processed_events = 0

    # Check from 2026 onward, not the entire 2015 history.
    for year in range(
        2026, date.today().year + 1
    ):
        print(f"Checking UFC events for {year}...")

        events = get_events(year)

        if events is None:
            break

        # Process oldest missing events first.
        events.sort(
            key=lambda e: e.get("event_date", "")
        )

        for event in events:
            event_id = str(
                event.get("event_id") or ""
            )
            event_date = str(
                event.get("event_date") or ""
            )[:10]

            status = str(
                event.get("status") or ""
            ).lower()

            if not event_id or not event_date:
                continue

            if event_date < START_DATE:
                continue

            if event_date > today:
                continue

            if status != "final":
                continue

            if event_id in completed:
                continue

            detail = request(
                "get_event",
                {"event_id": event_id},
                cost=2,
            )

            if detail is None:
                save_progress(rows, completed)
                print(
                    "Stopped safely. Progress saved."
                )
                return

            bouts = detail.get("bouts", [])

            if not isinstance(bouts, list):
                raise RuntimeError(
                    f"Invalid bouts for event {event_id}"
                )

            event_added = 0

            for bout in bouts:
                bout_id = str(
                    bout.get("bout_id") or ""
                )

                winner = bout.get("winner")

                if not bout_id or not winner:
                    continue

                if bout_id in rows:
                    continue

                rows[bout_id] = {
                    "bout_id": bout_id,
                    "event_id": event_id,
                    "event_date": event_date,
                    "fighter_a": (
                        bout.get("fighter_a_name")
                        or bout.get("fighter_a")
                        or ""
                    ),
                    "fighter_b": (
                        bout.get("fighter_b_name")
                        or bout.get("fighter_b")
                        or ""
                    ),
                    "winner": str(winner),
                    "method": (
                        bout.get("method") or ""
                    ),
                    "end_round": (
                        bout.get("end_round") or ""
                    ),
                    "weight_class": (
                        bout.get("weight_class") or ""
                    ),
                }

                event_added += 1
                new_fights += 1

            # Don't permanently skip an event with no results.
            if bouts and any(
                bout.get("winner")
                for bout in bouts
            ):
                completed.add(event_id)

            processed_events += 1

            # Save after EVERY event.
            save_progress(rows, completed)

            print(
                f"Event {event_id} ({event_date}): "
                f"{event_added} new fights",
                flush=True,
            )

    save_progress(rows, completed)

    print(
        json.dumps(
            {
                "new_fights": new_fights,
                "total_api_fights": len(rows),
                "events_processed": processed_events,
                "completed_events": len(completed),
                "estimated_credits_used": credits_used,
                "saved_to": str(RESULTS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
