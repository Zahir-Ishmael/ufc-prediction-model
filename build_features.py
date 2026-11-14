
#!/usr/bin/env python3
"""Build leakage-safe UFC pre-fight features.

Combines the historical master archive with new FighterAPI results.
FighterAPI-only fights have missing detailed statistics and odds;
these values are never fabricated.

All fights on the same date use fighter history from earlier dates.
"""

from pathlib import Path
from collections import defaultdict
import json
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SRC = DATA / "ufc_fights_2015_present.csv"
if not SRC.exists():
    SRC = DATA / "ufc_all_fights_wide.csv"

API_SRC = DATA / "fighterapi_results.csv"
OUT = DATA / "prefight_features.csv"
STATUS = DATA / "feature_status.json"


def num(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def mean(values):
    valid = [v for v in values if pd.notna(v)]
    return float(np.mean(valid)) if valid else np.nan


def ratio(numerator, denominator):
    if (
        pd.notna(numerator)
        and pd.notna(denominator)
        and denominator > 0
    ):
        return numerator / denominator
    return np.nan


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_id(value):
    value = clean_text(value)
    if value.endswith(".0"):
        value = value[:-2]
    return value


def normalize_name(value):
    value = clean_text(value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )
    value = value.casefold()
    return re.sub(r"[^a-z0-9]", "", value)


def fighter_snapshot(row, side, history, ratings):
    recent = history[-5:]
    last3 = history[-3:]

    fighter_id = clean_id(row[f"{side}_id"])

    values = {
        "age": num(row.get(f"{side}_age")),
        "reach": num(row.get(f"{side}_reach_cm")),
        "experience": len(history),
        "elo": ratings.get(fighter_id, 1500.0),
        "career_winrate": mean(
            [e["win"] for e in history]
        ),
        "recent3_winrate": mean(
            [e["win"] for e in last3]
        ),
        "recent5_weighted_winrate": (
            ratio(
                sum(
                    (i + 1) * e["win"]
                    for i, e in enumerate(recent)
                ),
                sum(range(1, len(recent) + 1)),
            )
            if recent else np.nan
        ),
        "opponent_elo": mean(
            [e["opponent_elo"] for e in history]
        ),
        "strikes_per_min": mean(
            [e["strikes_per_min"] for e in history]
        ),
        "takedowns_per_min": mean(
            [e["takedowns_per_min"] for e in history]
        ),
        "submissions_per_min": mean(
            [e["submissions_per_min"] for e in history]
        ),
        "control_per_min": mean(
            [e["control_per_min"] for e in history]
        ),
        "striking_defense": mean(
            [e["striking_defense"] for e in history]
        ),
        "takedown_defense": mean(
            [e["takedown_defense"] for e in history]
        ),
        "recent_striking": mean(
            [e["strikes_per_min"] for e in last3]
        ),
        "recent_control": mean(
            [e["control_per_min"] for e in last3]
        ),
        "recent_finish_rate": mean(
            [e["finish_win"] for e in recent]
        ),
        "recent_opponent_elo": mean(
            [e["opponent_elo"] for e in recent]
        ),
        "recent_fight_count": len(recent),
        "layoff_days": (
            (row["event_date"] - history[-1]["date"]).days
            if history else np.nan
        ),
    }

    if history:
        streak = 0
        last = history[-1]["win"]

        for event in reversed(history):
            if event["win"] != last:
                break
            streak += 1

        values["streak"] = streak if last else -streak
    else:
        values["streak"] = 0

    for field in [
        "strikes_per_min",
        "takedowns_per_min",
        "control_per_min",
    ]:
        values["trend_" + field] = (
            mean([e[field] for e in history[-3:]])
            - mean([e[field] for e in history[-6:-3]])
            if len(history) >= 6
            else np.nan
        )

    if len(history) < 2:
        style = "unknown"
    else:
        striking = (
            values["strikes_per_min"] >= 3.0
            if pd.notna(values["strikes_per_min"])
            else False
        )

        wrestling = (
            (
                values["takedowns_per_min"] >= 0.35
                if pd.notna(values["takedowns_per_min"])
                else False
            )
            or (
                values["control_per_min"] >= 0.65
                if pd.notna(values["control_per_min"])
                else False
            )
        )

        submission = (
            values["submissions_per_min"] >= 0.15
            if pd.notna(values["submissions_per_min"])
            else False
        )

        if striking and (wrestling or submission):
            style = "mixed"
        elif submission:
            style = "submission"
        elif wrestling:
            style = "wrestler"
        elif striking:
            style = "striker"
        else:
            style = "low_activity"

    values["style"] = style
    return values


def entry(row, side, other, win, opponent_elo):
    duration = num(row.get("fight_duration_sec"))

    minutes = (
        duration / 60
        if pd.notna(duration) and duration > 0
        else np.nan
    )

    def stat(name):
        return num(row.get(f"{side}_tot_{name}"))

    def opponent_stat(name):
        return num(row.get(f"{other}_tot_{name}"))

    striking_allowed = ratio(
        opponent_stat("sig_str_landed"),
        opponent_stat("sig_str_att"),
    )

    takedown_allowed = ratio(
        opponent_stat("td_landed"),
        opponent_stat("td_att"),
    )

    method = clean_text(
        row.get("finish_type")
    ).lower()

    return {
        "date": row["event_date"],
        "win": int(win),
        "opponent_elo": opponent_elo,
        "strikes_per_min": ratio(
            stat("sig_str_landed"), minutes
        ),
        "takedowns_per_min": ratio(
            stat("td_landed"), minutes
        ),
        "submissions_per_min": ratio(
            stat("sub_att"), minutes
        ),
        "control_per_min": (
            ratio(stat("ctrl_sec"), duration) * 60
            if pd.notna(duration) and duration > 0
            else np.nan
        ),
        "striking_defense": (
            1 - striking_allowed
            if pd.notna(striking_allowed)
            else np.nan
        ),
        "takedown_defense": (
            1 - takedown_allowed
            if pd.notna(takedown_allowed)
            else np.nan
        ),
        "finish_win": int(
            bool(win)
            and (
                "ko" in method
                or "tko" in method
                or "sub" in method
            )
        ),
    }


def build_name_map(historical):
    """Only map names that uniquely identify one historical ID."""
    candidates = defaultdict(set)

    for side in ("f1", "f2"):
        for name, fighter_id in zip(
            historical[f"{side}_name"],
            historical[f"{side}_id"],
        ):
            key = normalize_name(name)
            fid = clean_id(fighter_id)

            if key and fid:
                candidates[key].add(fid)

    return {
        name: next(iter(ids))
        for name, ids in candidates.items()
        if len(ids) == 1
    }


def resolve_fighter(name, name_map):
    key = normalize_name(name)

    if not key:
        return ""

    # Reuse historical IDs when the name is unambiguous.
    if key in name_map:
        return name_map[key]

    # Separate namespace for unmatched API fighters.
    return "api:" + key


def resolve_winner(value, fighter_a, fighter_b, id_a, id_b):
    winner = clean_text(value).lower()
    winner_id = clean_id(value).lower()

    if not winner:
        return None

    if winner in ("draw", "no contest", "nc", "tie"):
        return None

    if winner in ("f1", "fighter 1", "1", "a"):
        return 1

    if winner in ("f2", "fighter 2", "2", "b"):
        return 0

    if winner_id == id_a.lower():
        return 1

    if winner_id == id_b.lower():
        return 0

    normalized = normalize_name(value)

    if normalized == normalize_name(fighter_a):
        return 1

    if normalized == normalize_name(fighter_b):
        return 0

    return None


def load_and_merge():
    if not SRC.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {SRC}"
        )

    historical = pd.read_csv(
        SRC, low_memory=False
    )

    required = [
        "fight_id",
        "event_date",
        "f1_id",
        "f2_id",
        "f1_name",
        "f2_name",
        "winner",
        "f1_age",
        "f2_age",
        "f1_reach_cm",
        "f2_reach_cm",
    ]

    missing = [
        col for col in required
        if col not in historical.columns
    ]

    if missing:
        raise ValueError(
            f"Historical dataset missing columns: {missing}"
        )

    historical["event_date"] = pd.to_datetime(
        historical["event_date"],
        errors="coerce",
    )

    historical = historical.dropna(
        subset=[
            "fight_id",
            "event_date",
            "f1_id",
            "f2_id",
        ]
    ).copy()

    historical["fight_id"] = (
        historical["fight_id"].map(clean_id)
    )

    for side in ("f1", "f2"):
        historical[f"{side}_id"] = (
            historical[f"{side}_id"].map(clean_id)
        )

    historical = historical.drop_duplicates(
        "fight_id"
    )

    historical["_source"] = "historical"

    report = {
        "historical_fights": len(historical),
        "api_fights_available": 0,
        "api_fights_added": 0,
        "api_duplicates_skipped": 0,
        "api_unresolved_winners": 0,
        "api_missing_fighter_names": 0,
        "api_ambiguous_identity_warning": (
            "FighterAPI names are matched to historical "
            "IDs only when the normalized name is unique."
        ),
    }

    if not API_SRC.exists():
        print(
            "No FighterAPI results file found. "
            "Using historical dataset only."
        )
        return historical, report

    api = pd.read_csv(
        API_SRC, low_memory=False
    )

    report["api_fights_available"] = len(api)

    api_required = [
        "bout_id",
        "event_date",
        "fighter_a",
        "fighter_b",
        "winner",
    ]

    api_missing = [
        col for col in api_required
        if col not in api.columns
    ]

    if api_missing:
        raise ValueError(
            f"FighterAPI CSV missing columns: {api_missing}"
        )

    name_map = build_name_map(historical)

    # Detect overlap by date and fighter pair, even when
    # the two data sources use different fight IDs.
    known_fights = set()

    for _, row in historical.iterrows():
        a = normalize_name(row["f1_name"])
        b = normalize_name(row["f2_name"])

        if a and b:
            known_fights.add(
                (
                    row["event_date"].strftime("%Y-%m-%d"),
                    tuple(sorted((a, b))),
                )
            )

    existing_ids = set(
        historical["fight_id"].astype(str)
    )

    new_rows = []

    for _, bout in api.iterrows():
        fight_id = "fighterapi:" + clean_id(
            bout.get("bout_id")
        )

        if fight_id == "fighterapi:":
            continue

        event_date = pd.to_datetime(
            bout.get("event_date"),
            errors="coerce",
        )

        if pd.isna(event_date):
            continue

        fighter_a = clean_text(
            bout.get("fighter_a")
        )
        fighter_b = clean_text(
            bout.get("fighter_b")
        )

        if not fighter_a or not fighter_b:
            report["api_missing_fighter_names"] += 1
            continue

        id_a = resolve_fighter(
            fighter_a, name_map
        )
        id_b = resolve_fighter(
            fighter_b, name_map
        )

        if not id_a or not id_b or id_a == id_b:
            report["api_missing_fighter_names"] += 1
            continue

        winner = resolve_winner(
            bout.get("winner"),
            fighter_a,
            fighter_b,
            id_a,
            id_b,
        )

        if winner is None:
            report["api_unresolved_winners"] += 1
            continue

        date_key = event_date.strftime(
            "%Y-%m-%d"
        )

        matchup = (
            date_key,
            tuple(
                sorted(
                    (
                        normalize_name(fighter_a),
                        normalize_name(fighter_b),
                    )
                )
            ),
        )

        if (
            fight_id in existing_ids
            or matchup in known_fights
        ):
            report["api_duplicates_skipped"] += 1
            continue

        # Keep all unavailable advanced statistics as NaN.
        # This prevents fabricated striking, grappling,
        # physical attributes, or betting odds.
        record = {
            col: np.nan
            for col in historical.columns
        }

        record.update({
            "fight_id": fight_id,
            "event_date": event_date,
            "f1_id": id_a,
            "f2_id": id_b,
            "f1_name": fighter_a,
            "f2_name": fighter_b,
            "winner": fighter_a if winner == 1 else fighter_b,
            "finish_type": clean_text(
                bout.get("method")
            ),
            "_source": "fighterapi",
        })

        new_rows.append(record)
        known_fights.add(matchup)
        existing_ids.add(fight_id)

    report["api_fights_added"] = len(new_rows)

    if new_rows:
        additions = pd.DataFrame(
            new_rows,
            columns=historical.columns,
        )

        combined = pd.concat(
            [historical, additions],
            ignore_index=True,
        )
    else:
        combined = historical

    return combined, report


def main():
    data, merge_report = load_and_merge()

    today = pd.Timestamp.now().normalize()

    data = data[
        (data["event_date"] >= pd.Timestamp("2015-01-01"))
        & (data["event_date"] <= today)
    ].copy()

    data = (
        data.drop_duplicates("fight_id")
        .sort_values(["event_date", "fight_id"])
    )

    history = defaultdict(list)
    ratings = {}
    rows = []

    counts = {
        "winner_unknown": 0,
        "no_odds": 0,
        "api_feature_rows": 0,
    }

    for event_date, group in data.groupby(
        "event_date", sort=True
    ):
        pending = []

        for _, fight in group.iterrows():
            id1 = clean_id(fight["f1_id"])
            id2 = clean_id(fight["f2_id"])

            if not id1 or not id2 or id1 == id2:
                continue

            y = resolve_winner(
                fight["winner"],
                fight["f1_name"],
                fight["f2_name"],
                id1,
                id2,
            )

            if y is None:
                counts["winner_unknown"] += 1
                continue

            s1 = fighter_snapshot(
                fight, "f1", history[id1], ratings
            )
            s2 = fighter_snapshot(
                fight, "f2", history[id2], ratings
            )

            row = {
                "fight_id": fight["fight_id"],
                "date": event_date.strftime("%Y-%m-%d"),
                "fighter_1": fight["f1_name"],
                "fighter_2": fight["f2_name"],
                "f1_id": id1,
                "f2_id": id2,
                "y": y,
                "event_country": fight.get("event_country"),
                "event_state": fight.get("event_state"),
                "event_city": fight.get("event_city"),
            }

            for key in s1:
                if key == "style":
                    row["f1_style"] = s1[key]
                    row["f2_style"] = s2[key]
                else:
                    row["diff_" + key] = (
                        s1[key] - s2[key]
                        if (
                            pd.notna(s1[key])
                            and pd.notna(s2[key])
                        )
                        else np.nan
                    )

            for style in [
                "striker",
                "wrestler",
                "submission",
                "mixed",
                "low_activity",
            ]:
                row["style_" + style] = (
                    int(s1["style"] == style)
                    - int(s2["style"] == style)
                )

            wrestling_values = [
                s1["takedowns_per_min"],
                s2["takedowns_per_min"],
                s1["takedown_defense"],
                s2["takedown_defense"],
            ]

            row["wrestling_matchup"] = (
                s1["takedowns_per_min"]
                * (1 - s2["takedown_defense"])
                - s2["takedowns_per_min"]
                * (1 - s1["takedown_defense"])
                if all(pd.notna(v) for v in wrestling_values)
                else np.nan
            )

            striking_values = [
                s1["strikes_per_min"],
                s2["strikes_per_min"],
                s1["striking_defense"],
                s2["striking_defense"],
            ]

            row["striking_matchup"] = (
                s1["strikes_per_min"]
                * (1 - s2["striking_defense"])
                - s2["strikes_per_min"]
                * (1 - s1["striking_defense"])
                if all(pd.notna(v) for v in striking_values)
                else np.nan
            )

            for kind in ("open", "close"):
                probability = num(
                    fight.get(
                        f"f1_{kind}_prob_novig"
                    )
                )

                row[f"{kind}_prob"] = (
                    probability
                    if pd.notna(probability)
                    and 0 < probability < 1
                    else np.nan
                )

            if (
                pd.isna(row["open_prob"])
                or pd.isna(row["close_prob"])
            ):
                counts["no_odds"] += 1

            if fight.get("_source") == "fighterapi":
                counts["api_feature_rows"] += 1

            rows.append(row)

            elo1 = ratings.get(id1, 1500.0)
            elo2 = ratings.get(id2, 1500.0)

            expected = 1 / (
                1 + 10 ** ((elo2 - elo1) / 400)
            )

            delta = 24 * (y - expected)

            pending.extend([
                (
                    id1,
                    entry(
                        fight, "f1", "f2", y, elo2
                    ),
                    elo1 + delta,
                ),
                (
                    id2,
                    entry(
                        fight, "f2", "f1", 1 - y, elo1
                    ),
                    elo2 - delta,
                ),
            ])

        # Update after the entire date is processed.
        # Prevents same-day outcome leakage.
        for fighter_id, event, new_elo in pending:
            history[fighter_id].append(event)
            ratings[fighter_id] = new_elo

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError(
            "No valid fight features were generated."
        )

    result.to_csv(OUT, index=False)

    report = {
        "source_file": SRC.name,
        "api_source_file": (
            API_SRC.name if API_SRC.exists() else None
        ),
        "fight_count": len(result),
        "first_date": str(result["date"].min()),
        "last_date": str(result["date"].max()),
        "unknown_winner_excluded": counts["winner_unknown"],
        "missing_either_odds": counts["no_odds"],
        "api_feature_rows": counts["api_feature_rows"],
        **merge_report,
        "features": list(result.columns),
        "warning": (
            "FighterAPI-only fights lack detailed performance "
            "statistics, age, reach and betting odds. "
            "Missing values are preserved. Fighter identity "
            "matching uses unambiguous normalized names and "
            "requires validation. Closing odds are usable "
            "only immediately before a fight."
        ),
    }

    STATUS.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "features"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
