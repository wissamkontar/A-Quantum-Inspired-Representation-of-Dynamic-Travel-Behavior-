#!/usr/bin/env python3
"""
Create the final TimeUse+ preprocessing file used by the
quantum-inspired traveler behavioral profiling model.

Default inputs
--------------
activities.csv
    TimeUse+ activity/travel records.

weather.csv
    Weather-matched records.

Default output
--------------
activities_merged4.csv

Run
---
python preprocess_timeuse.py

"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


LOCATION_TOLERANCE = 300


ACTIVITY_REQUIRED_COLUMNS = {
    "participant_id",
    "event_type",
    "activity_name",
    "event_name_imputed",
    "event_mid_lat",
    "event_mid_lon",
    "trip_main_mode",
    "trip_distance",
    "started_at",
    "finished_at",
    "started_at_tz",
    "activity_duration",
    "event_duration",
}

WEATHER_REQUIRED_COLUMNS = {
    "started_at",
    "event_mid_lat",
    "event_mid_lon",
    "temp_celsius",
    "wind_speed_ms",
    "precip_depth_mm",
    "mid_lat_WGS84",
    "mid_lon_WGS84",
}


def validate_columns(
    df: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    """Check that an input file contains all required columns."""
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            + ", ".join(missing)
        )


def to_local_time(started_at: object, timezone_name: object) -> object:
    """Convert a UTC timestamp to the row-specific local timezone."""
    if pd.isna(started_at) or pd.isna(timezone_name):
        return None

    try:
        dt = pd.to_datetime(started_at, utc=True)
        dt_local = dt.tz_convert(ZoneInfo(str(timezone_name)))
        return dt_local.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Great-circle distance between two WGS84 points in kilometers."""
    earth_radius_km = 6371.0088

    lat1_r = radians(lat1)
    lon1_r = radians(lon1)
    lat2_r = radians(lat2)
    lon2_r = radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    )
    return 2 * earth_radius_km * asin(sqrt(a))


def create_activity_labels(act: pd.DataFrame) -> pd.DataFrame:
    """Create and fill the activity_new variable."""
    act = act.copy()

    # Keep only modeled event types.
    act = act[act["event_type"].isin(["stay", "track"])].copy()
    act = act.sort_values(
        ["participant_id", "started_at"]
    ).reset_index(drop=True)

    # Start from the TimeUse+ activity label.
    act["activity_new"] = act["activity_name"]

    # Track records represent travel between activities.
    act.loc[
        act["event_type"] == "track", "activity_new"
    ] = "in_transit"

    # Consolidate identified home/work locations.
    act.loc[
        act["event_name_imputed"] == "home", "activity_new"
    ] = "Home"
    act.loc[
        act["event_name_imputed"] == "work", "activity_new"
    ] = "Working"

    # Fill a missing label if the immediately preceding and following
    # records belong to the same participant and have the same activity.
    prev_pid = act["participant_id"].shift(1)
    next_pid = act["participant_id"].shift(-1)
    prev_activity = act["activity_new"].shift(1)
    next_activity = act["activity_new"].shift(-1)

    bridge_mask = (
        act["activity_new"].isna()
        & (prev_pid == act["participant_id"])
        & (next_pid == act["participant_id"])
        & prev_activity.notna()
        & (prev_activity == next_activity)
    )
    act.loc[
        bridge_mask, "activity_new"
    ] = prev_activity.loc[bridge_mask]

    # Build participant-specific sets of known Home and Working locations.
    home_coords = defaultdict(set)
    work_coords = defaultdict(set)

    for pid, group in act.groupby("participant_id", sort=False):
        group_with_location = group[group["event_mid_lat"].notna()]

        home_rows = group_with_location.loc[
            group_with_location["activity_new"] == "Home",
            ["event_mid_lat", "event_mid_lon"],
        ]
        for row in home_rows.itertuples(index=False):
            home_coords[pid].add(
                (row.event_mid_lat, row.event_mid_lon)
            )

        work_rows = group_with_location.loc[
            group_with_location["activity_new"] == "Working",
            ["event_mid_lat", "event_mid_lon"],
        ]
        for row in work_rows.itertuples(index=False):
            work_coords[pid].add(
                (row.event_mid_lat, row.event_mid_lon)
            )

    # Use known Home/Working locations to fill remaining missing activities.
    missing_rows = act.index[
        act["activity_new"].isna()
        & act["event_mid_lat"].notna()
    ]

    for idx in missing_rows:
        pid = act.at[idx, "participant_id"]
        lat = act.at[idx, "event_mid_lat"]
        lon = act.at[idx, "event_mid_lon"]

        for home_lat, home_lon in home_coords.get(pid, ()):
            if (
                abs(lat - home_lat) <= LOCATION_TOLERANCE
                and abs(lon - home_lon) <= LOCATION_TOLERANCE
            ):
                act.at[idx, "activity_new"] = "Home"
                break

        if pd.isna(act.at[idx, "activity_new"]):
            for work_lat, work_lon in work_coords.get(pid, ()):
                if (
                    abs(lat - work_lat) <= LOCATION_TOLERANCE
                    and abs(lon - work_lon) <= LOCATION_TOLERANCE
                ):
                    act.at[idx, "activity_new"] = "Working"
                    break

    return act


def assign_trip_attributes(act: pd.DataFrame) -> pd.DataFrame:
    """
    Assign the preceding trip's mode and distance to each activity stay.
    """
    act = act.copy()
    track_mask = act["event_type"] == "track"

    # Mode used to reach the destination.
    act["mode_choice_new"] = None
    act.loc[
        track_mask, "mode_choice_new"
    ] = act.loc[track_mask, "trip_main_mode"]

    act["mode_choice_new"] = (
        act.groupby("participant_id")["mode_choice_new"]
        .transform(lambda x: x.ffill())
    )

    # Track rows are only carriers of trip information.
    act.loc[track_mask, "mode_choice_new"] = None

    # Initial stay records before the first trip receive the next
    # available mode, matching the original notebook.
    act["mode_choice_new"] = (
        act.groupby("participant_id")["mode_choice_new"]
        .transform(lambda x: x.bfill())
    )

    # Trip distance used to reach the destination.
    act["trip_distance_new"] = None
    act.loc[
        track_mask, "trip_distance_new"
    ] = act.loc[track_mask, "trip_distance"]

    act["trip_distance_new"] = (
        act.groupby("participant_id")["trip_distance_new"]
        .transform(lambda x: x.ffill())
    )

    act.loc[track_mask, "trip_distance_new"] = None

    # Remove stay rows for which no preceding trip distance exists.
    missing_distance = (
        (act["event_type"] == "stay")
        & act["trip_distance_new"].isna()
    )
    act = act.loc[~missing_distance].copy()

    return act


def add_local_time_context(act: pd.DataFrame) -> pd.DataFrame:
    """
    Create local time and assign trip-departure hour/day to each stay.

    If several track rows precede a stay, the first track in that
    sequence supplies the trip departure time.
    """
    act = act.copy()

    act["started_at_local"] = [
        to_local_time(started_at, timezone_name)
        for started_at, timezone_name in zip(
            act["started_at"],
            act["started_at_tz"],
        )
    ]

    act["time_of_day_new"] = None
    act["day_of_week_new"] = None

    current_pid = None
    first_track_time = None
    in_track_sequence = False

    for idx in act.index:
        pid = act.at[idx, "participant_id"]
        event_type = act.at[idx, "event_type"]

        if pid != current_pid:
            current_pid = pid
            first_track_time = None
            in_track_sequence = False

        if event_type == "track":
            if not in_track_sequence:
                first_track_time = act.at[idx, "started_at_local"]
                in_track_sequence = True

        elif event_type == "stay":
            if first_track_time is not None:
                first_track_dt = pd.to_datetime(first_track_time)
                act.at[idx, "time_of_day_new"] = first_track_dt.hour
                act.at[idx, "day_of_week_new"] = (
                    first_track_dt.isoweekday()
                )

            in_track_sequence = False
            first_track_time = None

    return act


def attach_weather(
    act: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach weather by exact UTC timestamp and source coordinates,
    then forward-fill unmatched records within participant.
    """
    act = act.copy()

    act["started_at_utc"] = (
        pd.to_datetime(act["started_at"], utc=True)
        .dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    weather_lookup = {}
    for row in weather.itertuples(index=False):
        key = (
            row.started_at,
            row.event_mid_lat,
            row.event_mid_lon,
        )
        weather_lookup[key] = (
            row.temp_celsius,
            row.wind_speed_ms,
            row.precip_depth_mm,
        )

    act["temp_celsius_new"] = None
    act["wind_speed_ms_new"] = None
    act["precip_depth_mm_new"] = None

    for idx in act.index:
        key = (
            act.at[idx, "started_at_utc"],
            act.at[idx, "event_mid_lat"],
            act.at[idx, "event_mid_lon"],
        )

        if key in weather_lookup:
            temp, wind, precip = weather_lookup[key]
            act.at[idx, "temp_celsius_new"] = temp
            act.at[idx, "wind_speed_ms_new"] = wind
            act.at[idx, "precip_depth_mm_new"] = precip

    # Use the most recent available weather observation for unmatched
    # rows within each participant.
    for col in [
        "temp_celsius_new",
        "wind_speed_ms_new",
        "precip_depth_mm_new",
    ]:
        act[col] = (
            act.groupby("participant_id")[col]
            .transform(lambda x: x.ffill())
        )

    return act


def create_stay_episodes(act: pd.DataFrame) -> pd.DataFrame:
    """Remove travel rows and construct final stay durations."""
    stays = act[act["activity_new"] != "in_transit"].copy()
    stays = stays.reset_index(drop=True)

    stays["finished_at_utc"] = (
        pd.to_datetime(stays["finished_at"], utc=True)
        .dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    started = pd.to_datetime(stays["started_at_utc"], utc=True)
    finished = pd.to_datetime(stays["finished_at_utc"], utc=True)

    stays["event_duration_new"] = (
        (finished - started).dt.total_seconds() / 60
    ).round().astype("Int64")

    # Remove consecutive duplicate start timestamps within participant.
    stays = stays.sort_values(
        ["participant_id", "started_at_utc"]
    ).reset_index(drop=True)

    duplicate_start = (
        (stays["participant_id"] == stays["participant_id"].shift())
        & (stays["started_at_utc"] == stays["started_at_utc"].shift())
    )

    stays = stays.loc[~duplicate_start].reset_index(drop=True)

    # If trip departure context was unavailable, use the activity
    # episode's own local start time.
    missing_tod = stays["time_of_day_new"].isna()
    missing_dow = stays["day_of_week_new"].isna()

    stays.loc[
        missing_tod, "time_of_day_new"
    ] = pd.to_datetime(
        stays.loc[missing_tod, "started_at_local"]
    ).dt.hour

    stays.loc[
        missing_dow, "day_of_week_new"
    ] = (
        pd.to_datetime(
            stays.loc[missing_dow, "started_at_local"]
        ).dt.dayofweek
        + 1
    )

    return stays


def add_wgs84_and_home_distance(
    episodes: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add WGS84 coordinates and calculate distance from the participant's
    most frequently observed Home coordinate.
    """
    episodes = episodes.copy()

    # Last occurrence wins when duplicate source-coordinate keys exist,
    # matching the dictionary behavior in the notebook.
    wgs_lookup = {}
    for row in weather.itertuples(index=False):
        key = (row.event_mid_lat, row.event_mid_lon)
        wgs_lookup[key] = (
            row.mid_lat_WGS84,
            row.mid_lon_WGS84,
        )

    coordinates = episodes.apply(
        lambda row: wgs_lookup.get(
            (row["event_mid_lat"], row["event_mid_lon"]),
            (None, None),
        ),
        axis=1,
    )

    episodes["mid_lat_WGS84"] = [
        coord[0] for coord in coordinates
    ]
    episodes["mid_lon_WGS84"] = [
        coord[1] for coord in coordinates
    ]

    # Most frequently observed Home coordinate for each participant.
    home_coords = {}

    for pid, group in episodes.groupby("participant_id"):
        home_rows = group.loc[
            group["activity_new"] == "Home",
            ["mid_lat_WGS84", "mid_lon_WGS84"],
        ].dropna()

        if len(home_rows):
            counter = Counter(
                zip(
                    home_rows["mid_lat_WGS84"],
                    home_rows["mid_lon_WGS84"],
                )
            )
            home_coords[pid] = counter.most_common(1)[0][0]

    def distance_from_home(row: pd.Series) -> object:
        pid = row["participant_id"]
        lat = row["mid_lat_WGS84"]
        lon = row["mid_lon_WGS84"]

        if (
            pid not in home_coords
            or pd.isna(lat)
            or pd.isna(lon)
        ):
            return None

        home_lat, home_lon = home_coords[pid]

        return round(
            haversine_km(
                home_lat,
                home_lon,
                lat,
                lon,
            ),
            3,
        )

    episodes["distance_from_home"] = episodes.apply(
        distance_from_home,
        axis=1,
    )

    return episodes


def merge_consecutive_episodes(
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge consecutive rows for the same participant when activity and
    location are the same within the original 300-unit tolerance.

    The first row supplies retained attributes. End time is taken from
    the final row, and duration fields are summed.
    """
    episodes = episodes.sort_values(
        ["participant_id", "started_at_utc"]
    ).reset_index(drop=True)

    merged_rows = []
    current_pid = None
    group_buffer = []

    def flush_buffer(buffer):
        if not buffer:
            return None

        first = buffer[0].copy()

        first["finished_at_utc"] = buffer[-1]["finished_at_utc"]
        first["finished_at"] = buffer[-1]["finished_at"]

        first["event_duration_new"] = sum(
            row["event_duration_new"]
            for row in buffer
            if pd.notna(row["event_duration_new"])
        )

        first["activity_duration"] = sum(
            row["activity_duration"]
            for row in buffer
            if pd.notna(row["activity_duration"])
        )

        first["event_duration"] = sum(
            row["event_duration"]
            for row in buffer
            if pd.notna(row["event_duration"])
        )

        return first

    for _, row in episodes.iterrows():
        pid = row["participant_id"]
        activity = row["activity_new"]
        lat = row["event_mid_lat"]
        lon = row["event_mid_lon"]

        if pid != current_pid:
            if group_buffer:
                merged_rows.append(flush_buffer(group_buffer))

            group_buffer = [row]
            current_pid = pid
            continue

        last = group_buffer[-1]
        same_activity = activity == last["activity_new"]

        if pd.notna(lat) and pd.notna(last["event_mid_lat"]):
            same_location = (
                abs(lat - last["event_mid_lat"])
                <= LOCATION_TOLERANCE
                and abs(lon - last["event_mid_lon"])
                <= LOCATION_TOLERANCE
            )
        else:
            same_location = False

        if same_activity and same_location:
            group_buffer.append(row)
        else:
            merged_rows.append(flush_buffer(group_buffer))
            group_buffer = [row]

    if group_buffer:
        merged_rows.append(flush_buffer(group_buffer))

    merged = pd.DataFrame(merged_rows).reset_index(drop=True)

    # Remaining unlabeled activities become Other.
    merged["activity_new"] = merged["activity_new"].fillna("Other")

    return merged


def add_global_time(episodes: pd.DataFrame) -> pd.DataFrame:
    """
    Express episode start/end times as minutes from the earliest
    observation in the complete dataset.
    """
    episodes = episodes.copy()

    started = pd.to_datetime(
        episodes["started_at_utc"],
        utc=True,
    )
    finished = pd.to_datetime(
        episodes["finished_at_utc"],
        utc=True,
    )

    t0 = started.min()

    episodes["started_at_utc_global"] = (
        (started - t0).dt.total_seconds() // 60
    ).astype("Int64")

    episodes["finished_at_utc_global"] = (
        (finished - t0).dt.total_seconds() // 60
    ).astype("Int64")

    return episodes


def preprocess(
    activities_path: Path,
    weather_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Run the complete preprocessing pipeline."""
    print(f"Reading activity data: {activities_path}")
    act = pd.read_csv(activities_path)
    validate_columns(
        act,
        ACTIVITY_REQUIRED_COLUMNS,
        str(activities_path),
    )

    print(f"Reading weather data:  {weather_path}")
    weather = pd.read_csv(weather_path)
    validate_columns(
        weather,
        WEATHER_REQUIRED_COLUMNS,
        str(weather_path),
    )

    print(f"Raw activity rows:      {len(act):,}")
    print(
        f"Raw participants:       "
        f"{act['participant_id'].nunique():,}"
    )

    print("1/8  Creating activity labels...")
    act = create_activity_labels(act)

    print("2/8  Assigning trip mode and distance...")
    act = assign_trip_attributes(act)

    print("3/8  Creating local time and trip context...")
    act = add_local_time_context(act)

    print("4/8  Matching weather variables...")
    act = attach_weather(act, weather)

    print("5/8  Constructing stay episodes...")
    episodes = create_stay_episodes(act)

    print("6/8  Adding WGS84 coordinates and home distance...")
    episodes = add_wgs84_and_home_distance(
        episodes,
        weather,
    )

    print("7/8  Merging consecutive same-activity/location rows...")
    episodes = merge_consecutive_episodes(episodes)

    print("8/8  Creating global episode times...")
    episodes = add_global_time(episodes)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    episodes.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Saved:                  {output_path}")
    print(f"Final rows:             {len(episodes):,}")
    print(
        f"Final participants:     "
        f"{episodes['participant_id'].nunique():,}"
    )
    print(
        f"Missing activity_new:   "
        f"{episodes['activity_new'].isna().sum():,}"
    )

    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create activities_merged4.csv from TimeUse+ activity "
            "records and the prepared weather file."
        )
    )

    parser.add_argument(
        "--activities",
        type=Path,
        default=Path("activities.csv"),
        help="Path to TimeUse+ activities.csv",
    )

    parser.add_argument(
        "--weather",
        type=Path,
        default=Path("weather.csv"),
        help="Path to the prepared weather.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("activities_merged4.csv"),
        help="Output CSV path",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(
        activities_path=args.activities,
        weather_path=args.weather,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
