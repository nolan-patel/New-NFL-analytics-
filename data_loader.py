from __future__ import annotations

from functools import lru_cache
from datetime import date, timedelta
import pandas as pd

PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.parquet"
)

# IMPORTANT:
# nflverse changed player-stat delivery in 2025. Rather than using the old
# all-years player_stats.parquet and trying to filter it, load the official
# regular-season summary file for each season directly.
PLAYER_REG_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_reg_{season}.parquet"
)

TEAMS_URL = (
    "https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/"
    "teams_colors_logos.csv"
)

START_SEASON = 2023


def nfl_season_for_date(today: date | None = None) -> int:
    """
    Mirror nflverse's season rollover concept:
    the new NFL season becomes current on the Wednesday after Labor Day.
    """
    today = today or date.today()
    year = today.year

    # Labor Day = first Monday in September.
    sept1 = date(year, 9, 1)
    days_to_monday = (7 - sept1.weekday()) % 7
    labor_day = sept1 + timedelta(days=days_to_monday)
    rollover = labor_day + timedelta(days=2)  # Wednesday after Labor Day

    return year if today >= rollover else year - 1


def available_seasons(start_season: int = START_SEASON) -> tuple[int, ...]:
    """Return every NFL season from the project start through the current NFL season."""
    current = nfl_season_for_date()
    return tuple(range(start_season, current + 1))


DEFAULT_SEASONS = available_seasons()


@lru_cache(maxsize=8)
def load_pbp(season: int) -> pd.DataFrame:
    df = pd.read_parquet(PBP_URL.format(season=int(season)))
    if "season_type" in df.columns:
        df = df[df["season_type"].eq("REG")]
    if "play_type" in df.columns:
        df = df[df["play_type"].isin(["pass", "run"])]
    return df.reset_index(drop=True)


@lru_cache(maxsize=16)
def load_player_reg_season(season: int) -> pd.DataFrame:
    """Load nflverse's official REGULAR-SEASON player summary for one year."""
    url = PLAYER_REG_URL.format(season=int(season))
    df = pd.read_parquet(url).copy()

    # Season summaries use recent_team; normalize to team for the app.
    if "team" not in df.columns and "recent_team" in df.columns:
        df = df.rename(columns={"recent_team": "team"})
    elif "team" in df.columns and "recent_team" in df.columns:
        df["team"] = df["team"].fillna(df["recent_team"])

    if "season" not in df.columns:
        df["season"] = int(season)

    return df.reset_index(drop=True)


def load_player_stats(seasons=DEFAULT_SEASONS) -> pd.DataFrame:
    frames = []
    errors = []

    for season in seasons:
        try:
            frames.append(load_player_reg_season(int(season)))
        except Exception as exc:
            errors.append((season, str(exc)))

    if not frames:
        raise RuntimeError(f"Could not load any player seasons. Errors: {errors}")

    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


@lru_cache(maxsize=1)
def load_teams() -> pd.DataFrame:
    try:
        return pd.read_csv(TEAMS_URL)
    except Exception:
        return pd.DataFrame(
            columns=[
                "team_abbr",
                "team_name",
                "team_logo_espn",
                "team_color",
                "team_color2",
            ]
        )


def latest_available_player_season() -> int:
    """
    Return the newest season whose nflverse regular-season player summary loads.
    Falls back to earlier seasons if the newest file has not appeared yet.
    """
    for season in reversed(available_seasons()):
        try:
            df = load_player_reg_season(season)
            if not df.empty:
                return season
        except Exception:
            continue
    return START_SEASON


def latest_available_pbp_season() -> int:
    """Return the newest season whose nflverse play-by-play file can be loaded."""
    for season in reversed(available_seasons()):
        try:
            df = load_pbp(season)
            if not df.empty:
                return season
        except Exception:
            continue
    return START_SEASON
