from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------- Team analytics ----------------

def build_team_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.copy()
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])]

    off = (
        pbp.dropna(subset=["posteam"])
        .groupby(["season", "posteam"], as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "team", "epa": "off_epa"})
    )

    deff = (
        pbp.dropna(subset=["defteam"])
        .groupby(["season", "defteam"], as_index=False)["epa"]
        .mean()
        .rename(columns={"defteam": "team", "epa": "def_epa_allowed"})
    )
    deff["def_epa"] = -deff["def_epa_allowed"]

    onescore = pbp[pbp["score_differential"].abs() <= 8].copy()
    off_os = (
        onescore.dropna(subset=["posteam"])
        .groupby(["season", "posteam"], as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "team", "epa": "off_epa_onescore"})
    )
    def_os = (
        onescore.dropna(subset=["defteam"])
        .groupby(["season", "defteam"], as_index=False)["epa"]
        .mean()
        .rename(columns={"defteam": "team", "epa": "def_epa_allowed_onescore"})
    )
    def_os["def_epa_onescore"] = -def_os["def_epa_allowed_onescore"]

    out = off.merge(deff[["season", "team", "def_epa"]], on=["season", "team"])
    out = out.merge(off_os[["season", "team", "off_epa_onescore"]], on=["season", "team"], how="left")
    out = out.merge(def_os[["season", "team", "def_epa_onescore"]], on=["season", "team"], how="left")
    return out


def _turnover_flag(group: pd.DataFrame) -> pd.Series:
    ints = (
        group["interception"].fillna(0).gt(0)
        if "interception" in group.columns
        else pd.Series(False, index=group.index)
    )
    fumbles = (
        group["fumble_lost"].fillna(0).gt(0)
        if "fumble_lost" in group.columns
        else pd.Series(False, index=group.index)
    )
    return ints | fumbles


def build_off_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, team), g in pbp.dropna(subset=["posteam"]).groupby(["season", "posteam"]):
        is_pass = g["play_type"].eq("pass")
        is_run = g["play_type"].eq("run")
        yards = g["yards_gained"].fillna(0)
        third = g["down"].eq(3)
        third_rate = (
            g.loc[third, "first_down"].fillna(0).astype(float).mean()
            if "first_down" in g.columns and third.any()
            else np.nan
        )
        one_score = g["score_differential"].abs() <= 8

        rows.append({
            "season": season,
            "team": team,
            "plays": len(g),
            "off_epa": g["epa"].mean(),
            "off_epa_onescore": g.loc[one_score, "epa"].mean() if one_score.any() else np.nan,
            "off_success_rate": g["epa"].gt(0).mean(),
            "explosive_pass_rate": (is_pass & yards.ge(15)).mean(),
            "explosive_run_rate": (is_run & yards.ge(10)).mean(),
            "turnover_rate": _turnover_flag(g).mean(),
            "third_down_conv_rate": third_rate,
        })
    return pd.DataFrame(rows)


def build_def_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, team), g in pbp.dropna(subset=["defteam"]).groupby(["season", "defteam"]):
        is_pass = g["play_type"].eq("pass")
        is_run = g["play_type"].eq("run")
        yards = g["yards_gained"].fillna(0)
        drop = g["dropback"].fillna(0).eq(1) if "dropback" in g.columns else is_pass
        sack = g["sack"].fillna(0).gt(0) if "sack" in g.columns else pd.Series(False, index=g.index)
        qb_hit = g["qb_hit"].fillna(0).eq(1) if "qb_hit" in g.columns else pd.Series(False, index=g.index)
        pressure = sack | qb_hit

        third = g["down"].eq(3)
        third_rate = (
            g.loc[third, "first_down"].fillna(0).astype(float).mean()
            if "first_down" in g.columns and third.any()
            else np.nan
        )

        rows.append({
            "season": season,
            "team": team,
            "plays": len(g),
            "pressure_rate": pressure.loc[drop].mean() if drop.any() else np.nan,
            "def_success_rate": g["epa"].lt(0).mean(),
            "explosive_pass_rate": (is_pass & yards.ge(15)).mean(),
            "explosive_run_rate": (is_run & yards.ge(10)).mean(),
            "turnover_rate": _turnover_flag(g).mean(),
            "third_down_conv_rate": third_rate,
        })
    return pd.DataFrame(rows)


# ---------------- Player season aggregation ----------------

ID_COLS = [
    "player_id", "player_name", "player_display_name",
    "position", "position_group", "team", "season"
]

RATE_COLS = [
    "passing_cpoe", "pacr", "racr", "target_share", "air_yards_share", "wopr"
]


def _weighted_avg(g: pd.DataFrame, col: str, weight_col: str | None = None):
    if col not in g.columns:
        return np.nan
    vals = pd.to_numeric(g[col], errors="coerce")
    if weight_col and weight_col in g.columns:
        w = pd.to_numeric(g[weight_col], errors="coerce").fillna(0)
        mask = vals.notna() & w.gt(0)
        if mask.any():
            return np.average(vals[mask], weights=w[mask])
    return vals.mean()


def aggregate_player_seasons(stats: pd.DataFrame) -> pd.DataFrame:
    """
    nflverse regular-season player files are already one row per player-season.
    Normalize their schema and derive the ranking metrics directly.
    """
    df = stats.copy()

    if "team" not in df.columns and "recent_team" in df.columns:
        df = df.rename(columns={"recent_team": "team"})

    if "player_display_name" in df.columns:
        df["player_name"] = df["player_display_name"]
    elif "player_name" not in df.columns and "player_id" in df.columns:
        df["player_name"] = df["player_id"].astype(str)

    if "player_id" not in df.columns:
        df["player_id"] = df["player_name"].astype(str)

    if "position" not in df.columns:
        # Keep the app from crashing if a future source schema changes.
        df["position"] = ""

    if "position_group" not in df.columns:
        df["position_group"] = df["position"]

    if "games" not in df.columns:
        df["games"] = 1

    return derive_player_metrics(df)

def _safe_div(df, num, den, out):
    if num in df.columns and den in df.columns:
        df[out] = pd.to_numeric(df[num], errors="coerce") / pd.to_numeric(df[den], errors="coerce").replace(0, np.nan)


def derive_player_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # QB
    _safe_div(out, "passing_yards", "attempts", "yards_per_attempt")
    _safe_div(out, "passing_tds", "attempts", "pass_td_rate")
    _safe_div(out, "passing_interceptions", "attempts", "int_rate")
    _safe_div(out, "passing_epa", "attempts", "epa_per_attempt")
    if "attempts" in out.columns:
        sacks = out["sacks_suffered"].fillna(0) if "sacks_suffered" in out.columns else 0
        out["dropbacks"] = out["attempts"].fillna(0) + sacks
        if "sacks_suffered" in out.columns:
            _safe_div(out, "sacks_suffered", "dropbacks", "sack_rate")

    # RB
    _safe_div(out, "rushing_yards", "carries", "yards_per_carry")
    _safe_div(out, "rushing_epa", "carries", "epa_per_rush")
    _safe_div(out, "rushing_first_downs", "carries", "rush_first_down_rate")
    _safe_div(out, "rushing_tds", "carries", "rush_td_rate")

    # WR / TE
    _safe_div(out, "receiving_yards", "targets", "yards_per_target")
    _safe_div(out, "receiving_epa", "targets", "epa_per_target")
    _safe_div(out, "receptions", "targets", "catch_rate")
    _safe_div(out, "receiving_first_downs", "targets", "rec_first_down_rate")
    _safe_div(out, "receiving_tds", "targets", "rec_td_rate")

    # Defensive per-game production
    if "games" in out.columns:
        for col in [
            "def_tackles_solo", "def_tackle_assists", "def_tackles_for_loss",
            "def_sacks", "def_qb_hits", "def_interceptions",
            "def_pass_defended", "def_fumbles_forced"
        ]:
            if col in out.columns:
                _safe_div(out, col, "games", f"{col}_pg")

    return out


def normalize_position(pos: str) -> str:
    p = str(pos).upper()
    if p in {"FS", "SS", "SAF"}:
        return "S"
    if p in {"ILB", "MLB", "OLB"}:
        return "LB"
    if p in {"EDGE", "ED"}:
        return "EDGE"
    if p in {"NT", "DI"}:
        return "DT"
    if p == "DE":
        return "EDGE"
    return p


# ---------------- Data-driven rankings ----------------

RANK_CONFIG = {
    "QB": {
        "min": ("attempts", 200),
        "metrics": {
            "epa_per_attempt": 0.30,
            "passing_cpoe": 0.20,
            "yards_per_attempt": 0.15,
            "pass_td_rate": 0.15,
            "int_rate": -0.10,
            "sack_rate": -0.10,
        },
    },
    "RB": {
        "min": ("carries", 80),
        "metrics": {
            "epa_per_rush": 0.30,
            "yards_per_carry": 0.20,
            "rush_first_down_rate": 0.20,
            "rush_td_rate": 0.10,
            "receiving_epa": 0.10,
            "yards_per_target": 0.10,
        },
    },
    "WR": {
        "min": ("targets", 45),
        "metrics": {
            "epa_per_target": 0.25,
            "yards_per_target": 0.20,
            "rec_first_down_rate": 0.15,
            "catch_rate": 0.10,
            "target_share": 0.15,
            "air_yards_share": 0.10,
            "wopr": 0.05,
        },
    },
    "TE": {
        "min": ("targets", 30),
        "metrics": {
            "epa_per_target": 0.25,
            "yards_per_target": 0.20,
            "rec_first_down_rate": 0.20,
            "catch_rate": 0.15,
            "target_share": 0.10,
            "wopr": 0.10,
        },
    },
    "EDGE": {
        "min": ("games", 8),
        "metrics": {
            "def_sacks_pg": 0.30,
            "def_qb_hits_pg": 0.30,
            "def_tackles_for_loss_pg": 0.20,
            "def_fumbles_forced_pg": 0.10,
            "def_tackles_solo_pg": 0.10,
        },
    },
    "DT": {
        "min": ("games", 8),
        "metrics": {
            "def_sacks_pg": 0.20,
            "def_qb_hits_pg": 0.25,
            "def_tackles_for_loss_pg": 0.25,
            "def_tackles_solo_pg": 0.20,
            "def_fumbles_forced_pg": 0.10,
        },
    },
    "LB": {
        "min": ("games", 8),
        "metrics": {
            "def_tackles_solo_pg": 0.25,
            "def_tackle_assists_pg": 0.10,
            "def_tackles_for_loss_pg": 0.20,
            "def_sacks_pg": 0.15,
            "def_qb_hits_pg": 0.10,
            "def_interceptions_pg": 0.10,
            "def_pass_defended_pg": 0.10,
        },
    },
    "CB": {
        "min": ("games", 8),
        "metrics": {
            "def_pass_defended_pg": 0.30,
            "def_interceptions_pg": 0.25,
            "def_tackles_solo_pg": 0.15,
            "def_tackles_for_loss_pg": 0.10,
            "def_fumbles_forced_pg": 0.10,
            "def_qb_hits_pg": 0.10,
        },
    },
    "S": {
        "min": ("games", 8),
        "metrics": {
            "def_tackles_solo_pg": 0.25,
            "def_pass_defended_pg": 0.20,
            "def_interceptions_pg": 0.20,
            "def_tackle_assists_pg": 0.10,
            "def_tackles_for_loss_pg": 0.10,
            "def_fumbles_forced_pg": 0.10,
            "def_qb_hits_pg": 0.05,
        },
    },
}


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def rank_players(season_df: pd.DataFrame, position: str) -> pd.DataFrame:
    pos = normalize_position(position)
    df = season_df.copy()
    df["rank_position"] = df["position"].apply(normalize_position)
    df = df[df["rank_position"].eq(pos)].copy()

    cfg = RANK_CONFIG.get(pos)
    if cfg is None or df.empty:
        return pd.DataFrame()

    min_col, min_val = cfg["min"]
    if min_col in df.columns:
        df = df[pd.to_numeric(df[min_col], errors="coerce").fillna(0) >= min_val].copy()

    available = {
        metric: weight
        for metric, weight in cfg["metrics"].items()
        if metric in df.columns and pd.to_numeric(df[metric], errors="coerce").notna().sum() >= 3
    }
    if not available or df.empty:
        return pd.DataFrame()

    total_weight = sum(abs(w) for w in available.values())
    score = pd.Series(0.0, index=df.index)

    for metric, weight in available.items():
        vals = pd.to_numeric(df[metric], errors="coerce")
        vals = vals.fillna(vals.median())
        score += _zscore(vals) * weight

    df["data_score"] = score / total_weight
    # 50 is league-average, ~65 strong, ~80 elite for presentation.
    df["rating"] = (50 + 15 * df["data_score"]).clip(0, 100)
    df = df.sort_values("rating", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["percentile"] = (1 - (df["rank"] - 1) / max(len(df), 1)) * 100
    df["metrics_used"] = ", ".join(available.keys())
    return df


# ---------------- Projections + breakout model ----------------

PROJECTION_METRICS = {
    "QB": ["epa_per_attempt", "passing_cpoe", "yards_per_attempt", "pass_td_rate", "int_rate"],
    "RB": ["epa_per_rush", "yards_per_carry", "rush_first_down_rate", "yards_per_target"],
    "WR": ["epa_per_target", "yards_per_target", "rec_first_down_rate", "target_share", "air_yards_share", "wopr"],
    "TE": ["epa_per_target", "yards_per_target", "rec_first_down_rate", "target_share", "wopr"],
    "EDGE": ["def_sacks_pg", "def_qb_hits_pg", "def_tackles_for_loss_pg"],
    "DT": ["def_qb_hits_pg", "def_tackles_for_loss_pg", "def_tackles_solo_pg"],
    "LB": ["def_tackles_solo_pg", "def_tackles_for_loss_pg", "def_sacks_pg", "def_pass_defended_pg"],
    "CB": ["def_pass_defended_pg", "def_interceptions_pg", "def_tackles_solo_pg"],
    "S": ["def_tackles_solo_pg", "def_pass_defended_pg", "def_interceptions_pg"],
}

VOLUME_METRIC = {
    "QB": "attempts", "RB": "carries", "WR": "targets", "TE": "targets",
    "EDGE": "games", "DT": "games", "LB": "games", "CB": "games", "S": "games"
}


# Minimum projected role required to appear in the main projected rankings.
# The public nflverse player-stat source used here does not contain a universal
# offensive/defensive snap-count field, so these are playing-time proxies.
PROJECTED_RANK_MINIMUMS = {
    "QB": 200,    # projected pass attempts
    "RB": 80,     # projected carries
    "WR": 50,     # projected targets
    "TE": 35,     # projected targets
    "EDGE": 8,    # projected games
    "DT": 8,
    "LB": 8,
    "CB": 8,
    "S": 8,
}


def build_projections(
    all_seasons: pd.DataFrame,
    position: str,
    target_season: int,
    mode: str = "preseason",
) -> pd.DataFrame:
    """
    Dynamic position-specific projection model.

    preseason:
      - target season has not started
      - 55% previous season, 30% two years back, 15% three years back

    in_season:
      - current season data already exists
      - efficiency: 60% current season-to-date, 25% prior year, 15% two years back
      - volume: current-season pace is projected toward a 17-game season and blended
        with prior full-season volume

    A capped recent-trend adjustment is applied in either mode.
    """
    pos = normalize_position(position)
    metrics = PROJECTION_METRICS.get(pos, [])
    if not metrics:
        return pd.DataFrame()

    df = all_seasons.copy()
    df["rank_position"] = df["position"].apply(normalize_position)
    df = df[df["rank_position"].eq(pos)].copy()
    if df.empty:
        return pd.DataFrame()

    if mode == "in_season":
        anchor = target_season
        season_weights = {
            target_season: 0.60,
            target_season - 1: 0.25,
            target_season - 2: 0.15,
        }
    else:
        anchor = target_season - 1
        season_weights = {
            anchor: 0.55,
            anchor - 1: 0.30,
            anchor - 2: 0.15,
        }

    anchor_df = df[df["season"].eq(anchor)].copy()
    if anchor_df.empty:
        return pd.DataFrame()

    id_cols = ["player_id", "player_name", "team", "position"]
    base = anchor_df[id_cols].drop_duplicates("player_id").copy()
    base["position"] = pos
    base["projection_season"] = target_season
    base["projection_mode"] = mode

    vol = VOLUME_METRIC.get(pos)
    usable_metrics = [m for m in metrics if m in df.columns]
    if vol and vol in df.columns:
        usable_metrics.append(vol)

    seasons = list(season_weights.keys())

    for metric in usable_metrics:
        work = df[df["season"].isin(seasons)][["player_id", "season", metric] + (
            ["games"] if "games" in df.columns and metric == vol else []
        )].copy()

        # Force projection inputs to numeric dtype up front. nflverse can expose
        # mixed/object columns in some season files; keeping them numeric prevents
        # pandas assignment errors during in-season pacing.
        work[metric] = pd.to_numeric(work[metric], errors="coerce").astype("float64")

        # In-season volume: convert current volume into a projected 17-game pace.
        if mode == "in_season" and metric == vol and "games" in work.columns:
            current_mask = work["season"].eq(target_season)
            games = (
                pd.to_numeric(work.loc[current_mask, "games"], errors="coerce")
                .astype("float64")
                .replace(0, np.nan)
            )
            current_volume = pd.to_numeric(
                work.loc[current_mask, metric], errors="coerce"
            ).astype("float64")
            pace = (current_volume / games * 17.0).astype("float64")
            work.loc[current_mask, metric] = pace.to_numpy()

        pivot = work.pivot_table(
            index="player_id",
            columns="season",
            values=metric,
            aggfunc="last",
        )

        weighted_sum = pd.Series(0.0, index=pivot.index)
        weight_sum = pd.Series(0.0, index=pivot.index)

        for season, wt in season_weights.items():
            if season in pivot.columns:
                valid = pivot[season].notna()
                weighted_sum.loc[valid] += pivot.loc[valid, season] * wt
                weight_sum.loc[valid] += wt

        projected = weighted_sum / weight_sum.replace(0, np.nan)

        # Recent trend adjustment.
        latest = anchor
        previous = anchor - 1
        if latest in pivot.columns and previous in pivot.columns:
            curr = pivot[latest]
            prev = pivot[previous]
            delta = curr - prev
            scale = projected.abs().where(projected.abs() > 1e-6, 1.0)
            capped_delta = delta.clip(lower=-0.35 * scale, upper=0.35 * scale)
            projected = projected + 0.20 * capped_delta.fillna(0)

        base = base.merge(
            projected.rename(f"proj_{metric}").reset_index(),
            on="player_id",
            how="left",
        )

    # Keep fringe players available for the breakout list but out of the main
    # projected rankings unless their projected role is meaningful.
    if vol:
        proj_vol = f"proj_{vol}"
        min_role = PROJECTED_RANK_MINIMUMS.get(pos)
        if proj_vol in base.columns and min_role is not None:
            base = base[
                pd.to_numeric(base[proj_vol], errors="coerce").fillna(0) >= min_role
            ].copy()

    if base.empty:
        return pd.DataFrame()

    projected_eff = [f"proj_{m}" for m in metrics if f"proj_{m}" in base.columns]
    projected_eff = [
        c for c in projected_eff
        if pd.to_numeric(base[c], errors="coerce").notna().sum() >= 3
    ]
    if not projected_eff:
        return pd.DataFrame()

    score = pd.Series(0.0, index=base.index)
    used_weight = 0.0
    rank_weights = RANK_CONFIG.get(pos, {}).get("metrics", {})

    for metric in metrics:
        col = f"proj_{metric}"
        if col not in projected_eff:
            continue

        vals = pd.to_numeric(base[col], errors="coerce")
        vals = vals.fillna(vals.median())
        weight = rank_weights.get(metric, 1.0)

        score += _zscore(vals) * weight
        used_weight += abs(weight)

    if used_weight == 0:
        return pd.DataFrame()

    base["projected_data_score"] = score / used_weight
    base["projected_rating"] = (50 + 15 * base["projected_data_score"]).clip(0, 100)
    base = base.sort_values("projected_rating", ascending=False).reset_index(drop=True)
    base["projected_rank"] = np.arange(1, len(base) + 1)

    return base

def breakout_candidates(all_seasons: pd.DataFrame, position: str, target_season: int = 2026) -> pd.DataFrame:
    """
    Data-driven breakout candidates.

    Rewards:
      - above-average latest-season efficiency
      - positive year-over-year improvement
      - meaningful but not already extreme usage

    This intentionally identifies upward-trending candidates, not simply the
    highest-ranked established stars.
    """
    pos = normalize_position(position)
    metrics = PROJECTION_METRICS.get(pos, [])
    if not metrics:
        return pd.DataFrame()

    df = all_seasons.copy()
    df["rank_position"] = df["position"].apply(normalize_position)
    df = df[df["rank_position"].eq(pos)].copy()

    latest = target_season - 1
    prev = latest - 1

    cur = df[df["season"].eq(latest)].copy()
    old = df[df["season"].eq(prev)].copy()
    if cur.empty:
        return pd.DataFrame()

    available = [m for m in metrics if m in cur.columns and cur[m].notna().sum() >= 3]
    if not available:
        return pd.DataFrame()

    keep_cur = ["player_id", "player_name", "team", "position"] + available
    vol_col = VOLUME_METRIC.get(pos)
    if vol_col and vol_col in cur.columns:
        keep_cur.append(vol_col)

    cur = cur[keep_cur].drop_duplicates("player_id").copy()

    old_cols = ["player_id"] + [m for m in available if m in old.columns]
    old_small = old[old_cols].drop_duplicates("player_id").copy()
    old_small = old_small.rename(columns={m: f"{m}_prev" for m in available if m in old_small.columns})

    joined = cur.merge(old_small, on="player_id", how="left")

    lower_better = {"int_rate", "sack_rate"}
    eff = pd.Series(0.0, index=joined.index)
    trend = pd.Series(0.0, index=joined.index)
    eff_count = 0
    trend_count = 0

    for metric in available:
        direction = -1 if metric in lower_better else 1
        current = pd.to_numeric(joined[metric], errors="coerce")
        current = current.fillna(current.median())
        eff += direction * _zscore(current)
        eff_count += 1

        prev_col = f"{metric}_prev"
        if prev_col in joined.columns:
            previous = pd.to_numeric(joined[prev_col], errors="coerce")
            has_history = previous.notna()
            if has_history.sum() >= 3:
                delta = current - previous
                trend += direction * _zscore(delta.fillna(0))
                trend_count += 1

    eff /= max(eff_count, 1)
    if trend_count:
        trend /= trend_count

    # Opportunity score: sufficient involvement, but avoid just returning the
    # position's highest-volume established stars.
    if vol_col and vol_col in joined.columns:
        volume = pd.to_numeric(joined[vol_col], errors="coerce").fillna(0)
        vol_pct = volume.rank(pct=True)
        opportunity = 1 - (vol_pct - 0.55).abs() * 2
        opportunity = opportunity.clip(-1, 1)
    else:
        opportunity = pd.Series(0.0, index=joined.index)

    joined["breakout_score"] = 0.50 * eff + 0.35 * trend + 0.15 * opportunity
    joined["breakout_rating"] = (50 + 15 * joined["breakout_score"]).clip(0, 100)

    # Require some real usage when possible.
    if vol_col and vol_col in joined.columns:
        thresholds = {
            "QB": 100, "RB": 35, "WR": 25, "TE": 15,
            "EDGE": 4, "DT": 4, "LB": 4, "CB": 4, "S": 4,
        }
        min_usage = thresholds.get(pos, 0)
        joined = joined[pd.to_numeric(joined[vol_col], errors="coerce").fillna(0) >= min_usage]

    joined = joined.sort_values("breakout_rating", ascending=False).reset_index(drop=True)
    joined["breakout_rank"] = np.arange(1, len(joined) + 1)
    return joined

