from __future__ import annotations

import pandas as pd
import re
from datetime import date
import plotly.express as px
import streamlit as st

from analytics import (
    aggregate_player_seasons,
    breakout_candidates,
    build_def_metrics,
    build_off_metrics,
    build_projections,
    build_team_epa,
    rank_players,
)
from data_loader import (
    DEFAULT_SEASONS,
    available_seasons,
    latest_available_pbp_season,
    latest_available_player_season,
    load_pbp,
    load_player_stats,
    load_teams,
    nfl_season_for_date,
)


st.set_page_config(page_title="NFL Analytics Dashboard", layout="wide")


COLUMN_LABELS = {
    "season": "Season",
    "team": "Team",
    "plays": "Plays",
    "off_epa": "Off EPA/Play",
    "def_epa": "Def EPA/Play",
    "off_epa_onescore": "1-Score Off EPA/Play",
    "def_epa_onescore": "1-Score Def EPA/Play",
    "off_success_rate": "Off Success Rate",
    "def_success_rate": "Def Success Rate",
    "pressure_rate": "Pressure Rate",
    "explosive_pass_rate": "Explosive Pass Rate",
    "explosive_run_rate": "Explosive Run Rate",
    "turnover_rate": "Turnover Rate",
    "third_down_conv_rate": "3rd Down Conversion Rate",
    "rank": "Rank",
    "player_name": "Player",
    "rating": "Data Rating",
    "percentile": "Percentile",
    "games": "Games",
    "attempts": "Pass Attempts",
    "carries": "Carries",
    "targets": "Targets",
    "epa_per_attempt": "EPA/Attempt",
    "passing_cpoe": "CPOE",
    "yards_per_attempt": "Yards/Attempt",
    "pass_td_rate": "Pass TD Rate",
    "int_rate": "INT Rate",
    "sack_rate": "Sack Rate",
    "epa_per_rush": "EPA/Rush",
    "yards_per_carry": "Yards/Carry",
    "rush_first_down_rate": "Rush 1st-Down Rate",
    "yards_per_target": "Yards/Target",
    "epa_per_target": "EPA/Target",
    "rec_first_down_rate": "Rec 1st-Down Rate",
    "catch_rate": "Catch Rate",
    "target_share": "Target Share",
    "air_yards_share": "Air-Yards Share",
    "wopr": "WOPR",
    "def_sacks_pg": "Sacks/Game",
    "def_qb_hits_pg": "QB Hits/Game",
    "def_tackles_for_loss_pg": "TFL/Game",
    "def_fumbles_forced_pg": "Forced Fumbles/Game",
    "def_tackles_solo_pg": "Solo Tackles/Game",
    "def_tackle_assists_pg": "Assisted Tackles/Game",
    "def_interceptions_pg": "INTs/Game",
    "def_pass_defended_pg": "Passes Defended/Game",
    "projected_rank": "Projected Rank",
    "projected_rating": "Projected Rating",
    "proj_attempts": "Projected Pass Attempts",
    "proj_carries": "Projected Carries",
    "proj_targets": "Projected Targets",
    "proj_games": "Projected Games",
    "proj_epa_per_attempt": "Projected EPA/Attempt",
    "proj_passing_cpoe": "Projected CPOE",
    "proj_yards_per_attempt": "Projected Yards/Attempt",
    "proj_pass_td_rate": "Projected Pass TD Rate",
    "proj_int_rate": "Projected INT Rate",
    "proj_epa_per_rush": "Projected EPA/Rush",
    "proj_yards_per_carry": "Projected Yards/Carry",
    "proj_rush_first_down_rate": "Projected Rush 1st-Down Rate",
    "proj_yards_per_target": "Projected Yards/Target",
    "proj_epa_per_target": "Projected EPA/Target",
    "proj_rec_first_down_rate": "Projected Rec 1st-Down Rate",
    "proj_target_share": "Projected Target Share",
    "proj_air_yards_share": "Projected Air-Yards Share",
    "proj_wopr": "Projected WOPR",
    "proj_def_sacks_pg": "Projected Sacks/Game",
    "proj_def_qb_hits_pg": "Projected QB Hits/Game",
    "proj_def_tackles_for_loss_pg": "Projected TFL/Game",
    "proj_def_tackles_solo_pg": "Projected Solo Tackles/Game",
    "proj_def_pass_defended_pg": "Projected Passes Defended/Game",
    "proj_def_interceptions_pg": "Projected INTs/Game",
    "breakout_rank": "Breakout Rank",
    "breakout_rating": "Breakout Rating",
}

PERCENT_COLUMNS = {
    "off_success_rate", "def_success_rate", "pressure_rate",
    "explosive_pass_rate", "explosive_run_rate", "turnover_rate",
    "third_down_conv_rate", "pass_td_rate", "int_rate", "sack_rate",
    "rush_first_down_rate", "rec_first_down_rate", "catch_rate",
    "target_share", "air_yards_share", "percentile",
    "proj_pass_td_rate", "proj_int_rate", "proj_rush_first_down_rate",
    "proj_rec_first_down_rate", "proj_target_share", "proj_air_yards_share",
}


def clean_table(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Return a presentation-friendly copy with readable headers and % values."""
    if columns is not None:
        columns = [c for c in columns if c in df.columns]
        out = df[columns].copy()
    else:
        out = df.copy()

    for col in PERCENT_COLUMNS.intersection(out.columns):
        vals = pd.to_numeric(out[col], errors="coerce")
        # percentile is already 0-100; rate metrics are 0-1.
        if col != "percentile":
            out[col] = vals * 100

    out = out.rename(columns={c: COLUMN_LABELS.get(c, c.replace("_", " ").title()) for c in out.columns})
    return out.round(3)


def safe_hex(value, fallback="#2563EB"):
    """Normalize team colors from nflfastR into usable CSS hex values."""
    if not isinstance(value, str):
        return fallback
    value = value.strip()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        return value.upper()
    if re.fullmatch(r"[0-9A-Fa-f]{6}", value):
        return f"#{value.upper()}"
    return fallback


def team_theme(teams_df: pd.DataFrame, team_abbr: str | None):
    """Return primary, secondary, team name and logo for the selected team."""
    default = ("#0F172A", "#2563EB", "NFL", None)
    if not team_abbr or team_abbr == "All Teams" or teams_df.empty:
        return default

    if "team_abbr" not in teams_df.columns:
        return default

    row = teams_df[teams_df["team_abbr"].astype(str).eq(team_abbr)]
    if row.empty:
        return default

    r = row.iloc[0]
    primary = safe_hex(r.get("team_color"), "#0F172A")
    secondary = safe_hex(r.get("team_color2"), "#2563EB")
    if secondary == primary:
        secondary = "#F8FAFC"

    name = str(r.get("team_name", team_abbr))
    logo = r.get("team_logo_espn")
    if pd.isna(logo):
        logo = None

    return primary, secondary, name, logo


def inject_theme():
    """Light league-wide theme used on every tab except active Team Detail."""
    st.markdown(
        """
        <style>
        :root {
            --primary: #2563EB;
            --secondary: #7C3AED;
            --panel: rgba(255,255,255,.84);
            --panel-soft: rgba(255,255,255,.68);
            --border: rgba(30,41,59,.11);
            --text: #172033;
            --text-soft: #64748B;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 3%, rgba(96,165,250,.23) 0%, transparent 27%),
                radial-gradient(circle at 92% 8%, rgba(167,139,250,.20) 0%, transparent 30%),
                linear-gradient(145deg, #F8FBFF 0%, #EEF5FF 47%, #F7F3FF 100%);
            color: var(--text);
            transition: background .35s ease;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255,255,255,.93), rgba(239,246,255,.94));
            border-right: 1px solid rgba(37,99,235,.12);
        }

        [data-testid="stHeader"] {
            background: rgba(248,251,255,.78);
            backdrop-filter: blur(12px);
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-size: 2.55rem !important;
            letter-spacing: -0.045em;
            font-weight: 850 !important;
            background: linear-gradient(90deg, #172033 0%, #2563EB 58%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        h2, h3 {
            color: #172033 !important;
            letter-spacing: -0.02em;
        }

        p, .stCaption {
            color: var(--text-soft);
        }

        [data-baseweb="tab-list"] {
            gap: .45rem;
            padding: .35rem;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: rgba(255,255,255,.72);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 28px rgba(30,41,59,.06);
        }

        [data-baseweb="tab"] {
            border-radius: 12px;
            padding: .55rem .9rem;
            color: #334155 !important;
        }

        [aria-selected="true"][data-baseweb="tab"] {
            background: linear-gradient(135deg, rgba(37,99,235,.16), rgba(124,58,237,.14));
            color: #1D4ED8 !important;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(255,255,255,.96), rgba(248,250,252,.88));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 35px rgba(30,41,59,.07);
        }

        [data-testid="stMetricLabel"] {
            color: #64748B;
        }

        [data-testid="stMetricValue"] {
            color: #172033;
            font-weight: 800;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 12px 34px rgba(30,41,59,.06);
        }

        [data-testid="stPlotlyChart"] {
            background: rgba(255,255,255,.74);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: .45rem;
            box-shadow: 0 12px 34px rgba(30,41,59,.06);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 12px !important;
            border-color: rgba(37,99,235,.24) !important;
            background: rgba(255,255,255,.92) !important;
        }

        .team-hero {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1.15rem 1.25rem;
            margin: .35rem 0 1.25rem 0;
            border-radius: 20px;
            box-shadow: 0 16px 42px rgba(30,41,59,.12);
        }

        .team-hero img {
            width: 68px;
            height: 68px;
            object-fit: contain;
            filter: drop-shadow(0 8px 18px rgba(0,0,0,.18));
        }

        .team-hero-title {
            font-size: 1.42rem;
            font-weight: 800;
            color: white;
            line-height: 1.1;
        }

        .team-hero-subtitle {
            color: rgba(255,255,255,.88);
            margin-top: .28rem;
            font-size: .92rem;
        }

        hr {
            border-color: rgba(30,41,59,.10) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_team_detail_theme(primary: str, secondary: str):
    """
    Recolor the page only while the Team Detail tab (4th tab) is active.
    CSS :has() lets the browser respond instantly when the user switches tabs.
    """
    st.markdown(
        f"""
        <style>
        .stApp:has([data-baseweb="tab"]:nth-child(4)[aria-selected="true"]) {{
            background:
                radial-gradient(circle at 8% 4%, {primary}35 0%, transparent 30%),
                radial-gradient(circle at 92% 8%, {secondary}2F 0%, transparent 31%),
                linear-gradient(145deg, #F8FAFC 0%, {primary}12 50%, {secondary}16 100%);
        }}

        .stApp:has([data-baseweb="tab"]:nth-child(4)[aria-selected="true"])
        [data-testid="stHeader"] {{
            background: color-mix(in srgb, {primary} 8%, white 92%);
        }}

        .stApp:has([data-baseweb="tab"]:nth-child(4)[aria-selected="true"])
        [data-baseweb="tab"]:nth-child(4)[aria-selected="true"] {{
            background: linear-gradient(135deg, {primary}22, {secondary}28);
            color: {primary} !important;
        }}

        .stApp:has([data-baseweb="tab"]:nth-child(4)[aria-selected="true"])
        [data-testid="stMetric"] {{
            border-color: {secondary}35;
            box-shadow: 0 12px 35px {primary}14;
        }}

        .stApp:has([data-baseweb="tab"]:nth-child(4)[aria-selected="true"])
        [data-testid="stPlotlyChart"],
        .stApp:has([data-baseweb="tab"]:nth-child(4)[aria-selected="true"])
        [data-testid="stDataFrame"] {{
            border-color: {secondary}30;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def style_plot(fig, primary: str = "#2563EB", secondary: str = "#7C3AED"):
    """Match Plotly to the light gradient dashboard."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.30)",
        font=dict(color="#334155"),
        title_font=dict(color="#172033", size=20),
        xaxis=dict(
            gridcolor="rgba(100,116,139,.13)",
            zerolinecolor="rgba(100,116,139,.18)",
        ),
        yaxis=dict(
            gridcolor="rgba(100,116,139,.13)",
            zerolinecolor="rgba(100,116,139,.18)",
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_color="#172033",
            bordercolor=secondary,
        ),
    )
    return fig


@st.cache_data(ttl=21600, show_spinner=False)
def get_season_data(season: int):
    pbp = load_pbp(season)
    return build_team_epa(pbp), build_off_metrics(pbp), build_def_metrics(pbp)


@st.cache_data(ttl=21600, show_spinner=False)
def get_player_seasons():
    latest = latest_available_player_season()
    seasons = tuple(s for s in available_seasons() if s <= latest)
    stats = load_player_stats(seasons)
    return aggregate_player_seasons(stats)


@st.cache_data(ttl=21600, show_spinner=False)
def get_teams():
    return load_teams()


teams = get_teams()

latest_pbp = latest_available_pbp_season()
season_options = [s for s in available_seasons() if s <= latest_pbp]
if not season_options:
    season_options = list(DEFAULT_SEASONS)

season = st.sidebar.selectbox(
    "Season",
    options=season_options,
    index=len(season_options) - 1,
)

inject_theme()

st.title("NFL Analytics Dashboard")
st.caption(
    "Public nflverse data is downloaded automatically. Rankings and projections "
    "are built from regular-season player data."
)

with st.spinner(f"Loading {season} team data..."):
    try:
        team_epa, off_metrics, def_metrics = get_season_data(season)
    except Exception as exc:
        st.error("Could not load nflverse play-by-play data.")
        st.exception(exc)
        st.stop()

tabs = st.tabs([
    "Team EPA",
    "Offensive Metrics",
    "Defensive Metrics",
    "Team Detail",
    "Player Rankings",
    "Projections & Breakouts",
])

# -------- Team EPA --------
with tabs[0]:
    st.subheader("Offense vs Defense EPA per Play")
    situation = st.radio("Situation", ["All plays", "One-score games (±8 points)"], horizontal=True)
    x_col = "off_epa" if situation == "All plays" else "off_epa_onescore"
    y_col = "def_epa" if situation == "All plays" else "def_epa_onescore"

    df = team_epa.dropna(subset=[x_col, y_col]).copy()

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        text="team",
        hover_name="team",
        color_discrete_sequence=["#2563EB"],
        labels={x_col: "Offensive EPA/play", y_col: "Defensive EPA/play (higher = better)"},
        title=f"{season} NFL Team EPA",
    )
    fig.update_traces(textposition="top center", marker=dict(size=11))
    table_df = df

    style_plot(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(clean_table(table_df), use_container_width=True, hide_index=True)

# -------- Offense --------
with tabs[1]:
    st.subheader("Offensive Profile")
    metric_map = {
        "EPA/play": "off_epa",
        "One-score EPA/play": "off_epa_onescore",
        "Success Rate": "off_success_rate",
        "Explosive Pass Rate": "explosive_pass_rate",
        "Explosive Run Rate": "explosive_run_rate",
        "Turnover Rate": "turnover_rate",
        "3rd Down Conversion Rate": "third_down_conv_rate",
    }
    label = st.selectbox("Metric", list(metric_map), key="off_metric")
    col = metric_map[label]
    plot_df = off_metrics[["team", col]].dropna().copy()
    if col not in {"off_epa", "off_epa_onescore"}:
        plot_df[col] *= 100
    plot_df = plot_df.sort_values(col, ascending=False)

    fig_off = px.bar(
        plot_df,
        x="team",
        y=col,
        color_discrete_sequence=["#2563EB"],
        title=f"{label} — {season}",
    )
    off_table = off_metrics

    style_plot(fig_off)
    st.plotly_chart(fig_off, use_container_width=True)
    st.dataframe(
        clean_table(
            off_table,
            ["season", "team", "plays", "off_epa", "off_epa_onescore",
             "off_success_rate", "explosive_pass_rate", "explosive_run_rate",
             "turnover_rate", "third_down_conv_rate"],
        ),
        use_container_width=True,
        hide_index=True,
    )

# -------- Defense --------
with tabs[2]:
    st.subheader("Defensive Profile")
    metric_map = {
        "Pressure Rate": "pressure_rate",
        "Defensive Success Rate": "def_success_rate",
        "Explosive Pass Rate Allowed": "explosive_pass_rate",
        "Explosive Run Rate Allowed": "explosive_run_rate",
        "Turnover Rate Forced": "turnover_rate",
        "3rd Down Conversion Rate Allowed": "third_down_conv_rate",
    }
    label = st.selectbox("Metric", list(metric_map), key="def_metric")
    col = metric_map[label]
    plot_df = def_metrics[["team", col]].dropna().copy()
    plot_df[col] *= 100
    plot_df = plot_df.sort_values(col, ascending=False)

    fig_def = px.bar(
        plot_df,
        x="team",
        y=col,
        color_discrete_sequence=["#7C3AED"],
        title=f"{label} — {season}",
    )
    def_table = def_metrics

    style_plot(fig_def)
    st.plotly_chart(fig_def, use_container_width=True)
    st.dataframe(
        clean_table(
            def_table,
            ["season", "team", "plays", "pressure_rate", "def_success_rate",
             "explosive_pass_rate", "explosive_run_rate", "turnover_rate",
             "third_down_conv_rate"],
        ),
        use_container_width=True,
        hide_index=True,
    )

# -------- Team Detail --------
with tabs[3]:
    st.subheader("Team Detail")
    team_list = sorted(team_epa["team"].dropna().unique())
    team = st.selectbox("Team", team_list, index=0)

    detail_primary, detail_secondary, detail_name, detail_logo = team_theme(teams, team)
    inject_team_detail_theme(detail_primary, detail_secondary)
    detail_logo_html = f'<img src="{detail_logo}" />' if detail_logo else ""
    st.markdown(
        f"""
        <div class="team-hero" style="background:linear-gradient(110deg,{detail_primary}DD 0%,{detail_primary}99 48%,{detail_secondary}66 100%);border-color:{detail_secondary}66;">
            {detail_logo_html}
            <div>
                <div class="team-hero-title">{detail_name}</div>
                <div class="team-hero-subtitle">{season} offense + defense profile</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    e = team_epa[team_epa["team"].eq(team)]
    o = off_metrics[off_metrics["team"].eq(team)]
    d = def_metrics[def_metrics["team"].eq(team)]

    if not e.empty:
        r = e.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Off EPA/play", f"{r['off_epa']:.3f}")
        c2.metric("Def EPA/play", f"{r['def_epa']:.3f}")
        c3.metric("1-score Off EPA", f"{r['off_epa_onescore']:.3f}")
        c4.metric("1-score Def EPA", f"{r['def_epa_onescore']:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Offense")
        st.dataframe(
            clean_table(
                o,
                ["team", "plays", "off_epa", "off_epa_onescore",
                 "off_success_rate", "explosive_pass_rate", "explosive_run_rate",
                 "turnover_rate", "third_down_conv_rate"],
            ),
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.markdown("#### Defense")
        st.dataframe(
            clean_table(
                d,
                ["team", "plays", "pressure_rate", "def_success_rate",
                 "explosive_pass_rate", "explosive_run_rate", "turnover_rate",
                 "third_down_conv_rate"],
            ),
            use_container_width=True,
            hide_index=True,
        )

# -------- Player Rankings --------
with tabs[4]:
    st.subheader("Data-Driven Player Rankings")
    st.caption(
        "Rankings compare each player only against qualifying players at the same position."
    )

    try:
        with st.spinner("Loading player-season data..."):
            player_seasons = get_player_seasons()
    except Exception as exc:
        st.error("Could not build player-season data.")
        st.exception(exc)
    else:
        positions = ["QB", "RB", "WR", "TE", "EDGE", "DT", "LB", "CB", "S"]
        position = st.selectbox("Position", positions, key="rank_pos")

        season_players = player_seasons[player_seasons["season"].eq(season)].copy()
        st.caption(f"Loaded {len(season_players):,} regular-season player rows for {season}.")
        ranked = rank_players(season_players, position)
        if ranked.empty:
            st.warning(
                "The source loaded, but there were not enough qualifying players/metrics "
                "for this position. Try another position or season."
            )
        else:
            st.markdown(f"### {season} {position} Rankings")

            top_n = st.slider("Players to show", 10, min(100, len(ranked)), min(40, len(ranked)))
            show = ranked.head(top_n).copy()

            metric_cols = {
                "QB": ["attempts", "epa_per_attempt", "passing_cpoe", "yards_per_attempt", "pass_td_rate", "int_rate", "sack_rate"],
                "RB": ["carries", "epa_per_rush", "yards_per_carry", "rush_first_down_rate", "yards_per_target"],
                "WR": ["targets", "epa_per_target", "yards_per_target", "rec_first_down_rate", "catch_rate", "target_share", "air_yards_share", "wopr"],
                "TE": ["targets", "epa_per_target", "yards_per_target", "rec_first_down_rate", "catch_rate", "target_share", "wopr"],
                "EDGE": ["games", "def_sacks_pg", "def_qb_hits_pg", "def_tackles_for_loss_pg", "def_fumbles_forced_pg"],
                "DT": ["games", "def_sacks_pg", "def_qb_hits_pg", "def_tackles_for_loss_pg", "def_tackles_solo_pg"],
                "LB": ["games", "def_tackles_solo_pg", "def_tackles_for_loss_pg", "def_sacks_pg", "def_qb_hits_pg", "def_interceptions_pg", "def_pass_defended_pg"],
                "CB": ["games", "def_pass_defended_pg", "def_interceptions_pg", "def_tackles_solo_pg"],
                "S": ["games", "def_tackles_solo_pg", "def_pass_defended_pg", "def_interceptions_pg"],
            }

            cols = ["rank", "player_name", "team", "rating", "percentile"] + metric_cols[position]
            cols = [c for c in cols if c in show.columns]
            display = show[cols].copy()

            st.dataframe(clean_table(display), use_container_width=True, hide_index=True)

            chart = show.head(25).sort_values("rating")
            fig_rank = px.bar(
                chart,
                x="rating",
                y="player_name",
                orientation="h",
                hover_data=["team", "rank", "percentile"],
                color_discrete_sequence=["#2563EB"],
                title=f"Top {min(25, len(show))} {position}s — {season}",
            )
            style_plot(fig_rank)
            st.plotly_chart(fig_rank, use_container_width=True)

            with st.expander("How the ranking is calculated"):
                st.write(
                    "Metrics are standardized within the selected position and combined "
                    "using position-specific weights. The score therefore answers: "
                    "'How strong does this player's statistical profile look relative to "
                    "other qualifying players at the same position?'"
                )
                st.write(
                    "Minimum-volume filters prevent tiny samples from ranking at the top. "
                    "This is a transparent public-data model, not a proprietary scouting grade."
                )

# -------- Projections / Breakouts --------
with tabs[5]:
    latest_player_season = latest_available_player_season()
    today = date.today()

    # Determine the season users actually care about:
    # Jan-Feb -> the NFL season that began the prior calendar year is still active.
    # Mar-Aug -> preseason/upcoming season is the current calendar year.
    # Sep-Dec -> current calendar year is the active season.
    expected_season = today.year - 1 if today.month <= 2 else today.year

    # Only switch to rest-of-season mode when nflverse truly has player data for
    # that expected season. In August 2026, for example, latest data is still 2025,
    # so this correctly remains "2026 Preseason" rather than "2025 Rest-of-Season".
    if latest_player_season >= expected_season:
        projection_mode = "in_season"
        projection_year = expected_season
        projection_title = f"{projection_year} Rest-of-Season Projections & Breakout Candidates"
        projection_caption = (
            f"Uses live {projection_year} performance plus prior seasons. Current-season "
            f"efficiency is weighted most heavily, and volume is paced toward a 17-game season."
        )
    else:
        projection_mode = "preseason"
        projection_year = expected_season
        projection_title = f"{projection_year} Preseason Projections & Breakout Candidates"
        projection_caption = (
            f"Built from the latest available nflverse regular-season data. The model weights "
            f"{latest_player_season} most heavily, then the two prior seasons."
        )

    st.subheader(projection_title)
    st.caption(projection_caption)

    try:
        player_seasons = get_player_seasons()
    except Exception as exc:
        st.error("Could not load player seasons for projections.")
        st.exception(exc)
    else:
        positions = ["QB", "RB", "WR", "TE", "EDGE", "DT", "LB", "CB", "S"]
        proj_pos = st.selectbox("Position", positions, key="proj_pos")

        projection_error = None
        with st.spinner(f"Building {projection_year} {proj_pos} projections..."):
            try:
                projections = build_projections(
                    player_seasons,
                    position=proj_pos,
                    target_season=projection_year,
                    mode=projection_mode,
                )
                breakout = breakout_candidates(
                    player_seasons,
                    proj_pos,
                    target_season=projection_year,
                )
            except Exception as exc:
                projections = pd.DataFrame()
                breakout = pd.DataFrame()
                projection_error = exc

        if projection_error is not None:
            st.error(
                "The projection model hit a data-format issue. The rest of the dashboard "
                "will keep working."
            )
            with st.expander("Projection error details"):
                st.exception(projection_error)

        if projections.empty:
            st.warning(
                f"No {projection_year} {proj_pos} projections could be built from the available data."
            )
        else:
            ranking_label = (
                f"Projected {projection_year} {proj_pos} Rankings"
                if projection_mode == "preseason"
                else f"Projected Rest-of-Season {projection_year} {proj_pos} Rankings"
            )
            st.markdown(f"### {ranking_label}")

            role_notes = {
                "QB": "200 projected pass attempts",
                "RB": "80 projected carries",
                "WR": "50 projected targets",
                "TE": "35 projected targets",
                "EDGE": "8 projected games",
                "DT": "8 projected games",
                "LB": "8 projected games",
                "CB": "8 projected games",
                "S": "8 projected games",
            }
            st.caption(
                f"Main rankings require a meaningful projected role ({role_notes[proj_pos]}). "
                f"Lower-volume players can still appear in Breakout Candidates."
            )

            proj_cols = ["projected_rank", "player_name", "team", "projected_rating"]

            metric_map = {
                "QB": ["proj_attempts", "proj_epa_per_attempt", "proj_passing_cpoe",
                       "proj_yards_per_attempt", "proj_pass_td_rate", "proj_int_rate"],
                "RB": ["proj_carries", "proj_epa_per_rush", "proj_yards_per_carry",
                       "proj_rush_first_down_rate", "proj_yards_per_target"],
                "WR": ["proj_targets", "proj_epa_per_target", "proj_yards_per_target",
                       "proj_rec_first_down_rate", "proj_target_share",
                       "proj_air_yards_share", "proj_wopr"],
                "TE": ["proj_targets", "proj_epa_per_target", "proj_yards_per_target",
                       "proj_rec_first_down_rate", "proj_target_share", "proj_wopr"],
                "EDGE": ["proj_games", "proj_def_sacks_pg", "proj_def_qb_hits_pg",
                         "proj_def_tackles_for_loss_pg"],
                "DT": ["proj_games", "proj_def_qb_hits_pg",
                       "proj_def_tackles_for_loss_pg", "proj_def_tackles_solo_pg"],
                "LB": ["proj_games", "proj_def_tackles_solo_pg",
                       "proj_def_tackles_for_loss_pg", "proj_def_sacks_pg",
                       "proj_def_pass_defended_pg"],
                "CB": ["proj_games", "proj_def_pass_defended_pg",
                       "proj_def_interceptions_pg", "proj_def_tackles_solo_pg"],
                "S": ["proj_games", "proj_def_tackles_solo_pg",
                      "proj_def_pass_defended_pg", "proj_def_interceptions_pg"],
            }

            proj_cols += metric_map[proj_pos]
            proj_cols = [c for c in proj_cols if c in projections.columns]

            st.dataframe(
                clean_table(projections.head(50), proj_cols),
                use_container_width=True,
                hide_index=True,
            )

            chart = projections.head(20).sort_values("projected_rating")
            fig_proj = px.bar(
                chart,
                x="projected_rating",
                y="player_name",
                orientation="h",
                hover_data=["team", "projected_rank"],
                color_discrete_sequence=["#2563EB"],
                title=f"Top 20 Projected {proj_pos}s — {projection_year}",
            )
            style_plot(fig_proj)
            st.plotly_chart(fig_proj, use_container_width=True)

        st.markdown("---")
        st.markdown(f"### {projection_year} {proj_pos} Breakout Candidates")

        if breakout.empty:
            st.info("No breakout candidates could be built for this position.")
        else:
            breakout_cols = [
                c for c in [
                    "breakout_rank",
                    "player_name",
                    "team",
                    "breakout_rating",
                    "targets",
                    "carries",
                    "attempts",
                    "games",
                ]
                if c in breakout.columns
            ]
            st.dataframe(
                clean_table(breakout.head(20), breakout_cols),
                use_container_width=True,
                hide_index=True,
            )

            chart = breakout.head(15).sort_values("breakout_rating")
            fig_breakout = px.bar(
                chart,
                x="breakout_rating",
                y="player_name",
                orientation="h",
                hover_data=["team", "breakout_rank"],
                color_discrete_sequence=["#7C3AED"],
                title=f"Top 15 {proj_pos} Breakout Candidates — {projection_year}",
            )
            style_plot(fig_breakout)
            st.plotly_chart(fig_breakout, use_container_width=True)

        with st.expander("How the automatic season logic works"):
            if projection_mode == "preseason":
                st.write(
                    f"The app is currently in preseason mode. It uses the latest available "
                    f"completed season ({latest_player_season}) plus the two seasons before it "
                    f"to project {projection_year}."
                )
            else:
                st.write(
                    f"The app detected live {current_nfl_season} nflverse player data, so it "
                    f"automatically switched into rest-of-season mode. Current-season efficiency "
                    f"is weighted 60%, with the prior two seasons providing stability."
                )
            st.write(
                "The season dropdown also updates automatically using the NFL calendar and the "
                "newest nflverse play-by-play file that is actually available."
            )

st.sidebar.markdown("---")
if st.sidebar.button("Clear cached data"):
    st.cache_data.clear()
    st.rerun()
