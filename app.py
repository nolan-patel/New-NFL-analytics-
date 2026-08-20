import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="NFL Analytics Dashboard",
    page_icon="🏈",
    layout="wide",
)

SEASONS = [2023, 2024, 2025]

st.markdown("""
<style>
.block-container {max-width:1450px;padding-top:1.3rem;padding-bottom:2rem}
[data-testid="stMetric"]{background:white;border:1px solid #e5e7eb;padding:12px 14px;border-radius:14px}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def import_nfl():
    import nfl_data_py as nfl
    return nfl

@st.cache_data(ttl=3600, show_spinner=False)
def load_pbp(season):
    nfl = import_nfl()
    cols = [
        "season","season_type","week","posteam","defteam","play_type","epa",
        "pass","rush","qb_dropback","sack","qb_hit","interception","yards_gained"
    ]
    df = nfl.import_pbp_data([season], columns=cols, downcast=True, cache=False)
    if "season_type" in df.columns:
        df = df[df["season_type"].eq("REG")]
    return df

def unit_summary(df, team_col, prefix):
    work = df[
        df["play_type"].isin(["pass","run"])
        & df["epa"].notna()
        & df[team_col].notna()
    ].copy()
    work["success"] = work["epa"] > 0

    base = (
        work.groupby(team_col)
        .agg(
            epa=("epa","mean"),
            success=("success","mean"),
            plays=("epa","size"),
        )
        .reset_index()
        .rename(columns={team_col:"team"})
    )

    pass_epa = (
        work[work["play_type"].eq("pass")]
        .groupby(team_col)["epa"].mean().rename("pass_epa")
        .reset_index().rename(columns={team_col:"team"})
    )
    rush_epa = (
        work[work["play_type"].eq("run")]
        .groupby(team_col)["epa"].mean().rename("rush_epa")
        .reset_index().rename(columns={team_col:"team"})
    )

    out = base.merge(pass_epa,on="team",how="left").merge(rush_epa,on="team",how="left")
    return out.rename(columns={
        "epa":f"{prefix}_epa",
        "success":f"{prefix}_success",
        "plays":f"{prefix}_plays",
        "pass_epa":f"{prefix}_pass_epa",
        "rush_epa":f"{prefix}_rush_epa",
    })

@st.cache_data(ttl=3600, show_spinner=False)
def load_team_metrics(season):
    pbp = load_pbp(season)
    off = unit_summary(pbp, "posteam", "off")
    deff = unit_summary(pbp, "defteam", "def")
    return off.merge(deff,on="team",how="outer").sort_values("team").reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def load_players(season):
    nfl = import_nfl()
    weekly = nfl.import_weekly_data([season], columns=None, downcast=True)
    if weekly.empty:
        return pd.DataFrame()

    name_col = "player_display_name" if "player_display_name" in weekly.columns else "player_name"
    team_col = "recent_team" if "recent_team" in weekly.columns else "team"

    sum_cols = [c for c in [
        "completions","attempts","passing_yards","passing_tds","interceptions",
        "carries","rushing_yards","rushing_tds","targets","receptions",
        "receiving_yards","receiving_tds","receiving_yards_after_catch",
        "fantasy_points"
    ] if c in weekly.columns]

    keys = [c for c in ["player_id",name_col,"position",team_col] if c in weekly.columns]
    out = weekly.groupby(keys, dropna=False)[sum_cols].sum().reset_index()

    rename = {}
    if name_col != "player_display_name":
        rename[name_col] = "player_display_name"
    if team_col != "recent_team":
        rename[team_col] = "recent_team"
    return out.rename(columns=rename)

st.title("NFL Analytics Dashboard")
st.caption("Team efficiency, player rankings, and multi-season trends using nflverse data.")

with st.sidebar:
    season = st.selectbox("Season", SEASONS, index=2)
    side = st.radio("Unit", ["Offense","Defense"], horizontal=True)

with st.spinner(f"Loading {season} data..."):
    team_df = load_team_metrics(season)
    player_df = load_players(season)

unit_col = "off_epa" if side == "Offense" else "def_epa"
success_col = "off_success" if side == "Offense" else "def_success"
pass_col = "off_pass_epa" if side == "Offense" else "def_pass_epa"
rush_col = "off_rush_epa" if side == "Offense" else "def_rush_epa"

ranked = team_df.copy()
ranked["rank"] = ranked[unit_col].rank(
    method="min",
    ascending=(side=="Defense")
).astype("Int64")
ranked = ranked.sort_values(unit_col, ascending=(side=="Defense"))

team_options = ["All Teams"] + sorted(ranked["team"].dropna().unique().tolist())
selected_team = st.selectbox("Team", team_options)

focus = ranked.iloc[0] if selected_team=="All Teams" else ranked[ranked["team"]==selected_team].iloc[0]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Best Team", ranked.iloc[0]["team"], f'{ranked.iloc[0][unit_col]:+.3f} EPA/play')
c2.metric("Selected Rank", f'#{int(focus["rank"])}')
c3.metric("Pass EPA", f'{focus[pass_col]:+.3f}')
c4.metric("Rush EPA", f'{focus[rush_col]:+.3f}')

team_tab, player_tab, trends_tab, about_tab = st.tabs(
    ["Team Explorer","Player Rankings","Trends","Methodology"]
)

with team_tab:
    left,right = st.columns([1.15,.85])

    with left:
        shown = ranked if selected_team=="All Teams" else ranked[ranked["team"]==selected_team]
        display = shown[["rank","team",unit_col,success_col,pass_col,rush_col]].copy()
        display.columns = ["Rank","Team","EPA / Play","Success Rate","EPA / Pass","EPA / Rush"]
        display["Success Rate"] = (display["Success Rate"]*100).round(1)
        st.dataframe(display,use_container_width=True,hide_index=True)

    with right:
        top10 = ranked.head(10).copy()
        fig = px.bar(
            top10,x=unit_col,y="team",orientation="h",
            labels={unit_col:"EPA / Play","team":""}
        )
        fig.update_layout(
            height=430,
            yaxis={"categoryorder":"array","categoryarray":top10["team"][::-1]},
            margin=dict(l=5,r=5,t=10,b=5)
        )
        st.plotly_chart(fig,use_container_width=True)

    st.subheader("Offense vs Defense")
    scatter = team_df.dropna(subset=["off_epa","def_epa"]).copy()
    fig = px.scatter(
        scatter,
        x="off_epa",
        y="def_epa",
        text="team",
        labels={
            "off_epa":"Offensive EPA / Play",
            "def_epa":"Defensive EPA / Play (lower is better)"
        }
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=560)
    st.plotly_chart(fig,use_container_width=True)

with player_tab:
    if player_df.empty:
        st.warning("Player data not available.")
    else:
        pos = st.radio("Position",["QB","RB","WR","TE"],horizontal=True)
        pos_df = player_df[player_df["position"]==pos].copy()

        maps = {
            "QB":[("passing_yards","Passing Yards"),("passing_tds","Passing TD"),("interceptions","INT"),("fantasy_points","Fantasy Points")],
            "RB":[("rushing_yards","Rushing Yards"),("rushing_tds","Rushing TD"),("receptions","Receptions"),("receiving_yards","Receiving Yards")],
            "WR":[("receiving_yards","Receiving Yards"),("receptions","Receptions"),("receiving_tds","Receiving TD"),("targets","Targets")],
            "TE":[("receiving_yards","Receiving Yards"),("receptions","Receptions"),("receiving_tds","Receiving TD"),("targets","Targets")],
        }

        available = [(c,n) for c,n in maps[pos] if c in pos_df.columns]
        metric_name = st.selectbox("Rank by",[n for _,n in available])
        metric_col = {n:c for c,n in available}[metric_name]

        leaders = pos_df.sort_values(metric_col,ascending=(metric_col=="interceptions")).head(25).copy()
        cols = ["player_display_name","recent_team"]+[c for c,_ in available]
        leaders = leaders[cols]
        rename = {"player_display_name":"Player","recent_team":"Team"}
        rename.update({c:n for c,n in available})
        st.dataframe(leaders.rename(columns=rename),use_container_width=True,hide_index=True)

        chart = pos_df.sort_values(metric_col,ascending=False).head(15)
        fig = px.bar(
            chart,x=metric_col,y="player_display_name",orientation="h",
            labels={metric_col:metric_name,"player_display_name":""}
        )
        fig.update_layout(
            height=520,
            yaxis={"categoryorder":"array","categoryarray":chart["player_display_name"][::-1]}
        )
        st.plotly_chart(fig,use_container_width=True)

with trends_tab:
    trend_team = st.selectbox("Trend team", sorted(team_df["team"].dropna().unique()))
    frames = []
    for yr in SEASONS:
        try:
            d = load_team_metrics(yr)
            row = d[d["team"]==trend_team].copy()
            if not row.empty:
                row["season"] = yr
                frames.append(row)
        except Exception:
            pass

    if frames:
        trend = pd.concat(frames,ignore_index=True)
        long = trend.melt(
            id_vars=["season","team"],
            value_vars=["off_epa","def_epa"],
            var_name="unit",
            value_name="epa_per_play"
        )
        long["unit"] = long["unit"].map({"off_epa":"Offense","def_epa":"Defense"})
        fig = px.line(long,x="season",y="epa_per_play",color="unit",markers=True)
        fig.update_layout(height=450)
        st.plotly_chart(fig,use_container_width=True)

with about_tab:
    st.markdown("""
### Metrics
- **EPA / Play:** expected points added per play.
- **Success Rate:** share of plays with positive EPA.
- **Pass EPA / Rush EPA:** efficiency split by play type.

### Stack
Python • pandas • Plotly • Streamlit • nflverse / nflfastR
""")
