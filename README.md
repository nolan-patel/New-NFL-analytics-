# NFL Analytics Dashboard — Real Player Rankings + 2026 Projections

This version fixes the Position Rankings logic by aggregating nflverse's **weekly**
player stats into full season-level player rows before ranking them.

## Player ranking positions

- QB
- RB
- WR
- TE
- EDGE
- DT
- LB
- CB
- S

Offensive line is intentionally excluded because nflverse's standard public player
stats do not contain enough player-level blocking outcomes to build a responsible
OL ranking.

## Projection model

2026 projections use:
- 55% weight on 2025
- 30% weight on 2024
- 15% weight on 2023
- a small capped year-over-year trend adjustment

## Breakout model

Breakout candidates are based on:
- latest-season efficiency
- year-over-year improvement
- meaningful usage
- an opportunity adjustment that avoids simply returning the already-established
  highest-volume stars

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
