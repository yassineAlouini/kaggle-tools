# Inputs and reproducibility

## Leaderboards

```sh
kt leaderboard-plot --public-csv public.csv --private-csv private.csv --output plots
```

Each CSV needs `team_id,rank`; `team_name,score` are optional. Native Kaggle
`TeamId,TeamName,Rank,Score` and API-style `teamId,teamName,rank,score` headers are
accepted. IDs must be unique positive integers. Ranks must be positive integers;
tied ranks and gaps are preserved. Never infer ranks by sorting rounded scores.
Use the official rank column when available. Live API snapshots instead preserve
1-based order across all pages, and their JSON records that distinction.

The output CSV contains `team_id,team_name,public_rank,public_score,private_rank,
private_score,positions_gained`. Positive positions_gained means improvement.
The JSON contains coverage counts and public/private-only team IDs. Re-render a
saved matched snapshot directly with Python:

```python
import json
import pandas as pd
from kaggle_tools.leaderboard_plots import plot_leaderboards

plot_leaderboards(
    pd.read_csv("public_vs_private.csv"),
    "plots",
    metadata=json.load(open("public_vs_private.json")),
)
```

## Account history

```sh
kt competition-progress --input history.csv --as-of 2026-10-23T00:00:00Z \
  --include-unranked --output plots --title "Competition results"
```

Required columns: `competition_id,competition,deadline,rank,teams`.
Kaggle aliases `id,title,deadline,userRank,teamCount` are also accepted.
Optional columns (e.g. `url,category`) are retained. Competition IDs are unique;
rank must be integral and between 1 and teams. Empty or zero ranks are unranked.
A ranked competition needs a deadline. Unranked entries may have no date and
appear in an Undated group. `--as-of` is an offline snapshot date, not a request
for historical Kaggle standings. Without it, status is evaluated as of now.

Output `top_percent` is computed directly as `100 * rank / teams`.
The CSV retains every entered competition, including unranked entries. Output
`competition_progress_yearly.csv` records the mean top percentage and count of
ended ranked competitions per displayed year, along with entered and ranked counts.
The PDF packs multiple year subplots onto each page as space allows. The main PNG/SVG
shows the yearly overview and competition counts. HTML embeds Plotly and requires
no network access to display. Future deadline years beyond the snapshot year are
omitted from plots but retained in the source CSV.

## Live API caveats

The live commands use the installed Kaggle SDK and standard authentication.
Leaderboard pages use continuation tokens. The history API may return legacy
20-item pages without a continuation token despite accepting page_size=100;
continue by page number until an empty page. When tokens are returned, consume
them until exhausted. Both paths reject repeated IDs/tokens rather than silently
omitting or duplicating results.

The default leaderboard endpoint serves private results when available, otherwise
public. After the deadline, different default/forced-public responses establish
that two distinct boards are exposed. Identical responses cannot prove private
availability, so live comparison stops. Supply separately verified CSVs instead.

Useful upstream references:
- [Kaggle CLI and SDK usage](https://github.com/Kaggle/kaggle-cli)
- [Kaggle authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)

Live leaderboard pagination is paced to reduce rate limiting. On HTTP 429 the command
stops with a retry message and does not emit a partial report. Use the saved-data path
to regenerate figures during the server cooldown.
