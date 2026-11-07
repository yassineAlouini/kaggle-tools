---
name: kaggle-leaderboard-analysis
description: Compare Kaggle public and private team positions, or plot an account's competition history as top percentages by year with readable paginated figures. Use for leaderboard shakeups, finishing positions, and competition progress reports.
---

# Kaggle leaderboard analysis

Use the `kaggle_tools.leaderboard` data helpers and the `kt` plotting commands.
They perform read-only Kaggle requests and generate local artifacts. Do not
submit notebooks or change competition selections as part of this skill.

## Setup

From a kaggle-tools checkout, run commands with:

```sh
uv run --extra plots --extra kaggle kt --help
```

Outside a checkout, install `kaggle-tools[plots,kaggle]` from the repository and
use `kt`. Only the `plots` extra is needed for offline CSV input. Authenticate
using the user's existing Kaggle credentials; do not print credentials.

If this skill was linked from a checkout, resolve the skill directory's real
path to locate that checkout. Read [input formats](references/inputs.md) when
using saved data or troubleshooting live fetching.

## Public vs private positions

```sh
kt leaderboard-plot --competition COMPETITION_SLUG --output OUTPUT_DIR
# Optional: --highlight-team TEAM_ID --zoom 1000 --title "Competition title"
```

Infer the competition from the user's project or request when possible. Highlight
by team ID, never by a potentially duplicated team name. The full plot and zoom
keep the original board positions; do not rank rounded scores or renumber only
the matched teams. Report any unmatched teams from the metadata.

The API defaults to the public board when private results are unavailable. The
command stops before the deadline or when both responses are identical and private
availability cannot be established. Explain that limitation rather than drawing
a false perfect match. Explicitly identified public/private CSV exports can be
used even if their results happen to be identical.

## Competition history

```sh
kt competition-progress --output OUTPUT_DIR --include-unranked \
  --title "Competition results"
```

Live history belongs to the authenticated Kaggle account. A display title does
not select another account. Use a supplied CSV for another account's history.

- **Top percentage = 100 * rank / total teams. Lower is better.** For example,
  rank 10 of 200 is top 5%, not 95%. Use this percentage unless the user explicitly
  requests another measure.
- Keep **each competition's bar**. Group by deadline year. Show the unweighted
  mean top percentage of ended, ranked competitions above each yearly subplot;
  this heading supplements the bars rather than replacing them with yearly bars.
- Ongoing results are provisional and excluded from yearly means. Missing ranks
  are labeled empty rows when `--include-unranked` is used, never zero percentages.
- Preserve readable labels. The PDF packs multiple year subplots per page and
  adds pages as needed. The HTML includes all years and full names/ranks on hover.
  The PNG shows an overview of annual means and competition counts; link the
  PDF/HTML for the full history. Future deadline years beyond the snapshot year
  are omitted from charts while their source rows are retained.

## Deliver

Check rank bounds, matching/coverage counts, and the displayed percentages.
Inspect a dense page and any highlighted result. Return links to the current
PDF/HTML and CSV; mention the snapshot date and ongoing/missing results when relevant.
Use the recorded source/provenance JSON to distinguish live snapshots from supplied
CSV inputs. Do not present cross-year differences as a controlled measure of skill.

When refreshing a report, reuse the same output directory. If the user requests
cleanup, remove only identified superseded generated artifacts, keeping current
reports and source data. Never publish account history or generated personal
reports into a source-code PR unless the user requests that publication.
