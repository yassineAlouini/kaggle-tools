import json

import pandas as pd
import pytest

from kaggle_tools.cli import main
from kaggle_tools.leaderboard import match_leaderboards
from kaggle_tools.leaderboard_plots import plot_history, plot_leaderboards


def _history():
    return pd.DataFrame(
        {
            "competition_id": [1, 2, 3, 4, 5],
            "competition": ["Past", "Current ended", "Ongoing", "No rank", "Future"],
            "deadline": ["2020-01-01", "2026-01-01", "2026-12-01", "2026-01-02", "2030-01-01"],
            "rank": [10, 20, 1, None, None],
            "teams": [200, 200, 20, 100, 100],
        }
    )


def test_history_output_percentages_yearly_counts_and_future_filter(tmp_path):
    files = plot_history(_history(), tmp_path, as_of="2026-10-23", include_unranked=True)
    data = pd.read_csv(files["csv"])
    assert data.loc[data.competition == "Past", "top_percent"].item() == 5
    assert len(data) == 5  # future entry retained as source data
    summary = pd.read_csv(files["yearly_csv"]).set_index("year")
    assert 2030 not in summary.index
    assert summary.loc[2026, "entered"] == 3
    assert summary.loc[2026, "ranked"] == 2
    assert summary.loc[2026, "ended_ranked"] == 1
    assert summary.loc[2026, "mean_top_percent"] == 10
    meta = json.loads(files["json"].read_text())
    assert meta["future_year_ids_omitted_from_plots"] == [5]
    assert meta["pages"] >= 2
    assert files["pdf"].read_bytes().startswith(b"%PDF")
    assert not list(tmp_path.glob("competition_progress_page_*"))
    assert all(p.stat().st_size > 0 for p in files.values())


def test_leaderboard_plot_outputs_and_highlight_validation(tmp_path):
    public = pd.DataFrame({"team_id": [1, 2], "team_name": ["A", "B"], "rank": [1, 2]})
    private = pd.DataFrame({"team_id": [1, 2], "rank": [2, 1]})
    data, metadata = match_leaderboards(public, private)
    files = plot_leaderboards(data, tmp_path, highlight_team=2, metadata=metadata, zoom=1)
    assert json.loads(files["json"].read_text())["spearman_rho"] == pytest.approx(-1)
    assert all(p.stat().st_size > 0 for p in files.values())
    with pytest.raises(ValueError, match="Highlighted team"):
        plot_leaderboards(data, tmp_path, highlight_team=999)


def test_offline_cli_and_invalid_source_combinations(tmp_path, capsys):
    path = tmp_path / "history.csv"
    _history().to_csv(path, index=False)
    assert (
        main(
            [
                "competition-progress",
                "--input",
                str(path),
                "--as-of",
                "2026-10-23",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert "lower is better" in capsys.readouterr().out
    assert main(["leaderboard-plot", "--public-csv", str(path), "--output", str(tmp_path)]) == 1
    assert "both" in capsys.readouterr().err
