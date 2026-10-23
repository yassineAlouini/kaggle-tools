from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kaggle_tools import leaderboard as lb


def _board(ids, ranks, names=None):
    return pd.DataFrame(
        {
            "TeamId": ids,
            "Rank": ranks,
            "TeamName": names or ["same"] * len(ids),
            "Score": ["0.950"] * len(ids),
        }
    )


def _history():
    return pd.DataFrame(
        [
            {"id": 1, "title": "Past", "deadline": "2020-01-01", "userRank": 10, "teamCount": 200},
            {"id": 2, "title": "Ongoing", "deadline": "2026-12-01", "userRank": 1, "teamCount": 20},
            {
                "id": 3,
                "title": "Unranked",
                "deadline": "2020-02-01",
                "userRank": None,
                "teamCount": 0,
            },
            {
                "id": 4,
                "title": "Future",
                "deadline": "2030-01-01",
                "userRank": None,
                "teamCount": 0,
            },
        ]
    )


def test_matching_preserves_ties_gaps_and_duplicate_names():
    data, meta = lb.match_leaderboards(_board([1, 2, 3], [1, 1, 7]), _board([3, 1, 4], [2, 5, 6]))
    assert data.team_id.tolist() == [1, 3]
    assert data.public_rank.tolist() == [1, 7]
    assert data.private_rank.tolist() == [5, 2]
    assert data.positions_gained.tolist() == [-4, 5]
    assert meta["public_only_ids"] == [2]
    assert meta["private_only_ids"] == [4]


@pytest.mark.parametrize("ranks", [[0], [-1], [1.5], [np.inf], [np.nan]])
def test_invalid_board_rank_rejected(ranks):
    with pytest.raises(ValueError, match=r"positive integers|exceed team count"):
        lb.match_leaderboards(_board([1], ranks), _board([1], [1]))


def test_duplicate_and_unmatched_board_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        lb.match_leaderboards(_board([1, 1], [1, 2]), _board([1], [1]))
    with pytest.raises(ValueError, match="No teams"):
        lb.match_leaderboards(_board([1], [1]), _board([2], [1]))


def test_history_is_direct_top_percentage_not_complement():
    data = lb.normalize_history(_history(), as_of="2026-10-23")
    past = data[data.competition == "Past"].iloc[0]
    assert past.top_percent == 5
    assert "percent_teams_behind" not in data
    assert data[data.competition == "Ongoing"].status.item() == "ongoing"
    assert data[data.competition == "Unranked"].top_percent.isna().all()
    assert len(data) == 4


@pytest.mark.parametrize(("rank", "teams"), [(201, 200), (-1, 200), (1.5, 200), (1, 0), (1, 200.5)])
def test_bad_history_bounds_rejected(rank, teams):
    frame = _history().iloc[:1].copy()
    frame["userRank"], frame["teamCount"] = rank, teams
    with pytest.raises(ValueError, match=r"positive integers|exceed team count"):
        lb.normalize_history(frame)


def test_zero_rank_is_missing_and_missing_deadline_allowed_only_if_unranked():
    frame = _history().iloc[:1].copy()
    frame["userRank"], frame["deadline"] = 0, None
    data = lb.normalize_history(frame)
    assert data.top_percent.isna().all()
    assert data.status.item() == "unknown"
    frame["userRank"] = 1
    with pytest.raises(ValueError, match="deadlines"):
        lb.normalize_history(frame)


def test_history_legacy_pagination_does_not_stop_without_token():
    pages = [_history().iloc[:2].to_dict("records"), _history().iloc[2:].to_dict("records"), []]
    calls = []

    def list_competitions(**kwargs):
        calls.append(kwargs)
        return {"competitions": pages[kwargs["page"] - 1]}

    data, meta = lb.fetch_history(api=SimpleNamespace(competitions_list=list_competitions))
    assert len(data) == 4 and meta["ranked"] == 2
    assert [c["page"] for c in calls] == [1, 2, 3]


def test_history_token_pagination():
    calls = []

    def list_competitions(**kwargs):
        calls.append(kwargs)
        if kwargs["page_token"] is None:
            return {"competitions": _history().iloc[:2].to_dict("records"), "nextPageToken": "next"}
        return {"competitions": _history().iloc[2:].to_dict("records")}

    data, _ = lb.fetch_history(api=SimpleNamespace(competitions_list=list_competitions))
    assert len(data) == 4
    assert calls[1]["page_token"] == "next" and calls[1]["page"] == -1


def test_history_repeated_page_and_empty_account():
    row = _history().iloc[:1].to_dict("records")
    with pytest.raises(ValueError, match="Repeated"):
        lb.fetch_history(api=SimpleNamespace(competitions_list=lambda **kw: {"competitions": row}))
    data, meta = lb.fetch_history(
        api=SimpleNamespace(competitions_list=lambda **kw: {"competitions": []})
    )
    assert data.empty and meta["ranked"] == 0


def test_fetch_leaderboards_all_pages_and_fallback_check(monkeypatch):
    monkeypatch.setattr(lb, "_competition", lambda *a: {"deadline": "2020-01-01", "title": "Demo"})
    calls = []

    def page(api, slug, public, token):
        calls.append((public, token))
        first, second = (1, 2) if public else (2, 1)
        return {
            "submissions": [
                {"teamId": second if token else first, "teamName": "Duplicate", "score": "0.950"}
            ],
            **({} if token else {"nextPageToken": "next"}),
        }

    monkeypatch.setattr(lb, "_board_page", page)
    data, meta = lb.fetch_leaderboards("demo", api=object(), request_interval=0)
    assert data.private_rank.tolist() == [2, 1]
    assert calls == [(True, ""), (True, "next"), (False, ""), (False, "next")]
    assert meta["matched_teams"] == 2
    monkeypatch.setattr(
        lb,
        "_board_page",
        lambda *a: {"submissions": [{"teamId": 1, "teamName": "A", "score": "0.5"}]},
    )
    with pytest.raises(ValueError, match="Cannot verify"):
        lb.fetch_leaderboards("demo", api=object(), request_interval=0)


def test_fetch_stops_before_deadline_and_rejects_repeated_team(monkeypatch):
    monkeypatch.setattr(lb, "_competition", lambda *a: {"deadline": "2099-01-01"})
    with pytest.raises(ValueError, match="deadline"):
        lb.fetch_leaderboards("demo", api=object(), request_interval=0)
    monkeypatch.setattr(lb, "_competition", lambda *a: {"deadline": "2020-01-01"})
    monkeypatch.setattr(
        lb,
        "_board_page",
        lambda *a: {"submissions": [{"teamId": 1, "score": ".5"}], "nextPageToken": "next"},
    )
    with pytest.raises(ValueError, match="Repeated team"):
        lb.fetch_leaderboards("demo", api=object(), request_interval=0)


def test_rate_limit_is_actionable_and_not_a_partial_snapshot(monkeypatch):
    response = SimpleNamespace(status_code=429, headers={"Retry-After": "60"})

    def failed(*args, **kwargs):
        error = RuntimeError("HTTP error")
        error.response = response
        raise error

    api = SimpleNamespace(competitions_list=failed)
    with pytest.raises(ValueError, match=r"Kaggle rate limit.*No partial report"):
        lb.fetch_history(api=api)
