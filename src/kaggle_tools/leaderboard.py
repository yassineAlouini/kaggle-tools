"""Leaderboard data, with lessons from RSNA Knee (2026).

Kaggle's download contained only the public board while the default API exposed
private results. Never reconstruct ranks from rounded scores, mistake the default
API's public fallback for private results, or stop history pagination at page one.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dict(response: Any) -> dict:
    return response if isinstance(response, dict) else json.loads(str(response))


def _api() -> Any:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError("Live data requires kaggle-tools[kaggle].") from exc
    api = KaggleApi()
    api.authenticate()
    return api


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        response = getattr(exc, "response", None)
        if response is not None and response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "the server cooldown")
            raise ValueError(
                f"Kaggle rate limit reached (HTTP 429). Retry after {retry_after}; "
                "or use saved CSVs. No partial report was written."
            ) from exc
        raise


def _positive_ints(values: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(values, errors="raise")
    if not (np.isfinite(values) & (values >= 1) & (values % 1 == 0)).all():
        raise ValueError(f"{name} must contain finite, positive integers")
    return values.astype("int64")


def _board(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(
        columns={
            "TeamId": "team_id",
            "teamId": "team_id",
            "TeamName": "team_name",
            "teamName": "team_name",
            "Rank": "rank",
            "Score": "score",
        }
    ).copy()
    required = {"team_id", "rank"}
    if not frame.columns.is_unique or not required <= set(frame.columns):
        raise ValueError("Leaderboard needs distinct columns including team_id and rank")
    for col in required:
        frame[col] = _positive_ints(frame[col], col)
    if frame.team_id.duplicated().any():
        raise ValueError("Leaderboard contains duplicate team IDs")
    if "team_name" not in frame:
        frame["team_name"] = frame.team_id.astype(str)
    frame["team_name"] = frame.team_name.fillna(frame.team_id.astype(str)).astype(str)
    if "score" not in frame:
        frame["score"] = np.nan
    return frame[["team_id", "team_name", "rank", "score"]]


def match_leaderboards(public: pd.DataFrame, private: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join on team ID, preserving official ranks, including ties and gaps.

    Accept normalized columns or Kaggle CSV headers (TeamId, TeamName, Rank,
    Score). Team names and scores are optional. Unmatched IDs are recorded in
    metadata rather than assigned fabricated positions.
    """
    public, private = _board(public), _board(private)
    result = (
        public.rename(columns={"rank": "public_rank", "score": "public_score"})
        .merge(
            private.drop(columns="team_name").rename(
                columns={"rank": "private_rank", "score": "private_score"}
            ),
            on="team_id",
            validate="one_to_one",
        )
        .sort_values(["public_rank", "team_id"])
        .reset_index(drop=True)
    )
    if result.empty:
        raise ValueError("No teams have both a public and private position")
    result["positions_gained"] = result.public_rank - result.private_rank
    metadata = {
        "matched_teams": len(result),
        "public_teams": len(public),
        "private_teams": len(private),
        "public_only_ids": sorted(set(public.team_id) - set(private.team_id)),
        "private_only_ids": sorted(set(private.team_id) - set(public.team_id)),
        "rank_method": "Original positions preserved; never ranked from rounded scores",
    }
    return result, metadata


def _competition(api: Any, slug: str) -> dict:
    from kagglesdk.competitions.types.competition_api_service import ApiGetCompetitionRequest

    request = ApiGetCompetitionRequest()
    request.competition_name = slug
    with api.build_kaggle_client() as client:
        return _dict(_call(client.competitions.competition_api_client.get_competition, request))


def _board_page(api: Any, slug: str, public: bool, token: str) -> dict:
    from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

    request = ApiGetLeaderboardRequest()
    request.competition_name = slug
    request.override_public = public
    request.page_size = 200
    request.page_token = token
    with api.build_kaggle_client() as client:
        return _dict(_call(client.competitions.competition_api_client.get_leaderboard, request))


def _fetch_board(api: Any, slug: str, public: bool, request_interval: float) -> pd.DataFrame:
    rows, tokens, ids = [], set(), set()
    token = ""
    while True:
        page = _board_page(api, slug, public, token)
        batch = page.get("submissions", [])
        for row in batch:
            if row["teamId"] in ids:
                raise ValueError("Repeated team across leaderboard pages; retry a fresh snapshot")
            ids.add(row["teamId"])
        rows.extend(batch)
        token = page.get("nextPageToken", "")
        if not token:
            break
        if token in tokens or not batch:
            raise ValueError("Leaderboard pagination did not advance")
        tokens.add(token)
        time.sleep(request_interval)
    frame = pd.DataFrame(rows, columns=["teamId", "teamName", "score"])
    frame["rank"] = np.arange(1, len(frame) + 1)
    return _board(frame)


def fetch_leaderboards(
    competition: str, *, api: Any = None, request_interval: float = 1.25
) -> tuple[pd.DataFrame, dict]:
    """Fetch both full boards and match them. Uses standard Kaggle authentication.

    Fail before the deadline or when the default and forced-public boards are
    identical: in that case the API cannot establish that private results exist.
    Explicit public/private CSVs remain usable even when the boards are identical.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", competition):
        raise ValueError("Use a competition slug, not a URL or path")
    if request_interval < 0 or not np.isfinite(request_interval):
        raise ValueError("request_interval must be finite and nonnegative")
    api = _api() if api is None else api
    fetched_at = _now()
    info = _competition(api, competition)
    deadline = pd.to_datetime(str(info.get("deadline") or "NaT"), utc=True)
    if pd.isna(deadline) or deadline > pd.Timestamp(fetched_at):
        raise ValueError("Private results cannot be verified before a known competition deadline")
    time.sleep(request_interval)
    public = _fetch_board(api, competition, True, request_interval)
    time.sleep(request_interval)
    private = _fetch_board(api, competition, False, request_interval)

    # Numeric score equality also handles 0.500 vs 0.5. Non-numeric metrics retain text.
    def signature(frame):
        scores = pd.to_numeric(frame.score, errors="coerce")
        return list(zip(frame.team_id, scores.fillna(frame.score.astype(str)), strict=True))

    if signature(public) == signature(private):
        raise ValueError(
            "Cannot verify a private leaderboard: default and public results are identical. "
            "Private results may not be released. Supply explicitly labeled CSV exports."
        )
    result, metadata = match_leaderboards(public, private)
    metadata.update(
        {
            "competition": competition,
            "title": info.get("title", competition),
            "fetched_at": fetched_at,
            "source_url": f"https://www.kaggle.com/competitions/{competition}/leaderboard",
            "rank_method": "1-based order of all paginated API rows, not rounded-score ranking",
            "private_verification": "After deadline, default differs from forced public results",
            "note": "Snapshot; results may change after review or scoring reruns",
        }
    )
    return result, metadata


def normalize_history(frame: pd.DataFrame, *, as_of: str | None = None) -> pd.DataFrame:
    """Compute percentages from account history or canonical CSV columns.

    Required: competition_id, competition, deadline, rank, teams. Also accepts
    Kaggle's id/title/deadline/userRank/teamCount fields. Missing/zero ranks are
    retained as unranked, not treated as zero performance. Ranked rows need a
    deadline; year is the deadline year, not the user's submission year.
    """
    frame = frame.rename(
        columns={
            "id": "competition_id",
            "title": "competition",
            "userRank": "rank",
            "teamCount": "teams",
        }
    ).copy()
    required = {"competition_id", "competition", "deadline", "rank", "teams"}
    if not frame.columns.is_unique or not required <= set(frame.columns):
        raise ValueError(f"History requires distinct columns including {sorted(required)}")
    frame["competition_id"] = _positive_ints(frame.competition_id, "competition_id")
    if frame.competition_id.duplicated().any():
        raise ValueError("History contains duplicate competition IDs")
    if frame.competition.isna().any():
        raise ValueError("History requires competition names")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").replace(0, np.nan)
    frame["teams"] = pd.to_numeric(frame.teams, errors="raise")
    ranked = frame["rank"].notna()
    ranks = _positive_ints(frame.loc[ranked, "rank"], "rank")
    teams = _positive_ints(frame.loc[ranked, "teams"], "teams")
    if (ranks > teams).any():
        raise ValueError("Rank cannot exceed team count")
    dates = pd.to_datetime(frame.deadline, utc=True, errors="raise")
    if dates[ranked].isna().any():
        raise ValueError("Ranked competitions require deadlines for chronological plots")
    frame["deadline"] = dates
    frame["year"] = dates.dt.year.astype("Int64")
    frame["status"] = np.where(
        dates > pd.to_datetime(as_of or _now(), utc=True), "ongoing", "ended"
    )
    frame.loc[dates.isna(), "status"] = "unknown"
    frame = frame.drop(columns="percent_teams_behind", errors="ignore")
    frame["top_percent"] = np.nan
    frame.loc[ranked, "top_percent"] = 100 * ranks / teams
    return frame.sort_values(["deadline", "competition"]).reset_index(drop=True)


def fetch_history(*, api: Any = None) -> tuple[pd.DataFrame, dict]:
    """Fetch every entered competition for the authenticated account, not an arbitrary user.

    Legacy page-number responses can omit nextPageToken even when more pages
    exist. Continue until an empty page; use tokens when the server supplies them.
    """
    api = _api() if api is None else api
    fetched_at = _now()
    rows, seen, tokens = [], set(), set()
    page, token = 1, None
    while True:
        response = _dict(
            _call(
                api.competitions_list,
                group="entered",
                category="all",
                sort_by="earliestDeadline",
                page=page if token is None else -1,
                page_size=100,
                page_token=token,
            )
        )
        batch = response.get("competitions", [])
        if not batch:
            if response.get("nextPageToken"):
                raise ValueError("History pagination returned an empty page with a continuation")
            break
        for row in batch:
            if row["id"] in seen:
                raise ValueError("Repeated competition across pages; retry a fresh snapshot")
            seen.add(row["id"])
        rows.extend(batch)
        next_token = response.get("nextPageToken")
        if next_token:
            if next_token in tokens:
                raise ValueError("History pagination did not advance")
            tokens.add(next_token)
            token = next_token
        elif token is not None:
            break
        else:
            page += 1
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["id", "title", "deadline", "userRank", "teamCount"])
    # The SDK may omit unset optional fields on serialization.
    if "userRank" not in frame:
        frame["userRank"] = np.nan
    result = normalize_history(frame, as_of=fetched_at)
    metadata = {
        "fetched_at": fetched_at,
        "account": "Authenticated Kaggle account",
        "source": "Kaggle ListCompetitions(group=entered, category=all); all pages",
        "entered": len(result),
        "ranked": int(result["rank"].notna().sum()),
        "formula": "100 * rank / teams; top percentage, lower is better",
        "year_basis": "Competition deadline year, not individual submission year",
        "rank_basis": "Kaggle userRank and teamCount; ongoing results are provisional",
    }
    return result, metadata
