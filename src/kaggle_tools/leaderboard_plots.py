"""Reusable leaderboard figures; plotting dependencies are optional and lazy.

RSNA Knee (2026) motivated matching by team ID and preserving rank provenance.
History keeps individual competition bars in yearly subplots, using top percentage
(lower is better). Yearly headings summarize ended results without hiding the bars.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd

from kaggle_tools.leaderboard import _now, match_leaderboards, normalize_history


def _libraries():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Plotting requires kaggle-tools[plots].") from exc
    return plt, go


def _save(fig, interactive, data, metadata, output, stem, *, table=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=160, facecolor=fig.get_facecolor())
    charts = interactive if isinstance(interactive, list) else [interactive]
    content = "".join(
        chart.to_html(full_html=False, include_plotlyjs=(index == 0))
        for index, chart in enumerate(charts)
    )
    if table is not None:
        content += (
            "<details><summary>All entered competitions, including unranked entries</summary>"
        )
        content += table.to_html(index=False, na_rep="No reported value", escape=True)
        content += "</details>"
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(str(metadata['title']))}</title>"
        "<style>body{font:15px system-ui;margin:24px;color:#223047}"
        "td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>"
        f"<p>{html.escape(str(metadata.get('description', '')))}</p>" + content + "</html>"
    )
    (output / f"{stem}.html").write_text(page, encoding="utf-8")
    data.to_csv(output / f"{stem}.csv", index=False)
    (output / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {ext: output / f"{stem}.{ext}" for ext in ("png", "svg", "html", "csv", "json")}


def plot_leaderboards(
    data: pd.DataFrame,
    output: str | Path,
    *,
    title: str = "Public vs private positions",
    highlight_team: int | None = None,
    zoom: int = 1000,
    metadata: dict | None = None,
) -> dict[str, Path]:
    """Write PNG, SVG, standalone HTML, matched CSV and provenance JSON.

    Input is the matched table returned by fetch_leaderboards/match_leaderboards.
    Plot full positions and a zoom viewport, keeping unmatched teams in metadata.
    """
    if zoom < 1:
        raise ValueError("zoom must be positive")
    public = data.rename(columns={"public_rank": "rank", "public_score": "score"})
    private = data.rename(columns={"private_rank": "rank", "private_score": "score"})
    data, counts = match_leaderboards(public, private)
    if highlight_team is not None and highlight_team not in set(data.team_id):
        raise ValueError(f"Highlighted team {highlight_team} has no matched public/private result")
    metadata = {**counts, **(metadata or {}), "title": title, "generated_at": _now()}
    metadata["description"] = (
        "Each point is a team. Lower positions are better. Below the diagonal means improved; "
        "above means dropped. Original board positions are preserved. "
        f"Matched {len(data):,} teams; public-only {len(metadata['public_only_ids']):,}, "
        f"private-only {len(metadata['private_only_ids']):,}."
    )
    plt, go = _libraries()
    colors = {"Improved": "#148575", "Dropped": "#d66c51", "Unchanged": "#8893a2"}
    movement = np.select(
        [data.positions_gained > 0, data.positions_gained < 0],
        ["Improved", "Dropped"],
        default="Unchanged",
    )
    limit = int(data[["public_rank", "private_rank"]].max().max())
    rho = data.public_rank.corr(data.private_rank, method="spearman") if len(data) > 1 else np.nan
    rho_text = f"{rho:.3f}" if np.isfinite(rho) else "undefined"
    metadata["spearman_rho"] = float(rho) if np.isfinite(rho) else None
    own = data[data.team_id == highlight_team]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor("#f7f9fc")
    for ax, bound, panel in zip(
        axes,
        [limit, min(zoom, limit)],
        ["All matched teams", f"Zoom · first {min(zoom, limit):,} positions"],
        strict=True,
    ):
        ax.plot([1, bound], [1, bound], "--", color="#526174", lw=1, label="Same position")
        for name, color in colors.items():
            group = data[
                (movement == name) & (data.public_rank <= bound) & (data.private_rank <= bound)
            ]
            ax.scatter(
                group.public_rank,
                group.private_rank,
                s=14,
                c=color,
                alpha=0.5,
                edgecolors="none",
                rasterized=True,
                label=name,
            )
        for row in own.itertuples():
            if row.public_rank <= bound and row.private_rank <= bound:
                ax.scatter(
                    row.public_rank, row.private_rank, s=150, marker="*", c="#243e8b", zorder=5
                )
                ax.annotate(
                    f"{fill(row.team_name, 28)}\n#{row.public_rank:,} → #{row.private_rank:,}",
                    (row.public_rank, row.private_rank),
                    xytext=(0.03, 0.95),
                    textcoords="axes fraction",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d6deed"},
                    arrowprops={"arrowstyle": "-", "color": "#243e8b"},
                )
        ax.set(
            xlim=(0, bound * 1.02),
            ylim=(0, bound * 1.02),
            title=panel,
            xlabel="Public position · lower is better",
            ylabel="Private position · lower is better",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#e9edf3")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(fill(title, 80), x=0.07, y=0.97, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.07, 0.87, f"{len(data):,} matched teams · Spearman rho = {rho_text}", color="#536277"
    )
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=4, frameon=False)
    fig.text(0.07, 0.06, fill(metadata["description"], 145), fontsize=9, color="#536277")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.22, wspace=0.24)
    interactive = go.Figure()
    for name, color in colors.items():
        group = data[movement == name].copy()
        group["team_name"] = group.team_name.map(html.escape)
        interactive.add_trace(
            go.Scattergl(
                x=group.public_rank,
                y=group.private_rank,
                name=name,
                mode="markers",
                marker={"color": color, "size": 6, "opacity": 0.65},
                customdata=group[
                    ["team_name", "team_id", "positions_gained", "public_score", "private_score"]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Team %{customdata[1]}"
                    "<br>Public #%{x:,} → private #%{y:,}<br>Positions gained: %{customdata[2]:+,}"
                    "<br>Scores: %{customdata[3]} → %{customdata[4]}<extra></extra>"
                ),
            )
        )
    interactive.add_trace(
        go.Scatter(
            x=[1, limit],
            y=[1, limit],
            mode="lines",
            name="Same position",
            line={"color": "#526174", "dash": "dash"},
            hoverinfo="skip",
        )
    )
    for row in own.itertuples():
        interactive.add_trace(
            go.Scatter(
                x=[row.public_rank],
                y=[row.private_rank],
                mode="markers",
                name=html.escape(row.team_name),
                marker={"size": 17, "symbol": "star", "color": "#243e8b"},
            )
        )
    interactive.update_layout(
        template="plotly_white",
        title=html.escape(title),
        height=850,
        xaxis={"title": "Public position", "range": [0, limit * 1.02]},
        yaxis={"title": "Private position", "range": [0, limit * 1.02], "scaleanchor": "x"},
    )
    try:
        return _save(fig, interactive, data, metadata, output, "public_vs_private")
    finally:
        plt.close(fig)


def _history_overview(data, as_of, title, plt, go):
    eligible = data[data.year.notna() & (data.year <= pd.Timestamp(as_of).year)]
    years = range(int(eligible.year.min()), int(eligible.year.max()) + 1) if len(eligible) else []
    records = []
    for year in years:
        group = eligible[eligible.year == year]
        ended = group[(group.status == "ended") & group["rank"].notna()]
        records.append(
            {
                "year": year,
                "entered": len(group),
                "ranked": int(group["rank"].notna().sum()),
                "ended_ranked": len(ended),
                "mean_top_percent": float(ended.top_percent.mean()) if len(ended) else None,
            }
        )
    summary = pd.DataFrame(
        records, columns=["year", "entered", "ranked", "ended_ranked", "mean_top_percent"]
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(20, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.patch.set_facecolor("#f7f9fc")
    x = np.arange(len(summary))
    axes[0].bar(x, summary.mean_top_percent, color="#4b79b6", zorder=3)
    for i, row in enumerate(summary.to_dict("records")):
        if pd.notna(row["mean_top_percent"]):
            axes[0].text(
                i,
                row["mean_top_percent"] + 2,
                f"{row['mean_top_percent']:.1f}%",
                ha="center",
                fontsize=15,
                fontweight="bold",
            )
        else:
            axes[0].text(
                i, 3, "No ended\nranked results", ha="center", fontsize=10, color="#7d8899"
            )
    axes[0].set(ylim=(0, 110), ylabel="Mean top percentage · lower is better")
    axes[1].bar(x - 0.18, summary.entered, width=0.36, color="#b3c2d6", label="Entered")
    axes[1].bar(x + 0.18, summary.ranked, width=0.36, color="#148575", label="Ranked")
    for i, row in enumerate(summary.to_dict("records")):
        axes[1].text(i - 0.18, row["entered"] + 0.2, str(row["entered"]), ha="center", fontsize=12)
        axes[1].text(i + 0.18, row["ranked"] + 0.2, str(row["ranked"]), ha="center", fontsize=12)
    axes[1].set(ylabel="Competition count", xticks=x, xticklabels=summary.year)
    axes[1].set_ylim(0, max(summary.entered.max() * 1.25, 1) if len(summary) else 1)
    axes[1].legend(frameon=False, ncol=2, fontsize=14, loc="upper left")
    for ax in axes:
        ax.tick_params(labelsize=14)
        ax.yaxis.label.set_size(15)
        ax.grid(axis="y", color="#e1e7ef", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(fill(title, 90), x=0.06, y=0.97, ha="left", fontsize=27, fontweight="bold")
    fig.text(
        0.06,
        0.90,
        "Yearly overview · percentage = rank / teams * 100 · lower is better",
        fontsize=18,
        color="#243e8b",
    )
    fig.text(
        0.06,
        0.035,
        "Means use ended, ranked competitions with equal weight. "
        "Counts show all entered and all ranked competitions; "
        "ongoing results are excluded from means.",
        fontsize=12,
        color="#536277",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.11, hspace=0.17)
    chart = go.Figure()
    chart.add_trace(
        go.Bar(
            x=summary.year,
            y=summary.mean_top_percent,
            name="Mean top %",
            customdata=summary[["entered", "ranked", "ended_ranked"]],
            hovertemplate="%{x}<br>Mean top %{y:.1f}%<br>Entered: %{customdata[0]}"
            "<br>Ranked: %{customdata[1]}<br>Ended ranked: %{customdata[2]}<extra></extra>",
        )
    )
    chart.update_layout(
        template="plotly_white",
        title="Yearly overview · mean top percentage (lower is better)",
        height=500,
        font={"size": 16},
        xaxis={"dtick": 1},
        yaxis={"range": [0, 100], "title": "Mean top %"},
    )
    return fig, chart, summary


def plot_history(
    data: pd.DataFrame,
    output: str | Path,
    *,
    title: str = "Competition progress",
    metadata: dict | None = None,
    as_of: str | None = None,
    include_unranked: bool = False,
) -> dict[str, Path]:
    """Plot one bar per competition: top % = 100 * rank / teams, lower is better.

    Ongoing competitions are distinct. Unranked entries always remain in CSV and
    HTML tables; include_unranked also adds labeled empty rows to the figures.
    Yearly headings give the mean top percentage of ended, ranked entries.
    Metadata fetched_at supplies snapshot time
    unless as_of explicitly overrides it.
    """
    metadata = {**(metadata or {}), "title": title, "generated_at": _now()}
    as_of = as_of or metadata.get("fetched_at") or metadata["generated_at"]
    data = normalize_history(data, as_of=as_of)
    ranked = data["rank"].notna()
    shown = data.copy() if include_unranked else data[ranked].copy()
    future = shown.year.notna() & (shown.year > pd.Timestamp(as_of).year)
    metadata["future_year_ids_omitted_from_plots"] = shown.loc[future, "competition_id"].tolist()
    shown = shown[~future].copy()
    if shown.empty:
        raise ValueError(
            "No ranked competitions to plot; use include_unranked to show empty entries"
        )
    metadata.update(
        {
            "as_of": as_of,
            "entered": len(data),
            "ranked": int(ranked.sum()),
            "unranked": int((~ranked).sum()),
            "include_unranked": include_unranked,
            "formula": "100 * rank / teams; top percentage, lower is better",
            "year_basis": "Competition deadline year; not individual submission year",
            "description": (
                "One bar per competition. Top percentage = 100 * rank / teams; lower is better. "
                "Years follow competition deadlines. Orange bars are ongoing/provisional. "
                f"{int((~ranked).sum())} entries have no reported rank (not zero performance)."
            ),
        }
    )
    years = shown.year.astype("string").fillna("Undated")
    groups = [(year, shown[years == year]) for year in years.unique()]
    yearly = []
    headings = []
    for year, group in groups:
        ended = group[(group.status == "ended") & group["rank"].notna()]
        mean = float(ended.top_percent.mean()) if len(ended) else None
        yearly.append({"year": year, "ended_ranked": len(ended), "mean_top_percent": mean})
        overall = (
            f"Mean top {mean:.1f}% · {len(ended)} ended"
            if mean is not None
            else "No ended ranked results"
        )
        headings.append(f"{year} · {overall}")
    metadata["yearly_summary"] = (
        "Unweighted mean top percentage of ended, ranked competitions in each deadline year; "
        "ongoing and unranked entries excluded. Individual bars are retained."
    )
    metadata["description"] += (
        " Yearly headings show the mean top percentage of ended results only."
    )
    plt, go = _libraries()
    from matplotlib.backends.backend_pdf import PdfPages

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "competition_progress.pdf"
    overview, overview_chart, summary = _history_overview(data, as_of, title, plt, go)
    interactive = [overview_chart]
    # Pack pairs of years into rows, then fit as many rows as remain readable.
    page_rows, current, used = [], [], 0.0
    for offset in range(0, len(groups), 2):
        pair = list(range(offset, min(offset + 2, len(groups))))
        height = max(len(groups[index][1]) for index in pair) * 0.43 + 1.4
        if current and used + height > 10.5:
            page_rows.append(current)
            current, used = [], 0.0
        current.append((pair, height))
        used += height
    if current:
        page_rows.append(current)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(overview, facecolor=overview.get_facecolor())
        for page_number, layout in enumerate(page_rows, start=2):
            fig, axes = plt.subplots(
                len(layout),
                2,
                figsize=(22, max(14, sum(h for _, h in layout) + 3)),
                squeeze=False,
                gridspec_kw={"height_ratios": [h for _, h in layout]},
            )
            fig.patch.set_facecolor("#f7f9fc")
            for row_number, (panels, _) in enumerate(layout):
                for col in range(2):
                    ax = axes[row_number, col]
                    if col >= len(panels):
                        ax.set_visible(False)
                        continue
                    panel = panels[col]
                    year, group = groups[panel]
                    heading = headings[panel]
                    positions = np.arange(len(group))
                    colors = np.where(group.status == "ongoing", "#d79535", "#4b79b6")
                    labels = [fill(str(name), 38) for name in group.competition]
                    bars = ax.barh(
                        positions, group.top_percent, color=colors, height=0.62, zorder=3
                    )
                    for i, (bar, result) in enumerate(zip(bars, group.itertuples(), strict=True)):
                        if pd.isna(result.rank):
                            ax.text(
                                2, i, "No reported rank", va="center", color="#7d8899", fontsize=11
                            )
                            continue
                        if result.status == "ongoing":
                            bar.set_hatch("///")
                            bar.set_edgecolor("#805d27")
                        ax.text(
                            result.top_percent + 0.8,
                            i,
                            f"{result.top_percent:.1f}%",
                            va="center",
                            fontsize=12,
                            fontweight="bold",
                        )
                    ax.set_yticks(positions, labels)
                    ax.set_ylim(len(group) - 0.35, -0.65)
                    ax.set_xlim(0, 112)
                    ax.set_xticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
                    ax.tick_params(axis="y", length=0, labelsize=12, pad=10)
                    ax.tick_params(axis="x", length=0, labelsize=12, pad=8)
                    ax.set_xlabel("Top percentage · lower is better", fontsize=12, labelpad=12)
                    ax.set_title(heading, loc="left", fontsize=16, fontweight="bold", pad=20)
                    ax.grid(axis="x", color="#e1e7ef", zorder=0)
                    ax.axvline(10, ls="--", lw=1, color="#8898af", zorder=2)
                    ax.spines[:].set_visible(False)
            fig.suptitle(
                fill(title, 95), x=0.03, y=0.975, ha="left", fontsize=27, fontweight="bold"
            )
            fig.text(
                0.03,
                0.92,
                "Every competition is shown individually. "
                "Yearly headings summarize ended, ranked results only.",
                fontsize=15,
                color="#536277",
            )
            fig.text(
                0.03,
                0.025,
                "Top % = rank / teams * 100 · lower is better. "
                "Orange = ongoing. Empty rows = no reported rank.",
                fontsize=13,
                color="#536277",
            )
            fig.text(
                0.96, 0.025, f"Page {page_number} / {len(page_rows) + 1}", ha="right", fontsize=12
            )
            fig.subplots_adjust(
                left=0.17, right=0.97, top=0.84, bottom=0.10, wspace=1.0, hspace=0.65
            )
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
    for (_year, group), heading in zip(groups, headings, strict=True):
        positions = np.arange(len(group))
        colors = np.where(group.status == "ongoing", "#d79535", "#4b79b6")
        labels = [fill(str(name), 47) for name in group.competition]
        chart = go.Figure(
            go.Bar(
                x=group.top_percent,
                y=positions,
                orientation="h",
                marker_color=colors,
                customdata=group.assign(competition=group.competition.map(html.escape))[
                    ["competition", "rank", "teams", "status"]
                ],
                text=["" if pd.isna(x) else f"{x:.1f}%" for x in group.top_percent],
                textposition="outside",
                textfont={"size": 16},
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Top %{x:.2f}%"
                    "<br>Rank #%{customdata[1]:,.0f} / %{customdata[2]:,}"
                    "<br>%{customdata[3]}<extra></extra>"
                ),
            )
        )
        chart.update_layout(
            template="plotly_white",
            title={"text": html.escape(heading), "font": {"size": 25}},
            font={"size": 15},
            height=max(450, 240 + 55 * len(group)),
            margin={"l": 460, "r": 80, "t": 100, "b": 80},
            xaxis={"range": [0, 112], "title": "Top percentage · lower is better"},
            yaxis={
                "autorange": "reversed",
                "tickvals": positions,
                "ticktext": [html.escape(label).replace("\n", "<br>") for label in labels],
            },
        )
        for i in np.flatnonzero(group["rank"].isna().to_numpy()):
            chart.add_annotation(
                x=2,
                y=int(i),
                text="No reported rank",
                showarrow=False,
                xanchor="left",
                font={"size": 15},
            )
        interactive.append(chart)
    summary_path = output / "competition_progress_yearly.csv"
    summary.to_csv(summary_path, index=False)
    metadata["pages"] = len(page_rows) + 1
    metadata["preview_note"] = (
        "PNG/SVG show the yearly overview; PDF/HTML also contain individual bars"
    )
    result = _save(
        overview, interactive, data, metadata, output, "competition_progress", table=data
    )
    plt.close(overview)
    return {**result, "pdf": pdf_path, "yearly_csv": summary_path}
