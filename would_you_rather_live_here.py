"""
Would You Rather Live Here?
A data-driven city ranker. Built with Python, pandas, and Rich.
"""

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.text import Text
from rich import box
from rich.rule import Rule

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  DATA
#  Sources: World Happiness Report 2023 · Numbeo 2024 · Freedom House 2024
# ─────────────────────────────────────────────────────────────────────────────

CITIES = [
    # happiness  = WHR 2023 score (0–10)
    # cost       = Numbeo cost-of-living index (NYC ≈ 100; lower = cheaper)
    # freedom    = Freedom House 2024 (0–100; higher = more free)
    # safety     = Numbeo safety index (0–100)
    # qol        = Numbeo quality-of-life composite (higher = better)
    # internet   = average broadband speed in Mbps
    {"city": "Vienna",        "country": "Austria",      "flag": "🇦🇹", "happiness": 7.3,  "cost": 71,  "freedom": 95,  "safety": 79, "qol": 196, "internet": 86},
    {"city": "Copenhagen",    "country": "Denmark",      "flag": "🇩🇰", "happiness": 7.6,  "cost": 89,  "freedom": 97,  "safety": 74, "qol": 186, "internet": 165},
    {"city": "Stockholm",     "country": "Sweden",       "flag": "🇸🇪", "happiness": 7.4,  "cost": 75,  "freedom": 97,  "safety": 72, "qol": 179, "internet": 147},
    {"city": "Helsinki",      "country": "Finland",      "flag": "🇫🇮", "happiness": 7.8,  "cost": 78,  "freedom": 100, "safety": 78, "qol": 185, "internet": 112},
    {"city": "Amsterdam",     "country": "Netherlands",  "flag": "🇳🇱", "happiness": 7.4,  "cost": 79,  "freedom": 95,  "safety": 61, "qol": 180, "internet": 135},
    {"city": "Berlin",        "country": "Germany",      "flag": "🇩🇪", "happiness": 6.9,  "cost": 64,  "freedom": 95,  "safety": 63, "qol": 169, "internet": 91},
    {"city": "Lisbon",        "country": "Portugal",     "flag": "🇵🇹", "happiness": 6.0,  "cost": 53,  "freedom": 90,  "safety": 64, "qol": 163, "internet": 126},
    {"city": "Barcelona",     "country": "Spain",        "flag": "🇪🇸", "happiness": 6.5,  "cost": 58,  "freedom": 88,  "safety": 59, "qol": 161, "internet": 158},
    {"city": "London",        "country": "UK",           "flag": "🇬🇧", "happiness": 6.8,  "cost": 91,  "freedom": 93,  "safety": 47, "qol": 155, "internet": 99},
    {"city": "Paris",         "country": "France",       "flag": "🇫🇷", "happiness": 6.7,  "cost": 79,  "freedom": 90,  "safety": 48, "qol": 152, "internet": 140},
    {"city": "Toronto",       "country": "Canada",       "flag": "🇨🇦", "happiness": 7.0,  "cost": 72,  "freedom": 97,  "safety": 64, "qol": 175, "internet": 115},
    {"city": "Vancouver",     "country": "Canada",       "flag": "🇨🇦", "happiness": 7.0,  "cost": 76,  "freedom": 97,  "safety": 59, "qol": 170, "internet": 115},
    {"city": "New York",      "country": "USA",          "flag": "🇺🇸", "happiness": 6.9,  "cost": 100, "freedom": 83,  "safety": 44, "qol": 148, "internet": 125},
    {"city": "Austin",        "country": "USA",          "flag": "🇺🇸", "happiness": 6.9,  "cost": 76,  "freedom": 83,  "safety": 52, "qol": 170, "internet": 120},
    {"city": "Seattle",       "country": "USA",          "flag": "🇺🇸", "happiness": 6.9,  "cost": 84,  "freedom": 83,  "safety": 45, "qol": 168, "internet": 120},
    {"city": "Sydney",        "country": "Australia",    "flag": "🇦🇺", "happiness": 7.1,  "cost": 78,  "freedom": 97,  "safety": 68, "qol": 179, "internet": 70},
    {"city": "Melbourne",     "country": "Australia",    "flag": "🇦🇺", "happiness": 7.1,  "cost": 74,  "freedom": 97,  "safety": 67, "qol": 177, "internet": 70},
    {"city": "Tokyo",         "country": "Japan",        "flag": "🇯🇵", "happiness": 6.1,  "cost": 79,  "freedom": 96,  "safety": 83, "qol": 175, "internet": 118},
    {"city": "Seoul",         "country": "S. Korea",     "flag": "🇰🇷", "happiness": 6.0,  "cost": 65,  "freedom": 83,  "safety": 70, "qol": 166, "internet": 241},
    {"city": "Singapore",     "country": "Singapore",    "flag": "🇸🇬", "happiness": 6.6,  "cost": 85,  "freedom": 47,  "safety": 80, "qol": 171, "internet": 247},
    {"city": "Kuala Lumpur",  "country": "Malaysia",     "flag": "🇲🇾", "happiness": 5.9,  "cost": 38,  "freedom": 43,  "safety": 54, "qol": 160, "internet": 65},
    {"city": "Bangkok",       "country": "Thailand",     "flag": "🇹🇭", "happiness": 5.9,  "cost": 40,  "freedom": 30,  "safety": 50, "qol": 148, "internet": 56},
    {"city": "Dubai",         "country": "UAE",          "flag": "🇦🇪", "happiness": 6.7,  "cost": 69,  "freedom": 18,  "safety": 82, "qol": 162, "internet": 195},
    {"city": "Buenos Aires",  "country": "Argentina",    "flag": "🇦🇷", "happiness": 5.8,  "cost": 35,  "freedom": 82,  "safety": 37, "qol": 133, "internet": 36},
    {"city": "Medellín",      "country": "Colombia",     "flag": "🇨🇴", "happiness": 5.7,  "cost": 33,  "freedom": 65,  "safety": 42, "qol": 142, "internet": 30},
    {"city": "Zurich",        "country": "Switzerland",  "flag": "🇨🇭", "happiness": 7.2,  "cost": 130, "freedom": 96,  "safety": 77, "qol": 187, "internet": 172},
]

# Each criterion: label for display, short description, and whether lower raw = better (invert).
CRITERIA = {
    "happiness": {"label": "Happiness",        "desc": "Well-being score · World Happiness Report 2023",  "invert": False},
    "cost":      {"label": "Affordability",     "desc": "Cost of living (lower is better) · Numbeo 2024",  "invert": True},
    "freedom":   {"label": "Political Freedom", "desc": "Civil liberties & rights · Freedom House 2024",   "invert": False},
    "safety":    {"label": "Safety",            "desc": "Personal safety index · Numbeo 2024",             "invert": False},
    "qol":       {"label": "Quality of Life",   "desc": "Liveability composite · Numbeo 2024",             "invert": False},
    "internet":  {"label": "Internet Speed",    "desc": "Average broadband speed in Mbps",                 "invert": False},
}


# ─────────────────────────────────────────────────────────────────────────────
#  SCORING
# ─────────────────────────────────────────────────────────────────────────────

def normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """Rescale any metric to 0–100 so all criteria are comparable."""
    lo, hi = series.min(), series.max()
    scaled = (series - lo) / (hi - lo) * 100
    return (100 - scaled) if invert else scaled


def build_scored_df(weights: dict) -> pd.DataFrame:
    """Load cities, attach normalized columns, compute weighted score, sort."""
    df = pd.DataFrame(CITIES)

    for key, meta in CRITERIA.items():
        df[f"{key}_norm"] = normalize(df[key], meta["invert"])

    total = sum(weights.values()) or 1
    df["score"] = sum(
        (weights[k] / total) * df[f"{k}_norm"]
        for k in CRITERIA
        if weights[k] > 0
    )

    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bar(value: float, width: int = 20) -> str:
    """Unicode progress bar: ████████░░░░  for 60 out of 100."""
    filled = round(value / 100 * width)
    return "█" * filled + "░" * (width - filled)


def medal(rank: int) -> str:
    return {0: "🥇", 1: "🥈", 2: "🥉"}.get(rank, f"   {rank + 1}.")


def score_color(value: float) -> str:
    return "green" if value >= 65 else "yellow" if value >= 40 else "red"


# ─────────────────────────────────────────────────────────────────────────────
#  SCREENS
# ─────────────────────────────────────────────────────────────────────────────

def show_banner():
    console.print()
    console.print(Panel(
        Text.assemble(
            ("🌍  Would You Rather Live Here?\n", "bold white"),
            ("   A data-driven ranker for 26 cities across 6 criteria.\n", "dim"),
            ("   Answer 6 questions. Get your personal city ranking.", "italic cyan"),
        ),
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


def ask_weights() -> dict:
    console.print(Panel(
        "[bold]What matters to you?[/bold]\n"
        "Rate each criterion [bold cyan]0–10[/bold cyan]  "
        "(0 = don't care  ·  10 = most important)",
        border_style="dim",
    ))
    console.print()

    weights = {}
    for key, meta in CRITERIA.items():
        console.print(f"  [bold]{meta['label']}[/bold]  [dim]{meta['desc']}[/dim]")
        weights[key] = max(0, min(10, IntPrompt.ask("  Importance", default=5)))
        console.print()

    return weights


def show_top3(df: pd.DataFrame, weights: dict):
    """Render the top 3 cities as spotlight cards."""
    active = [k for k in CRITERIA if weights.get(k, 0) > 0]
    border_colors = ["gold1", "grey70", "dark_orange"]

    for i in range(min(3, len(df))):
        row = df.iloc[i]

        lines = [
            f"  {medal(i)}  [bold]{row['flag']} {row['city']}, {row['country']}[/bold]"
            f"   Score: [bold cyan]{row['score']:.1f}[/bold cyan] / 100\n",
        ]

        for key in active:
            val = row[f"{key}_norm"]
            c   = score_color(val)
            lines.append(
                f"  {CRITERIA[key]['label']:<20} "
                f"[{c}]{bar(val, 18)}[/{c}]  {val:.0f}"
            )

        console.print(Panel("\n".join(lines), border_style=border_colors[i], padding=(0, 2)))
        console.print()


def show_full_table(df: pd.DataFrame, weights: dict):
    """Render all 26 cities in a ranked table with per-metric scores."""
    active = [k for k in CRITERIA if weights.get(k, 0) > 0]

    table = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        show_header=True,
        padding=(0, 1),
    )
    table.add_column("#",       style="dim",  width=5)
    table.add_column("City",    style="bold", min_width=14)
    table.add_column("Country", style="dim",  min_width=11)
    table.add_column("Score",   justify="right", min_width=7)
    table.add_column("",        min_width=22)
    for key in active:
        table.add_column(CRITERIA[key]["label"], justify="right", min_width=8)

    for i, row in df.iterrows():
        sc = row["score"]
        c  = score_color(sc)

        metric_cols = [
            f"[{score_color(row[f'{k}_norm'])}]{row[f'{k}_norm']:.0f}[/{score_color(row[f'{k}_norm'])}]"
            for k in active
        ]

        table.add_row(
            medal(i) if i < 3 else f"  {i + 1}.",
            f"{row['flag']} {row['city']}",
            row["country"],
            f"[{c}]{sc:.1f}[/{c}]",
            f"[{c}]{bar(sc, 20)}[/{c}]",
            *metric_cols,
        )

    console.print(table)


def show_footer():
    console.print(
        "[dim]Data: World Happiness Report 2023 · Numbeo 2024 · Freedom House 2024\n"
        "Scores are relative — normalized within this city set.[/dim]"
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    show_banner()
    weights = ask_weights()

    if all(v == 0 for v in weights.values()):
        console.print("[red]All weights are 0 — run again and rate at least one criterion.[/red]")
        return

    df = build_scored_df(weights)

    console.print(Rule("[bold cyan]Your Results[/bold cyan]"))
    console.print()
    show_top3(df, weights)

    console.print(Rule("[dim]All 26 Cities Ranked[/dim]"))
    console.print()
    show_full_table(df, weights)
    console.print()
    show_footer()


if __name__ == "__main__":
    main()
