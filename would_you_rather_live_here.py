"""
Would You Rather Live Here?
A data-driven city ranker. Built with Python, pandas, and Rich.
"""

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.rule import Rule

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  DATA
#  Sources: Numbeo 2024 · GFSI 2024 · various tech/travel indices
#
#  nature       = proximity to nature: beaches, mountains, parks (0–100)
#  connectivity = great cities reachable in a weekend (0–100)
#  tech         = tech industry & job market strength (0–100)
#  hub          = global hub: airport, international status (0–100)
#  tz_diff      = hours difference from WIB / UTC+7 (Jakarta) — lower = closer
#  safety       = Numbeo personal safety index (0–100)
#  cost         = Numbeo cost-of-living index (NYC ≈ 100; lower = cheaper)
#  weather      = spring & autumn quality; penalises extreme summers (0–100)
#  architecture = castles, Victorian, medieval, Renaissance streetscapes (0–100)
#  indonesian   = size of Indonesian diaspora, food, cultural orgs (0–100)
# ─────────────────────────────────────────────────────────────────────────────

CITIES = [
    {"city": "Vienna",        "country": "Austria",      "flag": "🇦🇹", "nature": 65, "connectivity": 95, "tech": 55, "hub": 65, "tz_diff": 6,  "safety": 79, "cost": 71,  "weather": 72, "architecture": 95, "indonesian": 28},
    {"city": "Copenhagen",    "country": "Denmark",      "flag": "🇩🇰", "nature": 55, "connectivity": 80, "tech": 60, "hub": 62, "tz_diff": 6,  "safety": 74, "cost": 89,  "weather": 62, "architecture": 78, "indonesian": 23},
    {"city": "Stockholm",     "country": "Sweden",       "flag": "🇸🇪", "nature": 85, "connectivity": 75, "tech": 78, "hub": 60, "tz_diff": 6,  "safety": 72, "cost": 75,  "weather": 63, "architecture": 75, "indonesian": 25},
    {"city": "Helsinki",      "country": "Finland",      "flag": "🇫🇮", "nature": 87, "connectivity": 60, "tech": 65, "hub": 55, "tz_diff": 5,  "safety": 78, "cost": 78,  "weather": 55, "architecture": 50, "indonesian": 18},
    {"city": "Amsterdam",     "country": "Netherlands",  "flag": "🇳🇱", "nature": 40, "connectivity": 92, "tech": 72, "hub": 85, "tz_diff": 6,  "safety": 61, "cost": 79,  "weather": 70, "architecture": 80, "indonesian": 85},
    {"city": "Berlin",        "country": "Germany",      "flag": "🇩🇪", "nature": 60, "connectivity": 88, "tech": 80, "hub": 72, "tz_diff": 6,  "safety": 63, "cost": 64,  "weather": 65, "architecture": 63, "indonesian": 38},
    {"city": "Lisbon",        "country": "Portugal",     "flag": "🇵🇹", "nature": 75, "connectivity": 68, "tech": 68, "hub": 58, "tz_diff": 7,  "safety": 64, "cost": 53,  "weather": 88, "architecture": 83, "indonesian": 22},
    {"city": "Barcelona",     "country": "Spain",        "flag": "🇪🇸", "nature": 80, "connectivity": 88, "tech": 68, "hub": 68, "tz_diff": 6,  "safety": 59, "cost": 58,  "weather": 87, "architecture": 85, "indonesian": 25},
    {"city": "London",        "country": "UK",           "flag": "🇬🇧", "nature": 45, "connectivity": 90, "tech": 88, "hub": 98, "tz_diff": 7,  "safety": 47, "cost": 91,  "weather": 68, "architecture": 88, "indonesian": 58},
    {"city": "Paris",         "country": "France",       "flag": "🇫🇷", "nature": 50, "connectivity": 93, "tech": 75, "hub": 95, "tz_diff": 6,  "safety": 48, "cost": 79,  "weather": 76, "architecture": 92, "indonesian": 40},
    {"city": "Toronto",       "country": "Canada",       "flag": "🇨🇦", "nature": 70, "connectivity": 70, "tech": 82, "hub": 80, "tz_diff": 12, "safety": 64, "cost": 72,  "weather": 63, "architecture": 52, "indonesian": 48},
    {"city": "Vancouver",     "country": "Canada",       "flag": "🇨🇦", "nature": 95, "connectivity": 62, "tech": 75, "hub": 60, "tz_diff": 15, "safety": 59, "cost": 76,  "weather": 74, "architecture": 35, "indonesian": 45},
    {"city": "New York",      "country": "USA",          "flag": "🇺🇸", "nature": 40, "connectivity": 75, "tech": 85, "hub": 100,"tz_diff": 12, "safety": 44, "cost": 100, "weather": 65, "architecture": 58, "indonesian": 50},
    {"city": "Austin",        "country": "USA",          "flag": "🇺🇸", "nature": 70, "connectivity": 48, "tech": 80, "hub": 50, "tz_diff": 13, "safety": 52, "cost": 76,  "weather": 48, "architecture": 28, "indonesian": 35},
    {"city": "Seattle",       "country": "USA",          "flag": "🇺🇸", "nature": 90, "connectivity": 58, "tech": 88, "hub": 65, "tz_diff": 15, "safety": 45, "cost": 84,  "weather": 67, "architecture": 38, "indonesian": 52},
    {"city": "Sydney",        "country": "Australia",    "flag": "🇦🇺", "nature": 88, "connectivity": 55, "tech": 72, "hub": 75, "tz_diff": 3,  "safety": 68, "cost": 78,  "weather": 80, "architecture": 65, "indonesian": 75},
    {"city": "Melbourne",     "country": "Australia",    "flag": "🇦🇺", "nature": 75, "connectivity": 52, "tech": 65, "hub": 68, "tz_diff": 3,  "safety": 67, "cost": 74,  "weather": 76, "architecture": 73, "indonesian": 68},
    {"city": "Tokyo",         "country": "Japan",        "flag": "🇯🇵", "nature": 60, "connectivity": 80, "tech": 70, "hub": 90, "tz_diff": 2,  "safety": 83, "cost": 79,  "weather": 83, "architecture": 52, "indonesian": 62},
    {"city": "Seoul",         "country": "S. Korea",     "flag": "🇰🇷", "nature": 65, "connectivity": 82, "tech": 78, "hub": 82, "tz_diff": 2,  "safety": 70, "cost": 65,  "weather": 76, "architecture": 48, "indonesian": 55},
    {"city": "Singapore",     "country": "Singapore",    "flag": "🇸🇬", "nature": 55, "connectivity": 85, "tech": 85, "hub": 92, "tz_diff": 1,  "safety": 80, "cost": 85,  "weather": 30, "architecture": 38, "indonesian": 88},
    {"city": "Kuala Lumpur",  "country": "Malaysia",     "flag": "🇲🇾", "nature": 72, "connectivity": 82, "tech": 55, "hub": 72, "tz_diff": 1,  "safety": 54, "cost": 38,  "weather": 28, "architecture": 48, "indonesian": 90},
    {"city": "Bangkok",       "country": "Thailand",     "flag": "🇹🇭", "nature": 45, "connectivity": 78, "tech": 45, "hub": 75, "tz_diff": 0,  "safety": 50, "cost": 40,  "weather": 32, "architecture": 45, "indonesian": 45},
    {"city": "Dubai",         "country": "UAE",          "flag": "🇦🇪", "nature": 35, "connectivity": 80, "tech": 65, "hub": 92, "tz_diff": 3,  "safety": 82, "cost": 69,  "weather": 42, "architecture": 18, "indonesian": 68},
    {"city": "Buenos Aires",  "country": "Argentina",    "flag": "🇦🇷", "nature": 55, "connectivity": 45, "tech": 52, "hub": 58, "tz_diff": 10, "safety": 37, "cost": 35,  "weather": 73, "architecture": 78, "indonesian": 15},
    {"city": "Medellín",      "country": "Colombia",     "flag": "🇨🇴", "nature": 82, "connectivity": 48, "tech": 45, "hub": 40, "tz_diff": 12, "safety": 42, "cost": 33,  "weather": 90, "architecture": 40, "indonesian": 12},
    {"city": "Zurich",        "country": "Switzerland",  "flag": "🇨🇭", "nature": 92, "connectivity": 93, "tech": 72, "hub": 68, "tz_diff": 6,  "safety": 77, "cost": 130, "weather": 75, "architecture": 82, "indonesian": 30},
]

# Each criterion: label for display, short description, and whether lower raw = better (invert).
CRITERIA = {
    "nature":       {"label": "Nature",           "desc": "Beaches, mountains, parks, green space nearby",        "invert": False},
    "connectivity": {"label": "Short Trips",      "desc": "Great cities reachable in a weekend",                  "invert": False},
    "tech":         {"label": "Tech Industry",    "desc": "Tech companies, startups & job market strength",       "invert": False},
    "hub":          {"label": "Global Hub",       "desc": "Airport connectivity & international city status",     "invert": False},
    "tz_diff":      {"label": "Indonesia TZ",     "desc": "Hours from Jakarta (WIB/UTC+7) — closer = better",    "invert": True},
    "safety":       {"label": "Safety",           "desc": "Personal safety index · Numbeo 2024",                  "invert": False},
    "cost":         {"label": "Affordability",    "desc": "Cost of living · Numbeo 2024 (lower = better)",        "invert": True},
    "weather":      {"label": "Weather",          "desc": "Spring & autumn quality; penalises extreme summers",    "invert": False},
    "architecture": {"label": "Architecture",     "desc": "Castles, Victorian, medieval & Renaissance streetscapes", "invert": False},
    "indonesian":   {"label": "Indonesian Community", "desc": "Diaspora size, Indonesian food, cultural organisations", "invert": False},
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
            ("   A data-driven ranker for 26 cities across 10 criteria.\n", "dim"),
            ("   Answer 10 questions. Get your personal city ranking.", "italic cyan"),
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
        try:
            raw = input("  Importance 0–10 (press Enter for 5): ").strip()
            weights[key] = max(0, min(10, int(raw) if raw else 5))
        except ValueError:
            weights[key] = 5
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
        "[dim]Data: Numbeo 2024 · GFSI 2024 · tech/travel indices · timezone data\n"
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
