"""
Would You Rather Live Here? — Streamlit web app
Run with: streamlit run app.py
"""

import requests
from datetime import datetime

import streamlit as st
import plotly.express as px

from would_you_rather_live_here import CITIES, CRITERIA, build_scored_df

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Would You Rather Live Here?",
    page_icon="🌍",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE WEATHER DATA  (Open-Meteo — free, no API key)
#
#  @st.cache_data(ttl=...) means Streamlit saves the result and reuses it
#  instead of calling the API every time someone moves a slider.
#  ttl=604800 = 7 days in seconds. After 7 days it fetches fresh data.
# ─────────────────────────────────────────────────────────────────────────────

# Coordinates for each city (latitude, longitude)
CITY_COORDS = {
    "Vienna":       (48.21,  16.37),
    "Copenhagen":   (55.68,  12.57),
    "Stockholm":    (59.33,  18.07),
    "Helsinki":     (60.17,  24.94),
    "Amsterdam":    (52.37,   4.90),
    "Berlin":       (52.52,  13.41),
    "Lisbon":       (38.72,  -9.14),
    "Barcelona":    (41.39,   2.17),
    "London":       (51.51,  -0.13),
    "Paris":        (48.86,   2.35),
    "Toronto":      (43.65, -79.38),
    "Vancouver":    (49.28, -123.12),
    "New York":     (40.71, -74.01),
    "Austin":       (30.27, -97.74),
    "Seattle":      (47.61, -122.33),
    "Sydney":       (-33.87, 151.21),
    "Melbourne":    (-37.81, 144.96),
    "Tokyo":        (35.68,  139.65),
    "Seoul":        (37.57,  126.98),
    "Singapore":    ( 1.35,  103.82),
    "Kuala Lumpur": ( 3.14,  101.69),
    "Bangkok":      (13.76,  100.50),
    "Dubai":        (25.20,   55.27),
    "Buenos Aires": (-34.60, -58.38),
    "Medellín":     ( 6.25,  -75.57),
    "Zurich":       (47.38,    8.54),
}


def _weather_score(monthly_means: list, lat: float) -> int:
    """
    Turn 12 monthly temperature averages into a 0–100 score.
    Rewards mild spring & autumn (13–19°C). Penalises brutal summers.
    Accounts for Southern Hemisphere cities where seasons are flipped.
    """
    southern = lat < 0

    if southern:
        spring = sum(monthly_means[8:11]) / 3    # Sep–Nov
        autumn = sum(monthly_means[2:5])  / 3    # Mar–May
        summer = sum([monthly_means[11], monthly_means[0], monthly_means[1]]) / 3
    else:
        spring = sum(monthly_means[2:5])  / 3    # Mar–May
        autumn = sum(monthly_means[8:11]) / 3    # Sep–Nov
        summer = sum(monthly_means[5:8])  / 3    # Jun–Aug

    spring_score   = max(0, 100 - abs(spring - 16) * 6)
    autumn_score   = max(0, 100 - abs(autumn - 14) * 6)
    summer_penalty = max(0, (summer - 26) * 4)

    raw = (spring_score + autumn_score) / 2 - summer_penalty
    return max(0, min(100, round(raw)))


@st.cache_data(ttl=604800, show_spinner=False)
def fetch_live_weather() -> tuple[dict, str]:
    """
    Calls Open-Meteo for 3 years of monthly temperatures per city.
    Returns (scores_dict, last_updated_string).
    Falls back gracefully — if a city fails, it keeps the hardcoded score.
    """
    scores = {}
    for city, (lat, lon) in CITY_COORDS.items():
        try:
            url = (
                "https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                "&start_date=2021-01-01&end_date=2023-12-31"
                "&monthly=temperature_2m_mean&timezone=GMT"
            )
            data = requests.get(url, timeout=10).json()

            raw_temps  = data["monthly"]["temperature_2m_mean"]
            raw_dates  = data["monthly"]["time"]              # e.g. "2021-01-01"

            # Average each calendar month across the 3 years
            buckets = [[] for _ in range(12)]
            for date, temp in zip(raw_dates, raw_temps):
                if temp is not None:
                    month_index = int(date[5:7]) - 1          # "2021-03-01" → 2
                    buckets[month_index].append(temp)

            monthly_means = [sum(b) / len(b) if b else 15.0 for b in buckets]
            scores[city] = _weather_score(monthly_means, lat)

        except Exception:
            scores[city] = None   # will fall back to hardcoded value

    updated = datetime.now().strftime("%-d %b %Y")
    return scores, updated


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — sliders update the ranking in real time
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌍 City Ranker")
    st.markdown("Move the sliders to match your priorities. The ranking updates instantly.")
    st.divider()

    weights = {}
    for key, meta in CRITERIA.items():
        weights[key] = st.slider(
            meta["label"],
            min_value=0,
            max_value=10,
            value=5,
            help=meta["desc"],
        )

    st.divider()
    st.caption("Data: Numbeo 2024 · GFSI 2024 · Open-Meteo (live)")

# ─────────────────────────────────────────────────────────────────────────────
#  FETCH LIVE WEATHER & MERGE WITH CITIES
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Fetching live climate data from Open-Meteo…"):
    live_weather, last_updated = fetch_live_weather()

# Build a cities list with weather scores replaced by live values where available
live_cities = []
for city in CITIES:
    entry = city.copy()
    live_score = live_weather.get(city["city"])
    if live_score is not None:
        entry["weather"] = live_score
    live_cities.append(entry)

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🌍 Would You Rather Live Here?")
st.markdown(
    "A data-driven city ranker. **26 cities · 10 criteria · your weights.**  \n"
    "Drag the sliders on the left — rankings update in real time."
)
st.caption(f"🌡️ Weather scores: live data from Open-Meteo · last fetched {last_updated}")
st.divider()

if all(v == 0 for v in weights.values()):
    st.warning("All weights are 0. Adjust at least one slider to see rankings.")
    st.stop()

df = build_scored_df(weights, cities=live_cities)

# ─────────────────────────────────────────────────────────────────────────────
#  TOP 3 CARDS
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("🏆 Your Top 3 Cities")

medals  = ["🥇", "🥈", "🥉"]
columns = st.columns(3)
active  = [k for k in CRITERIA if weights.get(k, 0) > 0]

for i, col in enumerate(columns):
    row = df.iloc[i]
    with col:
        with st.container(border=True):
            st.markdown(f"### {medals[i]} {row['flag']} {row['city']}")
            st.caption(row["country"])
            st.metric("Score", f"{row['score']:.1f} / 100")
            for key in active:
                val = row[f"{key}_norm"]
                st.progress(int(val), text=f"{CRITERIA[key]['label']}: {val:.0f}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  BAR CHART — all 26 cities
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📊 All 26 Cities Ranked")

chart_df = df.copy()
chart_df["City"] = chart_df["flag"] + "  " + chart_df["city"]

fig = px.bar(
    chart_df,
    x="score",
    y="City",
    orientation="h",
    color="score",
    color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
    range_color=[0, 100],
    labels={"score": "Score (0–100)", "City": ""},
    height=720,
    text=chart_df["score"].round(1),
)
fig.update_traces(textposition="outside")
fig.update_layout(
    yaxis=dict(autorange="reversed"),
    coloraxis_showscale=False,
    xaxis=dict(range=[0, 108]),
    margin=dict(l=10, r=60, t=20, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FULL BREAKDOWN TABLE
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📋 Full breakdown by criterion"):
    norm_cols   = [f"{k}_norm" for k in active]
    label_names = {f"{k}_norm": CRITERIA[k]["label"] for k in active}

    display_df = (
        df[["flag", "city", "country", "score"] + norm_cols]
        .rename(columns={"flag": "", "city": "City", "country": "Country", "score": "Score", **label_names})
        .reset_index(drop=True)
    )
    display_df.index += 1

    numeric_cols = ["Score"] + [CRITERIA[k]["label"] for k in active]

    st.dataframe(
        display_df.style
            .background_gradient(subset=numeric_cols, cmap="RdYlGn", vmin=0, vmax=100)
            .format({col: "{:.1f}" if col == "Score" else "{:.0f}" for col in numeric_cols}),
        use_container_width=True,
        height=600,
    )
