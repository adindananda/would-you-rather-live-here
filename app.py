"""
Would You Rather Live Here? — Streamlit web app
Run with: streamlit run app.py
"""

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
    st.caption("Data: World Happiness Report 2023 · Numbeo 2024 · Freedom House 2024")

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🌍 Would You Rather Live Here?")
st.markdown(
    "A data-driven city ranker. **26 cities · 6 criteria · your weights.**  \n"
    "Drag the sliders on the left — rankings update in real time."
)
st.divider()

if all(v == 0 for v in weights.values()):
    st.warning("All weights are 0. Adjust at least one slider to see rankings.")
    st.stop()

df = build_scored_df(weights)

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
    display_df.index += 1  # rank starts at 1

    numeric_cols = ["Score"] + [CRITERIA[k]["label"] for k in active]

    st.dataframe(
        display_df.style
            .background_gradient(subset=numeric_cols, cmap="RdYlGn", vmin=0, vmax=100)
            .format({col: "{:.1f}" if col == "Score" else "{:.0f}" for col in numeric_cols}),
        use_container_width=True,
        height=600,
    )
