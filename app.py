
from pathlib import Path
import html

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Aurum | Gold Price Intelligence",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset" / "gold_prices_daily.csv"
MODEL = BASE / "models" / "gold_price_model.pkl"
COMPARISON = BASE / "models" / "model_comparison.csv"

GOLD = "#D4AF37"
LIGHT_GOLD = "#F5D77A"
BG = "#090B0E"
PANEL = "#15191F"

# ==========================================
# 2. GOLD AND BLACK THEME
# ==========================================

st.markdown("""
<style>
:root {
    color-scheme: dark;
}

.stApp {
    background: #090B0E;
    color: #F4F0E8;
}

[data-testid="stSidebar"] {
    background: #111419;
    border-right: 1px solid #383024;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #F5D77A !important;
}

[data-testid="stMetric"] {
    background: #15191F;
    border: 1px solid #47391D;
    border-radius: 14px;
    padding: 20px;
}

[data-testid="stMetricLabel"] {
    color: #C8C0AF;
}

[data-testid="stMetricValue"] {
    color: #F5D77A;
}

[data-testid="stMetricDelta"] {
    font-weight: 600;
}

.stButton > button,
.stDownloadButton > button {
    background: #D4AF37;
    color: #101010;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #F5D77A;
    color: #101010;
}

.hero {
    padding: 32px;
    background: linear-gradient(
        120deg, #1E1A11, #101318 65%, #15191F
    );
    border: 1px solid #5C4821;
    border-radius: 20px;
    margin-bottom: 28px;
}

.hero-title {
    font-size: clamp(2rem, 4vw, 3.4rem);
    font-weight: 800;
    color: #F5D77A;
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #C9C0AD;
    font-size: 1.05rem;
}

.section-label {
    color: #D4AF37;
    font-weight: 700;
    letter-spacing: 2px;
    font-size: 0.8rem;
    margin-bottom: 10px;
}

.info-panel {
    background: #15191F;
    border: 1px solid #3C3324;
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
}

.small-note {
    color: #AFA99D;
    font-size: 0.85rem;
}

hr {
    border-color: #383024;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    df = pd.read_csv(DATA)

    required = ["Date", "Gold_24K_1g"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df["Gold_24K_1g"] = pd.to_numeric(
        df["Gold_24K_1g"], errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Gold_24K_1g"]
    )

    df = df[df["Gold_24K_1g"] > 0]

    df = df.sort_values("Date")
    df = df.drop_duplicates(
        subset="Date", keep="last"
    )

    return df.reset_index(drop=True)


@st.cache_resource
def load_model():
    if not MODEL.exists():
        return None

    return joblib.load(MODEL)


@st.cache_data
def load_comparison():
    if not COMPARISON.exists():
        return None

    return pd.read_csv(COMPARISON)


def rupees(value):
    return f"₹{value:,.2f}"


def make_chart(df, title, show_average=True):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Gold_24K_1g"],
        mode="lines",
        name="24K Gold Price",
        line=dict(
            color=GOLD,
            width=3
        ),
        fill="tozeroy",
        fillcolor="rgba(212,175,55,0.08)",
        hovertemplate=(
            "%{x|%d %b %Y}"
            "<br>₹%{y:,.2f}/g"
            "<extra></extra>"
        )
    ))

    if show_average and len(df) >= 7:
        avg = df["Gold_24K_1g"].rolling(
            7, min_periods=7
        ).mean()

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=avg,
            mode="lines",
            name="7-observation Average",
            line=dict(
                color="#FFFFFF",
                width=2,
                dash="dot"
            )
        ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color="#EDE7DA"),
        height=420,
        margin=dict(
            l=25, r=25, t=65, b=30
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="#292D33"
        ),
        yaxis=dict(
            title="Gold Price (INR/gram)",
            showgrid=True,
            gridcolor="#292D33",
            tickprefix="₹"
        ),
        legend=dict(
            orientation="h",
            y=1.12
        ),
        hovermode="x unified"
    )

    return fig


def make_comparison_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["Model"],
        y=df["MAE"],
        marker_color=[
            GOLD if i == df["MAE"].idxmin()
            else "#666A72"
            for i in df.index
        ],
        text=df["MAE"].round(2),
        textposition="outside"
    ))

    fig.update_layout(
        title="Model Error Comparison (MAE)",
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        height=360,
        xaxis_title="Model",
        yaxis_title="MAE (₹/gram)"
    )

    return fig


# ==========================================
# 4. LOAD PROJECT FILES
# ==========================================

try:
    data = load_data()
except Exception as error:
    st.error(f"Unable to load gold dataset: {error}")
    st.info(
        "Check dataset/gold_prices_daily.csv "
        "and its Date and Gold_24K_1g columns."
    )
    st.stop()

if data.empty:
    st.error("The gold price dataset has no valid records.")
    st.stop()

try:
    model_artifact = load_model()
    model_error = None
except Exception as error:
    model_artifact = None
    model_error = str(error)

try:
    comparison = load_comparison()
except Exception as error:
    comparison = None
    st.warning(f"Model comparison could not load: {error}")


# ==========================================
# 5. PRICE AND PREDICTION
# ==========================================

latest = data.iloc[-1]
latest_price = float(latest["Gold_24K_1g"])
latest_date = latest["Date"]

previous_price = (
    float(data.iloc[-2]["Gold_24K_1g"])
    if len(data) > 1 else latest_price
)

change = latest_price - previous_price
change_pct = (
    change / previous_price * 100
    if previous_price else 0
)

# Only run the saved ML model when its feature
# requirements are explicitly known.
# Otherwise display a clearly labeled baseline.

prediction = latest_price
prediction_source = "Last-price baseline"
prediction_note = (
    "The baseline assumes the next observed "
    "price equals the latest recorded price."
)

# Supported automatic inference:
# sklearn models trained using exactly one
# previous-price feature, or named lag features.

candidate = model_artifact

if isinstance(model_artifact, dict):
    candidate = model_artifact.get(
        "model",
        model_artifact.get("estimator")
    )

if candidate is not None and hasattr(
    candidate, "predict"
):
    feature_names = getattr(
        candidate, "feature_names_in_", None
    )

    if feature_names is not None:
        feature_names = list(feature_names)

        known_features = {
            "Gold_24K_1g": latest_price,
            "lag_1": latest_price,
            "lag_2": (
                float(data.iloc[-2]["Gold_24K_1g"])
                if len(data) >= 2 else np.nan
            ),
            "lag_3": (
                float(data.iloc[-3]["Gold_24K_1g"])
                if len(data) >= 3 else np.nan
            ),
            "lag_7": (
                float(data.iloc[-7]["Gold_24K_1g"])
                if len(data) >= 7 else np.nan
            )
        }

        if all(
            name in known_features
            for name in feature_names
        ):
            features = pd.DataFrame([{
                name: known_features[name]
                for name in feature_names
            }])

            if features.notna().all().all():
                try:
                    result = float(
                        candidate.predict(features)[0]
                    )

                    if np.isfinite(result) and result > 0:
                        prediction = result
                        prediction_source = "ML model"
                        prediction_note = (
                            "Prediction from the saved "
                            "model using its named lag features."
                        )
                except Exception as error:
                    prediction_note = (
                        "Model inference unavailable: "
                        + str(error)
                    )

prediction_change = prediction - latest_price

# ==========================================
# 6. SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("## 🪙 AURUM")
    st.caption("GOLD PRICE INTELLIGENCE")

    st.divider()

    st.markdown("### Market Information")
    st.write("**Market:** India")
    st.write("**Purity:** 24 Karat")
    st.write("**Currency:** INR")
    st.write("**Unit:** Per Gram")

    st.divider()

    st.markdown("### Dataset")
    st.write(f"**Records:** {len(data):,}")
    st.write(
        "**Start:** "
        + data["Date"].min().strftime("%d %b %Y")
    )
    st.write(
        "**Latest:** "
        + latest_date.strftime("%d %b %Y")
    )

    st.divider()

    st.markdown("### Model Status")

    if model_artifact is not None:
        st.success("Saved model loaded")
    elif model_error:
        st.warning("Model could not be loaded")
    else:
        st.warning("Model file not found")

    st.caption(f"Forecast method: {prediction_source}")

    st.divider()

    st.caption(
        "Educational machine-learning project. "
        "Not financial advice."
    )


# ==========================================
# 7. HERO HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="section-label">
        PREMIUM MARKET ANALYTICS
    </div>
    <div class="hero-title">
        🪙 AURUM
    </div>
    <div class="hero-subtitle">
        Gold Price Intelligence Dashboard
        <br>
        Indian 24K Gold • INR per Gram •
        Historical Analysis & Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 8. PRICE OVERVIEW
# ==========================================

st.markdown("## 📊 Gold Market Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Latest 24K Gold Price",
        rupees(latest_price) + "/g"
    )

with c2:
    st.metric(
        "Previous Observation",
        rupees(previous_price) + "/g"
    )

with c3:
    st.metric(
        "Latest Recorded Change",
        rupees(change),
        f"{change_pct:+.2f}%"
    )

with c4:
    st.metric(
        "Next Observation Estimate",
        rupees(prediction) + "/g",
        f"{prediction_change:+.2f} INR",
        delta_color="off"
    )

st.caption(
    f"Latest recorded date: "
    f"{latest_date.strftime('%d %B %Y')} | "
    f"Estimate method: {prediction_source}"
)

st.info(
    "The estimate refers to the next recorded "
    "observation, not necessarily the next calendar day. "
    + prediction_note
)

st.divider()


# ==========================================
# 9. HISTORICAL PRICE CHART
# ==========================================

st.markdown("## 📈 Historical Gold Price Trends")

period = st.selectbox(
    "Select Historical Period",
    [
        "7 Days",
        "30 Days",
        "90 Days",
        "1 Year",
        "5 Years",
        "All History"
    ],
    index=3
)

days_map = {
    "7 Days": 7,
    "30 Days": 30,
    "90 Days": 90,
    "1 Year": 365,
    "5 Years": 1825
}

if period == "All History":
    filtered = data.copy()
else:
    cutoff = latest_date - pd.Timedelta(
        days=days_map[period]
    )
    filtered = data[
        data["Date"] >= cutoff
    ].copy()

st.plotly_chart(
    make_chart(
        filtered,
        f"Indian 24K Gold Price — {period}"
    ),
    width="stretch"
)

st.caption(
    "The chart uses recorded observations only. "
    "Missing dates are not filled with invented prices."
)

st.divider()


# ==========================================
# 10. MARKET STATISTICS
# ==========================================

st.markdown("## 📉 Market Statistics")

prices = filtered["Gold_24K_1g"]

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric(
        "Highest Price",
        rupees(prices.max())
    )

with s2:
    st.metric(
        "Lowest Price",
        rupees(prices.min())
    )

with s3:
    st.metric(
        "Average Price",
        rupees(prices.mean())
    )

with s4:
    st.metric(
        "Price Volatility (Std. Dev.)",
        rupees(prices.std())
        if len(prices) > 1 else "N/A"
    )

st.divider()


# ==========================================
# 11. RECENT PRICE MOVEMENT
# ==========================================

st.markdown("## 🗓️ Recent Price Movement")

recent = data.tail(30).copy()

recent["Change"] = (
    recent["Gold_24K_1g"].diff()
)

recent["Change_Percent"] = (
    recent["Gold_24K_1g"].pct_change() * 100
)

recent_display = recent[
    ["Date", "Gold_24K_1g",
     "Change", "Change_Percent"]
].copy()

recent_display.columns = [
    "Date",
    "Gold Price (₹/g)",
    "Change (₹)",
    "Change (%)"
]

st.dataframe(
    recent_display.sort_values(
        "Date", ascending=False
    ),
    width="stretch",
    hide_index=True,
    column_config={
        "Date": st.column_config.DateColumn(
            "Date",
            format="DD MMM YYYY"
        ),
        "Gold Price (₹/g)":
            st.column_config.NumberColumn(
                "Gold Price (₹/g)",
                format="₹%.2f"
            ),
        "Change (₹)":
            st.column_config.NumberColumn(
                "Change (₹)",
                format="₹%.2f"
            ),
        "Change (%)":
            st.column_config.NumberColumn(
                "Change (%)",
                format="%.2f%%"
            )
    }
)

st.divider()


# ==========================================
# 12. MACHINE LEARNING MODEL
# ==========================================

st.markdown("## 🤖 Machine Learning Analysis")

st.markdown("""
<div class="info-panel">
    <b style="color:#F5D77A;">
        Gold Price Prediction Engine
    </b>
    <p>
        The project uses historical Indian gold prices
        to evaluate price prediction models.
        Model accuracy should be compared with a
        simple last-price baseline.
    </p>
</div>
""", unsafe_allow_html=True)

if comparison is not None and not comparison.empty:

    st.markdown("### Model Performance")

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True
    )

    # Flexible column-name detection
    names = {
        str(col).lower().replace(
            " ", "_"
        ): col
        for col in comparison.columns
    }

    model_col = names.get(
        "model", names.get("model_name")
    )

    mae_col = names.get("mae")

    if model_col is not None and mae_col is not None:
        chart_df = comparison[
            [model_col, mae_col]
        ].copy()

        chart_df.columns = ["Model", "MAE"]

        chart_df["MAE"] = pd.to_numeric(
            chart_df["MAE"], errors="coerce"
        )

        chart_df = chart_df.dropna(
            subset=["MAE"]
        ).reset_index(drop=True)

        if not chart_df.empty:
            st.plotly_chart(
                make_comparison_chart(chart_df),
                width="stretch"
            )
else:
    st.warning(
        "Model comparison data is unavailable. "
        "Check models/model_comparison.csv."
    )

st.markdown("### Prediction Summary")

p1, p2 = st.columns(2)

with p1:
    st.metric(
        "Latest Observed Price",
        rupees(latest_price) + "/g"
    )

with p2:
    st.metric(
        "Estimated Next Observation",
        rupees(prediction) + "/g"
    )

st.caption(
    f"Prediction method: {prediction_source}. "
    "The next observation date is not guaranteed."
)

if prediction_source != "ML model":
    st.warning(
        "The saved model's feature structure has not "
        "been verified for inference. The displayed "
        "estimate is a last-price baseline, not an "
        "ML-generated forecast."
    )

st.divider()


# ==========================================
# 13. DOWNLOAD DATA
# ==========================================

st.markdown("## 📥 Download Historical Data")

st.write(
    "Download the prepared Indian 24K gold "
    "price dataset for analysis."
)

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Gold Price Dataset",
    data=csv,
    file_name="indian_24k_gold_prices.csv",
    mime="text/csv"
)

st.divider()


# ==========================================
# 14. PROJECT INFORMATION
# ==========================================

st.markdown("## ℹ️ About This Project")

st.markdown("""
<div class="info-panel">
    <h3>Gold Price Prediction Using Machine Learning</h3>
    <p>
        Aurum is a data science dashboard designed
        to analyze historical Indian gold prices
        and present predictive insights.
    </p>
    <p>
        <b>Programming Language:</b> Python<br>
        <b>Dashboard:</b> Streamlit<br>
        <b>Data Processing:</b> Pandas, NumPy<br>
        <b>Machine Learning:</b> Scikit-learn<br>
        <b>Visualization:</b> Plotly<br>
        <b>Currency:</b> Indian Rupees<br>
        <b>Gold Purity:</b> 24 Karat<br>
        <b>Measurement:</b> INR per Gram
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        text-align:center;
        color:#9C927F;
        padding:25px;
        border-top:1px solid #383024;
    ">
        AURUM | GOLD PRICE INTELLIGENCE
        <br>
        Built with Python, Streamlit and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
