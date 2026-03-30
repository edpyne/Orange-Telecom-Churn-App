"""
============================================================
  Orange Telecom Churn Prediction System
  Streamlit App  –  app.py
============================================================
  Run:  streamlit run app.py
============================================================
"""

import io
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Orange Telecom Churn Prediction",
    page_icon  = "📡",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
MODEL_PATH = "model.pkl"
TARGET_COL = "churned"
DROP_COLS  = ["phone_number"]

RISK_BANDS = {
    "Low Risk":    (0.00, 0.40, "#10b981", "🟢"),
    "Medium Risk": (0.40, 0.70, "#f59e0b", "🟡"),
    "High Risk":   (0.70, 1.00, "#ef4444", "🔴"),
}

# Human-readable feature descriptions
FEATURE_LABELS = {
    "total_charge":                    "Total charge amount",
    "total_intl_minutes":              "International call minutes",
    "total_minutes":                   "Total call minutes across all periods",
    "intl_charge_per_min":             "International charge per minute",
    "total_day_minutes":               "Daytime call minutes",
    "night_charge_per_min":            "Night-time charge per minute",
    "charge_per_call":                 "Average charge per call",
    "eve_charge_per_min":              "Evening charge per minute",
    "total_eve_minutes":               "Evening call minutes",
    "day_charge_per_min":              "Daytime charge per minute",
    "state":                           "Customer's state / region",
    "total_eve_calls":                 "Number of evening calls",
    "total_night_calls":               "Number of night-time calls",
    "service_call_rate":               "Customer service contact rate",
    "total_night_minutes":             "Night-time call minutes",
    "total_calls":                     "Total calls made",
    "total_day_calls":                 "Number of daytime calls",
    "account_length":                  "Account tenure (days)",
    "number_customer_service_calls":   "Customer service call count",
    "total_intl_calls":                "Number of international calls",
    "intl_plan":                       "International plan subscription",
    "voice_mail_plan":                 "Voicemail plan subscription",
    "total_day_charge":                "Total daytime charges",
    "total_eve_charge":                "Total evening charges",
    "total_night_charge":              "Total night-time charges",
    "total_intl_charge":               "Total international charges",
    "charge_per_day":                  "Average charge per account day",
    "calls_per_day":                   "Average calls per account day",
    "number_vmail_messages":           "Number of voicemail messages",
    "area_code":                       "Customer area code",
}

CHURN_REASONS = {
    "total_charge":                   ("💸 High total bill",            "Customer's overall charges are significantly elevated"),
    "number_customer_service_calls":  ("📞 Frequent support contacts",  "Repeated service calls indicate dissatisfaction or unresolved issues"),
    "service_call_rate":              ("📞 High support contact rate",  "Customer contacts support unusually often relative to their tenure"),
    "total_day_minutes":              ("☀️  Heavy daytime usage",        "Very high daytime usage may correlate with bill shock"),
    "intl_plan":                      ("🌍 International plan flag",     "International plan subscribers have higher churn risk if ROI feels low"),
    "total_intl_minutes":             ("🌐 High international usage",    "Elevated international usage drives up costs and churn risk"),
    "charge_per_call":                ("💰 High cost per call",          "Customer pays above-average per call — perceived poor value"),
    "voice_mail_plan":                ("📬 Voicemail plan flag",         "Low engagement with voicemail plan can signal low product stickiness"),
    "account_length":                 ("📅 Short account tenure",        "Newer customers churn more; loyalty programmes could help"),
    "total_eve_minutes":              ("🌆 High evening usage",          "Sustained high usage across periods suggests cost concerns"),
    "total_night_minutes":            ("🌙 High night usage",            "Night-time heavy usage contributes to overall cost load"),
    "total_intl_charge":              ("💳 High international charges",  "International charges are disproportionately high"),
    "charge_per_day":                 ("📆 High daily charge",           "Daily billing rate is elevated compared to peers"),
}

RECOMMENDATIONS = {
    "High Risk":   [
        "🎁  Offer a personalised retention discount or loyalty credit",
        "📋  Proactively reach out via account manager / dedicated support",
        "📦  Propose a value-bundle upgrade addressing usage patterns",
        "🔔  Trigger automated win-back campaign within 48 hours",
    ],
    "Medium Risk": [
        "📧  Send a targeted satisfaction survey and usage summary",
        "💡  Highlight underutilised features that match their usage",
        "🤝  Offer a mid-cycle account review call",
        "🎫  Provide a one-time loyalty reward or service credit",
    ],
    "Low Risk":    [
        "✅  Customer appears satisfied — maintain quality of service",
        "🌟  Enrol in a loyalty / referral programme to deepen engagement",
        "📊  Monitor usage trends monthly for early churn signals",
    ],
}

# ─────────────────────────────────────────────────────────
# THEME CSS  (dark / light toggle via sidebar)
# ─────────────────────────────────────────────────────────
DARK_CSS = """
<style>
:root { --bg: #0f172a; --card: #1e293b; --border: #334155;
        --text: #f1f5f9; --muted: #94a3b8; --accent: #6366f1; }
.stApp { background-color: var(--bg); color: var(--text); }
.metric-card { background: var(--card); border: 1px solid var(--border);
               border-radius: 12px; padding: 18px 22px; margin-bottom: 12px; }
.section-header { color: var(--accent); font-size: 1.15rem;
                  font-weight: 700; margin: 18px 0 6px; }
.risk-badge { display:inline-block; padding:6px 18px; border-radius:20px;
              font-weight:700; font-size:1.1rem; margin-top:6px; }
.insight-row { background: var(--card); border-left: 4px solid var(--accent);
               border-radius:8px; padding:10px 14px; margin:6px 0; }
div[data-testid="stSidebar"] { background: var(--card); }
</style>
"""

LIGHT_CSS = """
<style>
:root { --bg: #f8fafc; --card: #ffffff; --border: #e2e8f0;
        --text: #1e293b; --muted: #64748b; --accent: #4f46e5; }
.stApp { background-color: var(--bg); color: var(--text); }
.metric-card { background: var(--card); border: 1px solid var(--border);
               border-radius: 12px; padding: 18px 22px; margin-bottom: 12px;
               box-shadow: 0 1px 4px rgba(0,0,0,.07); }
.section-header { color: var(--accent); font-size: 1.15rem;
                  font-weight: 700; margin: 18px 0 6px; }
.risk-badge { display:inline-block; padding:6px 18px; border-radius:20px;
              font-weight:700; font-size:1.1rem; margin-top:6px; }
.insight-row { background: var(--card); border-left: 4px solid var(--accent);
               border-radius:8px; padding:10px 14px; margin:6px 0; }
div[data-testid="stSidebar"] { background: var(--card);
                                border-right: 1px solid var(--border); }
</style>
"""

# ─────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ─────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (must mirror train_model.py)
# ─────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_charge_per_min"]   = df["total_day_charge"]   / (df["total_day_minutes"]   + 1e-6)
    df["eve_charge_per_min"]   = df["total_eve_charge"]   / (df["total_eve_minutes"]   + 1e-6)
    df["night_charge_per_min"] = df["total_night_charge"] / (df["total_night_minutes"] + 1e-6)
    df["intl_charge_per_min"]  = df["total_intl_charge"]  / (df["total_intl_minutes"]  + 1e-6)
    df["total_minutes"]  = (df["total_day_minutes"] + df["total_eve_minutes"] +
                            df["total_night_minutes"] + df["total_intl_minutes"])
    df["total_calls"]    = (df["total_day_calls"] + df["total_eve_calls"] +
                            df["total_night_calls"] + df["total_intl_calls"])
    df["total_charge"]   = (df["total_day_charge"] + df["total_eve_charge"] +
                            df["total_night_charge"] + df["total_intl_charge"])
    df["charge_per_call"] = df["total_charge"] / (df["total_calls"] + 1e-6)
    df["charge_per_day"]  = df["total_charge"] / (df["account_length"] + 1e-6)
    df["calls_per_day"]   = df["total_calls"]  / (df["account_length"] + 1e-6)
    df["service_call_rate"] = df["number_customer_service_calls"] / (df["account_length"] + 1e-6)
    return df

# ─────────────────────────────────────────────────────────
# RANDOM CUSTOMER GENERATOR
# ─────────────────────────────────────────────────────────
def generate_random_customer(stats: dict, seed: int = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    row = {}
    base_cols = [
        "state", "account_length", "area_code", "intl_plan",
        "voice_mail_plan", "number_vmail_messages",
        "total_day_minutes", "total_day_calls", "total_day_charge",
        "total_eve_minutes", "total_eve_calls", "total_eve_charge",
        "total_night_minutes", "total_night_calls", "total_night_charge",
        "total_intl_minutes", "total_intl_calls", "total_intl_charge",
        "number_customer_service_calls",
    ]
    for col in base_cols:
        if col not in stats:
            continue
        s = stats[col]
        if s["type"] == "numeric":
            val = float(np.clip(
                rng.normal(s["mean"], s["std"]),
                s["min"], s["max"]
            ))
            if col in ["account_length", "total_day_calls", "total_eve_calls",
                       "total_night_calls", "total_intl_calls",
                       "number_customer_service_calls", "number_vmail_messages",
                       "area_code"]:
                val = int(round(val))
            row[col] = val
        else:
            choices = list(s["freq"].keys())
            probs   = list(s["freq"].values())
            row[col] = rng.choice(choices, p=np.array(probs) / sum(probs))
    return pd.DataFrame([row])

# ─────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────
def predict(pipeline, df_input: pd.DataFrame):
    df_feat = engineer_features(df_input)
    prob    = pipeline.predict_proba(df_feat)[0][1]
    return prob, df_feat

# ─────────────────────────────────────────────────────────
# SHAP EXPLANATION
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_shap_values(_pipeline, df_feat_json: str, num_cols: list, cat_cols: list):
    """Compute SHAP values.  df_feat is passed as JSON to allow caching."""
    df_feat = pd.read_json(io.StringIO(df_feat_json))
    preprocessor = _pipeline.named_steps["preprocessor"]
    lgbm_model   = _pipeline.named_steps["clf"]

    X_transformed = preprocessor.transform(df_feat)
    explainer     = shap.TreeExplainer(lgbm_model)
    shap_vals     = explainer.shap_values(X_transformed)

    feature_names = num_cols + cat_cols
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]   # binary class 1
    else:
        sv = shap_vals[0]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap":    sv,
        "abs":     np.abs(sv),
    }).sort_values("abs", ascending=False)
    return shap_df

# ─────────────────────────────────────────────────────────
# GAUGE CHART
# ─────────────────────────────────────────────────────────
def gauge_chart(prob: float, dark: bool) -> go.Figure:
    pct = prob * 100
    if pct < 40:
        color = "#10b981"
    elif pct < 70:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    bg = "#1e293b" if dark else "#ffffff"
    tx = "#f1f5f9" if dark else "#1e293b"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = pct,
        number = {"suffix": "%", "font": {"size": 44, "color": tx}},
        delta  = {"reference": 40, "increasing": {"color": "#ef4444"},
                  "decreasing": {"color": "#10b981"}},
        gauge  = {
            "axis":  {"range": [0, 100], "tickwidth": 1,
                      "tickcolor": tx, "tickfont": {"color": tx}},
            "bar":   {"color": color, "thickness": 0.25},
            "bgcolor": bg,
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#d1fae5" if not dark else "#064e3b"},
                {"range": [40, 70], "color": "#fef3c7" if not dark else "#78350f"},
                {"range": [70,100], "color": "#fee2e2" if not dark else "#7f1d1d"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": pct,
            },
        },
        title  = {"text": "Churn Probability", "font": {"size": 16, "color": tx}},
    ))
    fig.update_layout(
        height      = 300,
        margin      = dict(t=40, b=20, l=20, r=20),
        paper_bgcolor = bg,
        plot_bgcolor  = bg,
    )
    return fig

# ─────────────────────────────────────────────────────────
# SHAP BAR CHART
# ─────────────────────────────────────────────────────────
def shap_bar_chart(shap_df: pd.DataFrame, dark: bool, top_n: int = 10) -> go.Figure:
    top    = shap_df.head(top_n).copy()
    top    = top.sort_values("shap")
    labels = [FEATURE_LABELS.get(f, f) for f in top["feature"]]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in top["shap"]]

    bg = "#1e293b" if dark else "#ffffff"
    tx = "#f1f5f9" if dark else "#1e293b"

    fig = go.Figure(go.Bar(
        x           = top["shap"],
        y           = labels,
        orientation = "h",
        marker_color= colors,
        text        = [f"{v:+.3f}" for v in top["shap"]],
        textposition= "outside",
        textfont    = {"color": tx},
    ))
    fig.update_layout(
        title       = dict(text="Top SHAP Feature Contributions", font=dict(color=tx, size=15)),
        xaxis_title = "SHAP Value (impact on churn probability)",
        xaxis       = dict(color=tx, gridcolor="#334155" if dark else "#e2e8f0"),
        yaxis       = dict(color=tx),
        height      = 380,
        margin      = dict(t=50, b=30, l=20, r=80),
        paper_bgcolor = bg,
        plot_bgcolor  = bg,
    )
    return fig

# ─────────────────────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────────────────────
def batch_predict(pipeline, df_upload: pd.DataFrame) -> pd.DataFrame:
    df_clean = df_upload.drop(columns=[c for c in DROP_COLS if c in df_upload.columns],
                              errors="ignore")
    if TARGET_COL in df_clean.columns:
        df_clean = df_clean.drop(columns=[TARGET_COL])
    df_feat  = engineer_features(df_clean)
    probs    = pipeline.predict_proba(df_feat)[:, 1]
    df_out   = df_upload.copy()
    df_out["churn_probability_%"] = (probs * 100).round(2)
    df_out["risk_level"] = pd.cut(
        probs, bins=[0, 0.4, 0.7, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True
    )
    return df_out

# ─────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────
def main():
    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Orange_logo.svg/320px-Orange_logo.svg.png",
                 width=140)
        st.markdown("## ⚙️ Settings")
        dark_mode = st.toggle("🌙 Dark Mode", value=True)
        st.markdown("---")
        st.markdown("### 📊 About the Model")
        artefacts = load_model()
        st.metric("Model AUC", f"{artefacts['auc']:.4f}")
        st.metric("Algorithm", "LightGBM")
        st.metric("Training Samples", "5,000")
        st.markdown("---")
        st.markdown("### 📂 Batch Prediction")
        uploaded_file = st.file_uploader(
            "Upload CSV for batch prediction",
            type=["csv"],
            help="Upload a CSV file with the same columns as the training data"
        )
        st.markdown("---")
        st.caption("Built by Senior ML Engineer | Orange Telecom 2024")

    # ── Apply theme ───────────────────────────────────────
    st.markdown(DARK_CSS if dark_mode else LIGHT_CSS, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────
    st.markdown("""
    <h1 style='text-align:center; font-size:2.2rem; font-weight:800; margin-bottom:4px;'>
        📡 Orange Telecom Churn Prediction System
    </h1>
    <p style='text-align:center; color:#94a3b8; margin-bottom:28px;'>
        AI-powered customer churn intelligence &nbsp;|&nbsp; LightGBM + SHAP Explainability
    </p>
    """, unsafe_allow_html=True)

    # ── Load model ────────────────────────────────────────
    artefacts  = load_model()
    pipeline   = artefacts["pipeline"]
    num_cols   = artefacts["num_cols"]
    cat_cols   = artefacts["cat_cols"]
    stats      = artefacts["data_stats"]

    # ══════════════════════════════════════════════════════
    # BATCH PREDICTION TAB
    # ══════════════════════════════════════════════════════
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("## 📂 Batch Prediction Results")
        df_up = pd.read_csv(uploaded_file)
        st.write(f"Uploaded **{len(df_up):,} rows**")
        try:
            results = batch_predict(pipeline, df_up)
            # Summary stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", f"{len(results):,}")
            col2.metric("High Risk",   f"{(results['risk_level']=='High Risk').sum():,}")
            col3.metric("Medium Risk", f"{(results['risk_level']=='Medium Risk').sum():,}")

            # Distribution chart
            risk_counts = results["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            color_map = {"Low Risk": "#10b981", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"}
            fig_dist = px.bar(risk_counts, x="Risk Level", y="Count",
                              color="Risk Level", color_discrete_map=color_map,
                              title="Risk Level Distribution")
            fig_dist.update_layout(
                paper_bgcolor="#1e293b" if dark_mode else "#ffffff",
                plot_bgcolor ="#1e293b" if dark_mode else "#ffffff",
                font_color   ="#f1f5f9" if dark_mode else "#1e293b",
                showlegend   = False,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Results table
            st.dataframe(results[["churn_probability_%", "risk_level"]
                                   + [c for c in results.columns
                                      if c not in ["churn_probability_%","risk_level"]]
                                  ].head(200),
                         use_container_width=True)

            # Download button
            csv_bytes = results.to_csv(index=False).encode()
            st.download_button(
                "⬇️  Download Full Results CSV",
                data     = csv_bytes,
                file_name= "churn_predictions.csv",
                mime     = "text/csv",
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
        st.markdown("---")

    # ══════════════════════════════════════════════════════
    # SINGLE CUSTOMER SECTION
    # ══════════════════════════════════════════════════════
    st.markdown("## 🎲 Generate Random Customer")

    # Session state for customer data
    if "customer_df" not in st.session_state:
        st.session_state.customer_df = None

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🎲 Generate Random Customer", type="primary", use_container_width=True):
            seed = np.random.randint(0, 99999)
            st.session_state.customer_df = generate_random_customer(stats, seed=seed)
            st.session_state.seed = seed

    if st.session_state.customer_df is not None:
        df_cust = st.session_state.customer_df

        # ── Display customer profile ──────────────────────
        st.markdown('<p class="section-header">👤 Customer Profile</p>',
                    unsafe_allow_html=True)
        profile_cols = st.columns(4)
        display_fields = [
            ("State",          df_cust["state"].iloc[0]),
            ("Account Length", f"{df_cust['account_length'].iloc[0]} days"),
            ("Area Code",      df_cust["area_code"].iloc[0]),
            ("Intl Plan",      df_cust["intl_plan"].iloc[0]),
            ("Voicemail Plan", df_cust["voice_mail_plan"].iloc[0]),
            ("Day Minutes",    f"{df_cust['total_day_minutes'].iloc[0]:.1f} min"),
            ("Eve Minutes",    f"{df_cust['total_eve_minutes'].iloc[0]:.1f} min"),
            ("Night Minutes",  f"{df_cust['total_night_minutes'].iloc[0]:.1f} min"),
            ("Intl Minutes",   f"{df_cust['total_intl_minutes'].iloc[0]:.1f} min"),
            ("Day Calls",      df_cust["total_day_calls"].iloc[0]),
            ("Intl Calls",     df_cust["total_intl_calls"].iloc[0]),
            ("Service Calls",  df_cust["number_customer_service_calls"].iloc[0]),
        ]
        for i, (label, val) in enumerate(display_fields):
            profile_cols[i % 4].metric(label, val)

        # ── Predict ───────────────────────────────────────
        prob, df_feat = predict(pipeline, df_cust)

        # Determine risk band
        risk_label, risk_color, risk_emoji = "Low Risk", "#10b981", "🟢"
        for band, (lo, hi, color, emoji) in RISK_BANDS.items():
            if lo <= prob < hi or (band == "High Risk" and prob >= 0.70):
                risk_label, risk_color, risk_emoji = band, color, emoji
                break

        st.markdown("---")
        st.markdown("## 📈 Prediction Output")

        left, right = st.columns([1, 1])

        with left:
            # Gauge
            fig_gauge = gauge_chart(prob, dark_mode)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk badge
            st.markdown(
                f'<div style="text-align:center">'
                f'<span class="risk-badge" style="background:{risk_color}22; '
                f'color:{risk_color}; border:2px solid {risk_color};">'
                f'{risk_emoji} {risk_label}'
                f'</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="text-align:center; color:#94a3b8; margin-top:8px;">'
                f'Churn Probability: <strong>{prob*100:.1f}%</strong></p>',
                unsafe_allow_html=True,
            )

            # Progress bar with color zones
            st.markdown('<p class="section-header">Risk Meter</p>', unsafe_allow_html=True)
            st.progress(min(prob, 1.0))
            c1, c2, c3 = st.columns(3)
            c1.markdown('<small style="color:#10b981">● Low (0–40%)</small>', unsafe_allow_html=True)
            c2.markdown('<small style="color:#f59e0b">● Medium (40–70%)</small>', unsafe_allow_html=True)
            c3.markdown('<small style="color:#ef4444">● High (70–100%)</small>', unsafe_allow_html=True)

        with right:
            # SHAP
            with st.spinner("Computing SHAP explanations…"):
                try:
                    shap_df = get_shap_values(
                        pipeline,
                        df_feat.to_json(),
                        num_cols, cat_cols,
                    )
                    fig_shap = shap_bar_chart(shap_df, dark_mode)
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP computation skipped: {e}")
                    shap_df = None

        # ── Explainability Panel ──────────────────────────
        st.markdown("---")
        st.markdown("## 🔍 Churn Insights & Explainability")

        insight_left, insight_right = st.columns([1.1, 0.9])

        with insight_left:
            st.markdown('<p class="section-header">🧠 Why This Customer Might Churn</p>',
                        unsafe_allow_html=True)

            if shap_df is not None:
                top_factors = shap_df[shap_df["shap"] > 0].head(5)
                shown = 0
                for _, row_s in top_factors.iterrows():
                    feat = row_s["feature"]
                    if feat in CHURN_REASONS:
                        icon_label, detail = CHURN_REASONS[feat]
                        st.markdown(
                            f'<div class="insight-row">'
                            f'<strong>{icon_label}</strong><br>'
                            f'<small style="color:#94a3b8">{detail}</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        shown += 1
                if shown == 0:
                    st.info("No strong single churn driver detected — risk is distributed across multiple factors.")
            else:
                # Fallback: rule-based
                df_eng = engineer_features(df_cust)
                reasons = []
                if df_eng["number_customer_service_calls"].iloc[0] >= 4:
                    reasons.append(("📞 Frequent support contacts",
                                    "Customer has called support 4+ times — strong churn predictor"))
                if df_eng["total_charge"].iloc[0] > 70:
                    reasons.append(("💸 High total bill",
                                    "Total charges are significantly above average"))
                if df_cust["intl_plan"].iloc[0] == "yes" and df_eng["total_intl_charge"].iloc[0] > 4:
                    reasons.append(("🌍 Costly international usage",
                                    "International charges are elevated under the current plan"))
                if df_cust["account_length"].iloc[0] < 50:
                    reasons.append(("📅 Short account tenure",
                                    "Newer customers are more likely to churn"))
                for icon_label, detail in reasons:
                    st.markdown(
                        f'<div class="insight-row"><strong>{icon_label}</strong><br>'
                        f'<small style="color:#94a3b8">{detail}</small></div>',
                        unsafe_allow_html=True,
                    )
                if not reasons:
                    st.info("No dominant churn driver detected.")

        with insight_right:
            st.markdown('<p class="section-header">💼 Recommended Actions</p>',
                        unsafe_allow_html=True)
            for rec in RECOMMENDATIONS.get(risk_label, []):
                st.markdown(
                    f'<div class="insight-row">{rec}</div>',
                    unsafe_allow_html=True,
                )

            # Computed usage metrics summary
            st.markdown('<p class="section-header">📊 Usage Summary</p>',
                        unsafe_allow_html=True)
            df_eng = engineer_features(df_cust)
            usage_metrics = {
                "Total Monthly Charge": f"${df_eng['total_charge'].iloc[0]:.2f}",
                "Avg. Charge/Call":     f"${df_eng['charge_per_call'].iloc[0]:.3f}",
                "Total Minutes":        f"{df_eng['total_minutes'].iloc[0]:.0f} min",
                "Service Call Rate":    f"{df_eng['service_call_rate'].iloc[0]:.4f}",
            }
            for k, v in usage_metrics.items():
                col_a, col_b = st.columns([2, 1])
                col_a.write(k)
                col_b.write(f"**{v}**")

        # ── Feature importance table ──────────────────────
        if shap_df is not None:
            with st.expander("📋 Full SHAP Feature Importance Table"):
                display_shap = shap_df.copy()
                display_shap["Feature Description"] = display_shap["feature"].map(
                    lambda f: FEATURE_LABELS.get(f, f)
                )
                display_shap["Direction"] = display_shap["shap"].apply(
                    lambda v: "↑ Increases churn risk" if v > 0 else "↓ Reduces churn risk"
                )
                st.dataframe(
                    display_shap[["Feature Description", "shap", "Direction"]
                                 ].rename(columns={"shap": "SHAP Value"}),
                    use_container_width=True,
                )

    # ── Footer ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#64748b; font-size:0.82rem;">'
        'Orange Telecom Churn Prediction System &nbsp;|&nbsp; '
        'LightGBM · SHAP · Streamlit &nbsp;|&nbsp; '
        f'Model AUC: <strong>{load_model()["auc"]:.4f}</strong>'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
