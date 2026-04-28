"""
============================================================
  Orange Telecom Churn Prediction System
  Streamlit App  –  app.py
============================================================
  Works locally AND on Streamlit Cloud.
  If model.pkl is absent the model is trained automatically
  from the CSV that lives in the same directory.

  Run:  streamlit run app.py
============================================================
"""

import io
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Orange Telecom Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "model.pkl")
DATA_PATH  = os.path.join(_HERE, "Orange_Telecom_Churn_Data.csv")
TARGET_COL = "churned"
DROP_COLS  = ["phone_number"]
RANDOM_SEED = 42

# Columns the batch CSV must contain (raw, before feature engineering)
REQUIRED_BATCH_COLS = [
    "state", "account_length", "area_code", "intl_plan", "voice_mail_plan",
    "number_vmail_messages",
    "total_day_minutes",   "total_day_calls",   "total_day_charge",
    "total_eve_minutes",   "total_eve_calls",   "total_eve_charge",
    "total_night_minutes", "total_night_calls", "total_night_charge",
    "total_intl_minutes",  "total_intl_calls",  "total_intl_charge",
    "number_customer_service_calls",
]

RISK_BANDS = {
    "Low Risk":    (0.00, 0.40, "#10b981", "🟢"),
    "Medium Risk": (0.40, 0.70, "#f59e0b", "🟡"),
    "High Risk":   (0.70, 1.00, "#ef4444", "🔴"),
}

FEATURE_LABELS = {
    "total_charge":                  "Total charge amount",
    "total_intl_minutes":            "International call minutes",
    "total_minutes":                 "Total call minutes across all periods",
    "intl_charge_per_min":           "International charge per minute",
    "total_day_minutes":             "Daytime call minutes",
    "night_charge_per_min":          "Night-time charge per minute",
    "charge_per_call":               "Average charge per call",
    "eve_charge_per_min":            "Evening charge per minute",
    "total_eve_minutes":             "Evening call minutes",
    "day_charge_per_min":            "Daytime charge per minute",
    "state":                         "Customer's state / region",
    "total_eve_calls":               "Number of evening calls",
    "total_night_calls":             "Number of night-time calls",
    "service_call_rate":             "Customer service contact rate",
    "total_night_minutes":           "Night-time call minutes",
    "total_calls":                   "Total calls made",
    "total_day_calls":               "Number of daytime calls",
    "account_length":                "Account tenure (days)",
    "number_customer_service_calls": "Customer service call count",
    "total_intl_calls":              "Number of international calls",
    "intl_plan":                     "International plan subscription",
    "voice_mail_plan":               "Voicemail plan subscription",
    "total_day_charge":              "Total daytime charges",
    "total_eve_charge":              "Total evening charges",
    "total_night_charge":            "Total night-time charges",
    "total_intl_charge":             "Total international charges",
    "charge_per_day":                "Average charge per account day",
    "calls_per_day":                 "Average calls per account day",
    "number_vmail_messages":         "Number of voicemail messages",
    "area_code":                     "Customer area code",
}

CHURN_REASONS = {
    "total_charge":                  ("💸 High total bill",
                                      "Customer's overall charges are significantly elevated"),
    "number_customer_service_calls": ("📞 Frequent support contacts",
                                      "Repeated service calls indicate dissatisfaction or unresolved issues"),
    "service_call_rate":             ("📞 High support contact rate",
                                      "Customer contacts support unusually often relative to tenure"),
    "total_day_minutes":             ("☀️  Heavy daytime usage",
                                      "Very high daytime usage may correlate with bill shock"),
    "intl_plan":                     ("🌍 International plan flag",
                                      "International plan subscribers churn more if perceived ROI is low"),
    "total_intl_minutes":            ("🌐 High international usage",
                                      "Elevated international usage drives up costs and churn risk"),
    "charge_per_call":               ("💰 High cost per call",
                                      "Customer pays above-average per call — perceived poor value"),
    "voice_mail_plan":               ("📬 Voicemail plan flag",
                                      "Low engagement with voicemail plan signals low product stickiness"),
    "account_length":                ("📅 Short account tenure",
                                      "Newer customers churn more; loyalty programmes could help"),
    "total_eve_minutes":             ("🌆 High evening usage",
                                      "Sustained high usage across periods suggests cost concerns"),
    "total_night_minutes":           ("🌙 High night usage",
                                      "Night-time heavy usage contributes to overall cost load"),
    "total_intl_charge":             ("💳 High international charges",
                                      "International charges are disproportionately high"),
    "charge_per_day":                ("📆 High daily charge",
                                      "Daily billing rate is elevated compared to peers"),
}

RECOMMENDATIONS = {
    "High Risk": [
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
    "Low Risk": [
        "✅  Customer appears satisfied — maintain quality of service",
        "🌟  Enrol in a loyalty / referral programme to deepen engagement",
        "📊  Monitor usage trends monthly for early churn signals",
    ],
}

# ─────────────────────────────────────────────────────────
# CSS THEMES
# ─────────────────────────────────────────────────────────
DARK_CSS = """
<style>
:root{--bg:#0f172a;--card:#1e293b;--border:#334155;
      --text:#f1f5f9;--muted:#94a3b8;--accent:#6366f1}
.stApp{background-color:var(--bg);color:var(--text)}
.metric-card{background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:18px 22px;margin-bottom:12px}
.section-header{color:var(--accent);font-size:1.15rem;
  font-weight:700;margin:18px 0 6px}
.risk-badge{display:inline-block;padding:6px 18px;border-radius:20px;
  font-weight:700;font-size:1.1rem;margin-top:6px}
.insight-row{background:var(--card);border-left:4px solid var(--accent);
  border-radius:8px;padding:10px 14px;margin:6px 0}
div[data-testid="stSidebar"]{background:var(--card)}
</style>
"""

LIGHT_CSS = """
<style>
:root{--bg:#f8fafc;--card:#ffffff;--border:#e2e8f0;
      --text:#1e293b;--muted:#64748b;--accent:#4f46e5}
.stApp{background-color:var(--bg);color:var(--text)}
.metric-card{background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:18px 22px;margin-bottom:12px;
  box-shadow:0 1px 4px rgba(0,0,0,.07)}
.section-header{color:var(--accent);font-size:1.15rem;
  font-weight:700;margin:18px 0 6px}
.risk-badge{display:inline-block;padding:6px 18px;border-radius:20px;
  font-weight:700;font-size:1.1rem;margin-top:6px}
.insight-row{background:var(--card);border-left:4px solid var(--accent);
  border-radius:8px;padding:10px 14px;margin:6px 0}
div[data-testid="stSidebar"]{background:var(--card);
  border-right:1px solid var(--border)}
</style>
"""

# ─────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (shared by training + inference)
# ─────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_charge_per_min"]   = df["total_day_charge"]   / (df["total_day_minutes"]   + 1e-6)
    df["eve_charge_per_min"]   = df["total_eve_charge"]   / (df["total_eve_minutes"]   + 1e-6)
    df["night_charge_per_min"] = df["total_night_charge"] / (df["total_night_minutes"] + 1e-6)
    df["intl_charge_per_min"]  = df["total_intl_charge"]  / (df["total_intl_minutes"]  + 1e-6)
    df["total_minutes"] = (df["total_day_minutes"] + df["total_eve_minutes"] +
                           df["total_night_minutes"] + df["total_intl_minutes"])
    df["total_calls"]   = (df["total_day_calls"] + df["total_eve_calls"] +
                           df["total_night_calls"] + df["total_intl_calls"])
    df["total_charge"]  = (df["total_day_charge"] + df["total_eve_charge"] +
                           df["total_night_charge"] + df["total_intl_charge"])
    df["charge_per_call"]   = df["total_charge"] / (df["total_calls"]    + 1e-6)
    df["charge_per_day"]    = df["total_charge"] / (df["account_length"] + 1e-6)
    df["calls_per_day"]     = df["total_calls"]  / (df["account_length"] + 1e-6)
    df["service_call_rate"] = (df["number_customer_service_calls"] /
                               (df["account_length"] + 1e-6))
    return df


# ─────────────────────────────────────────────────────────
# TRAINING  (runs automatically if model.pkl is missing)
# ─────────────────────────────────────────────────────────
def _compute_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in df.columns:
        if col == TARGET_COL:
            continue
        s = df[col]
        if s.dtype in [np.float64, np.int64, np.float32, np.int32]:
            stats[col] = {
                "type": "numeric",
                "min":  float(s.min()),  "max":  float(s.max()),
                "mean": float(s.mean()), "std":  float(s.std()),
                "p25":  float(s.quantile(0.25)),
                "p75":  float(s.quantile(0.75)),
            }
        else:
            stats[col] = {
                "type":   "categorical",
                "values": s.dropna().unique().tolist(),
                "freq":   s.value_counts(normalize=True).to_dict(),
            }
    return stats


def _train() -> dict:
    df_raw = pd.read_csv(DATA_PATH)
    df_raw[TARGET_COL] = df_raw[TARGET_COL].astype(int)
    df_raw.drop(columns=[c for c in DROP_COLS if c in df_raw.columns],
                inplace=True)

    df = engineer_features(df_raw.copy())
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
        ]), cat_cols),
    ], remainder="drop")

    lgbm = LGBMClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        num_leaves=50, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=6,
        random_state=RANDOM_SEED, verbosity=-1, n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", lgbm)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    auc_scores = cross_val_score(pipeline, X, y, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
    pipeline.fit(X, y)

    artefacts = {
        "pipeline":   pipeline,
        "auc":        round(float(auc_scores.mean()), 4),
        "num_cols":   num_cols,
        "cat_cols":   cat_cols,
        "data_stats": _compute_stats(df_raw),
    }
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artefacts, f)
    except OSError:
        pass
    return artefacts


# ─────────────────────────────────────────────────────────
# LOAD OR TRAIN
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_model() -> dict:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return _train()


# ─────────────────────────────────────────────────────────
# RANDOM CUSTOMER GENERATOR
# ─────────────────────────────────────────────────────────
_BASE_COLS = [
    "state", "account_length", "area_code", "intl_plan", "voice_mail_plan",
    "number_vmail_messages",
    "total_day_minutes",   "total_day_calls",   "total_day_charge",
    "total_eve_minutes",   "total_eve_calls",   "total_eve_charge",
    "total_night_minutes", "total_night_calls", "total_night_charge",
    "total_intl_minutes",  "total_intl_calls",  "total_intl_charge",
    "number_customer_service_calls",
]
_INT_COLS = {
    "account_length", "total_day_calls", "total_eve_calls",
    "total_night_calls", "total_intl_calls",
    "number_customer_service_calls", "number_vmail_messages", "area_code",
}

def generate_random_customer(stats: dict, seed: int = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    row = {}
    for col in _BASE_COLS:
        if col not in stats:
            continue
        s = stats[col]
        if s["type"] == "numeric":
            val = float(np.clip(rng.normal(s["mean"], s["std"]),
                                s["min"], s["max"]))
            if col in _INT_COLS:
                val = int(round(val))
            row[col] = val
        else:
            choices = list(s["freq"].keys())
            probs   = np.array(list(s["freq"].values()), dtype=float)
            probs  /= probs.sum()
            row[col] = rng.choice(choices, p=probs)
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────
def predict(pipeline, df_input: pd.DataFrame):
    df_feat = engineer_features(df_input)
    prob    = float(pipeline.predict_proba(df_feat)[0][1])
    return prob, df_feat


# ─────────────────────────────────────────────────────────
# SHAP EXPLANATIONS
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_shap_values(_pipeline, df_feat_json: str,
                    num_cols: tuple, cat_cols: tuple) -> pd.DataFrame:
    df_feat      = pd.read_json(io.StringIO(df_feat_json))
    preprocessor = _pipeline.named_steps["preprocessor"]
    lgbm_model   = _pipeline.named_steps["clf"]
    X_t          = preprocessor.transform(df_feat)
    explainer    = shap.TreeExplainer(lgbm_model)
    sv           = explainer.shap_values(X_t)
    if isinstance(sv, list):
        sv = sv[1][0]
    else:
        sv = sv[0]
    feat_names = list(num_cols) + list(cat_cols)
    return (pd.DataFrame({"feature": feat_names, "shap": sv,
                           "abs": np.abs(sv)})
              .sort_values("abs", ascending=False))


# ─────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────
def gauge_chart(prob: float, dark: bool) -> go.Figure:
    pct   = prob * 100
    color = "#10b981" if pct < 40 else ("#f59e0b" if pct < 70 else "#ef4444")
    bg    = "#1e293b" if dark else "#ffffff"
    tx    = "#f1f5f9" if dark else "#1e293b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 44, "color": tx}},
        delta={"reference": 40,
               "increasing": {"color": "#ef4444"},
               "decreasing": {"color": "#10b981"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": tx, "tickfont": {"color": tx}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": bg, "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#d1fae5" if not dark else "#064e3b"},
                {"range": [40, 70], "color": "#fef3c7" if not dark else "#78350f"},
                {"range": [70,100], "color": "#fee2e2" if not dark else "#7f1d1d"},
            ],
            "threshold": {"line": {"color": color, "width": 4},
                          "thickness": 0.75, "value": pct},
        },
        title={"text": "Churn Probability",
               "font": {"size": 16, "color": tx}},
    ))
    fig.update_layout(height=300,
                      margin=dict(t=40, b=20, l=20, r=20),
                      paper_bgcolor=bg, plot_bgcolor=bg)
    return fig


def shap_bar_chart(shap_df: pd.DataFrame, dark: bool,
                   top_n: int = 10) -> go.Figure:
    top    = shap_df.head(top_n).sort_values("shap")
    labels = [FEATURE_LABELS.get(f, f) for f in top["feature"]]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in top["shap"]]
    bg     = "#1e293b" if dark else "#ffffff"
    tx     = "#f1f5f9" if dark else "#1e293b"
    fig = go.Figure(go.Bar(
        x=top["shap"], y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in top["shap"]],
        textposition="outside", textfont={"color": tx},
    ))
    fig.update_layout(
        title=dict(text="Top SHAP Feature Contributions",
                   font=dict(color=tx, size=15)),
        xaxis_title="SHAP Value  (impact on churn probability)",
        xaxis=dict(color=tx,
                   gridcolor="#334155" if dark else "#e2e8f0"),
        yaxis=dict(color=tx),
        height=380, margin=dict(t=50, b=30, l=20, r=80),
        paper_bgcolor=bg, plot_bgcolor=bg,
    )
    return fig


# ─────────────────────────────────────────────────────────
# BATCH PREDICTION  (fixed: validates columns before predict)
# ─────────────────────────────────────────────────────────
def batch_predict(pipeline, df_upload: pd.DataFrame) -> pd.DataFrame:
    # Normalise column names: strip whitespace and lowercase so headers
    # like "Total_Day_Charge" or " state " still match.
    df_c = df_upload.copy()
    df_c.columns = df_c.columns.str.strip().str.lower()

    # Drop identifier / target columns if present
    df_c = df_c.drop(
        columns=[c for c in DROP_COLS + [TARGET_COL] if c in df_c.columns],
        errors="ignore",
    )

    # ── Validate required columns ────────────────────────
    missing = [c for c in REQUIRED_BATCH_COLS if c not in df_c.columns]
    if missing:
        missing_str = "\n".join(f"  • {c}" for c in missing)
        raise ValueError(
            f"Your CSV is missing **{len(missing)} required column(s)**:\n\n"
            f"{missing_str}\n\n"
            "Make sure the file has the same column names as the training data.\n"
            "Column names are case-insensitive and leading/trailing spaces are ignored."
        )

    probs  = pipeline.predict_proba(engineer_features(df_c))[:, 1]
    df_out = df_upload.copy()
    df_out["churn_probability_%"] = (probs * 100).round(2)
    df_out["risk_level"] = pd.cut(
        probs, bins=[0, 0.4, 0.7, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True,
    )
    return df_out


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    first_run = not os.path.exists(MODEL_PATH)
    if first_run:
        _banner = st.warning(
            "⚙️  **First-run setup** — training the LightGBM model "
            "(≈ 30 s on Streamlit Cloud)…  Please wait.")

    artefacts = load_or_train_model()

    if first_run:
        _banner.empty()

    pipeline  = artefacts["pipeline"]
    num_cols  = artefacts["num_cols"]
    cat_cols  = artefacts["cat_cols"]
    stats     = artefacts["data_stats"]
    auc_score = artefacts["auc"]

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        try:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/"
                "4/44/Orange_logo.svg/320px-Orange_logo.svg.png",
                width=140,
            )
        except Exception:
            st.markdown("### 📡 Orange Telecom")

        st.markdown("## ⚙️ Settings")
        dark_mode = st.toggle("🌙 Dark Mode", value=True)
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.metric("CV AUC",           f"{auc_score:.4f}")
        st.metric("Algorithm",        "LightGBM")
        st.metric("Training Samples", "5,000")
        st.markdown("---")
        st.markdown("### 📂 Batch Prediction")
        st.caption(
            "Upload a CSV with these columns:\n"
            + ", ".join(f"`{c}`" for c in REQUIRED_BATCH_COLS)
        )
        uploaded_file = st.file_uploader(
            "Upload CSV for batch scoring", type=["csv"],
            help="Must contain the same columns as the training data.")
        st.markdown("---")
        st.caption("Orange Telecom Churn System · LightGBM + SHAP")

    # ── Apply theme ───────────────────────────────────────
    st.markdown(DARK_CSS if dark_mode else LIGHT_CSS, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;font-size:2.2rem;font-weight:800;"
        "margin-bottom:4px;'>📡 Orange Telecom Churn Prediction System</h1>"
        "<p style='text-align:center;color:#94a3b8;margin-bottom:28px;'>"
        "AI-powered customer churn intelligence &nbsp;|&nbsp; "
        "LightGBM + SHAP Explainability</p>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════
    # BATCH SECTION
    # ══════════════════════════════════════════════════════
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("## 📂 Batch Prediction Results")
        df_up = pd.read_csv(uploaded_file)
        st.write(f"Uploaded **{len(df_up):,} rows**")
        try:
            results = batch_predict(pipeline, df_up)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Customers", f"{len(results):,}")
            c2.metric("High Risk",   f"{(results['risk_level']=='High Risk').sum():,}")
            c3.metric("Medium Risk", f"{(results['risk_level']=='Medium Risk').sum():,}")

            # pandas-version-agnostic: always produces "Risk Level" / "Count"
            _vc = results["risk_level"].value_counts().reset_index()
            _vc.columns = ["Risk Level", "Count"]
            risk_counts = _vc

            cmap = {"Low Risk": "#10b981", "Medium Risk": "#f59e0b",
                    "High Risk": "#ef4444"}
            fig_d = px.bar(risk_counts, x="Risk Level", y="Count",
                           color="Risk Level", color_discrete_map=cmap,
                           title="Risk Level Distribution")
            fig_d.update_layout(
                paper_bgcolor="#1e293b" if dark_mode else "#ffffff",
                plot_bgcolor ="#1e293b" if dark_mode else "#ffffff",
                font_color   ="#f1f5f9" if dark_mode else "#1e293b",
                showlegend=False)
            st.plotly_chart(fig_d, use_container_width=True)

            front = ["churn_probability_%", "risk_level"]
            other = [c for c in results.columns if c not in front]
            st.dataframe(results[front + other].head(200),
                         use_container_width=True)
            st.download_button(
                "⬇️  Download Full Results CSV",
                data=results.to_csv(index=False).encode(),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )
        except ValueError as exc:
            # Show the missing-columns message clearly, not as a traceback
            st.error(str(exc))
            st.info(
                "💡 **Tip:** Download the template below to see the exact "
                "column names your CSV needs."
            )
            template = pd.DataFrame(columns=REQUIRED_BATCH_COLS)
            st.download_button(
                "⬇️  Download Column Template CSV",
                data=template.to_csv(index=False).encode(),
                file_name="batch_template.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")
        st.markdown("---")

    # ══════════════════════════════════════════════════════
    # SINGLE CUSTOMER – MANUAL INPUT FORM
    # ══════════════════════════════════════════════════════
    st.markdown("## 👤 Customer Details")
    st.markdown(
        '<p style="color:#94a3b8;margin-bottom:16px;">'
        "Enter the customer's details below and click <strong>Predict Churn</strong> "
        "to get an instant risk assessment with explanations and recommendations.</p>",
        unsafe_allow_html=True,
    )

    if "customer_df" not in st.session_state:
        st.session_state.customer_df = None

    def _num(col, key, fallback):
        return stats.get(col, {}).get(key, fallback)

    US_STATES = [
        "AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID",
        "IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC",
        "ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD",
        "TN","TX","UT","VA","VT","WA","WI","WV","WY",
    ]
    cat_state_vals = stats.get("state", {}).get("values", US_STATES)
    cat_state_vals = sorted(cat_state_vals) if cat_state_vals else US_STATES

    with st.form("customer_form"):
        st.markdown('<p class="section-header">📋 Account Information</p>',
                    unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        state         = a1.selectbox("State", options=cat_state_vals, index=0)
        area_code     = a2.number_input(
            "Area Code", min_value=200, max_value=999,
            value=int(_num("area_code", "mean", 415)), step=1)
        account_length = a3.number_input(
            "Account Length (days)",
            min_value=int(_num("account_length", "min", 1)),
            max_value=int(_num("account_length", "max", 243)),
            value=int(_num("account_length", "mean", 100)), step=1)
        number_vmail_messages = a4.number_input(
            "Voicemail Messages",
            min_value=int(_num("number_vmail_messages", "min", 0)),
            max_value=int(_num("number_vmail_messages", "max", 51)),
            value=int(_num("number_vmail_messages", "mean", 8)), step=1)

        b1, b2 = st.columns(2)
        intl_plan_vals        = stats.get("intl_plan",       {}).get("values", ["no", "yes"])
        voice_mail_plan_vals  = stats.get("voice_mail_plan", {}).get("values", ["no", "yes"])
        intl_plan        = b1.selectbox("International Plan",  options=intl_plan_vals)
        voice_mail_plan  = b2.selectbox("Voicemail Plan",      options=voice_mail_plan_vals)

        st.markdown('<p class="section-header">☀️ Daytime Usage</p>',
                    unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        total_day_minutes = d1.number_input(
            "Day Minutes",
            min_value=0.0, max_value=float(_num("total_day_minutes", "max", 350)),
            value=float(_num("total_day_minutes", "mean", 180)), step=0.1, format="%.1f")
        total_day_calls   = d2.number_input(
            "Day Calls",
            min_value=0, max_value=int(_num("total_day_calls", "max", 165)),
            value=int(_num("total_day_calls", "mean", 100)), step=1)
        total_day_charge  = d3.number_input(
            "Day Charge ($)",
            min_value=0.0, max_value=float(_num("total_day_charge", "max", 60)),
            value=float(_num("total_day_charge", "mean", 30)), step=0.01, format="%.2f")

        st.markdown('<p class="section-header">🌆 Evening Usage</p>',
                    unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        total_eve_minutes = e1.number_input(
            "Evening Minutes",
            min_value=0.0, max_value=float(_num("total_eve_minutes", "max", 364)),
            value=float(_num("total_eve_minutes", "mean", 200)), step=0.1, format="%.1f")
        total_eve_calls   = e2.number_input(
            "Evening Calls",
            min_value=0, max_value=int(_num("total_eve_calls", "max", 170)),
            value=int(_num("total_eve_calls", "mean", 100)), step=1)
        total_eve_charge  = e3.number_input(
            "Evening Charge ($)",
            min_value=0.0, max_value=float(_num("total_eve_charge", "max", 31)),
            value=float(_num("total_eve_charge", "mean", 17)), step=0.01, format="%.2f")

        st.markdown('<p class="section-header">🌙 Night-time Usage</p>',
                    unsafe_allow_html=True)
        n1, n2, n3 = st.columns(3)
        total_night_minutes = n1.number_input(
            "Night Minutes",
            min_value=0.0, max_value=float(_num("total_night_minutes", "max", 395)),
            value=float(_num("total_night_minutes", "mean", 200)), step=0.1, format="%.1f")
        total_night_calls   = n2.number_input(
            "Night Calls",
            min_value=0, max_value=int(_num("total_night_calls", "max", 175)),
            value=int(_num("total_night_calls", "mean", 100)), step=1)
        total_night_charge  = n3.number_input(
            "Night Charge ($)",
            min_value=0.0, max_value=float(_num("total_night_charge", "max", 18)),
            value=float(_num("total_night_charge", "mean", 9)), step=0.01, format="%.2f")

        st.markdown('<p class="section-header">🌐 International Usage</p>',
                    unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)
        total_intl_minutes = i1.number_input(
            "Intl Minutes",
            min_value=0.0, max_value=float(_num("total_intl_minutes", "max", 20)),
            value=float(_num("total_intl_minutes", "mean", 10)), step=0.1, format="%.1f")
        total_intl_calls   = i2.number_input(
            "Intl Calls",
            min_value=0, max_value=int(_num("total_intl_calls", "max", 20)),
            value=int(_num("total_intl_calls", "mean", 4)), step=1)
        total_intl_charge  = i3.number_input(
            "Intl Charge ($)",
            min_value=0.0, max_value=float(_num("total_intl_charge", "max", 5.5)),
            value=float(_num("total_intl_charge", "mean", 2.76)), step=0.01, format="%.2f")

        st.markdown('<p class="section-header">📞 Support History</p>',
                    unsafe_allow_html=True)
        number_customer_service_calls = st.number_input(
            "Number of Customer Service Calls",
            min_value=0, max_value=int(_num("number_customer_service_calls", "max", 9)),
            value=int(_num("number_customer_service_calls", "mean", 1)), step=1,
            help="Higher values (4+) are a strong churn signal.")

        st.markdown("")
        submitted = st.form_submit_button(
            "🔍 Predict Churn", type="primary", use_container_width=True)

    if submitted:
        st.session_state.customer_df = pd.DataFrame([{
            "state":                          state,
            "account_length":                 account_length,
            "area_code":                      area_code,
            "intl_plan":                      intl_plan,
            "voice_mail_plan":                voice_mail_plan,
            "number_vmail_messages":          number_vmail_messages,
            "total_day_minutes":              total_day_minutes,
            "total_day_calls":                total_day_calls,
            "total_day_charge":               total_day_charge,
            "total_eve_minutes":              total_eve_minutes,
            "total_eve_calls":                total_eve_calls,
            "total_eve_charge":               total_eve_charge,
            "total_night_minutes":            total_night_minutes,
            "total_night_calls":              total_night_calls,
            "total_night_charge":             total_night_charge,
            "total_intl_minutes":             total_intl_minutes,
            "total_intl_calls":               total_intl_calls,
            "total_intl_charge":              total_intl_charge,
            "number_customer_service_calls":  number_customer_service_calls,
        }])

    if st.session_state.customer_df is not None:
        df_cust = st.session_state.customer_df

        st.markdown("---")
        st.markdown('<p class="section-header">👤 Customer Summary</p>',
                    unsafe_allow_html=True)
        pc = st.columns(4)
        fields = [
            ("State",          df_cust["state"].iloc[0]),
            ("Account Length", f"{df_cust['account_length'].iloc[0]} days"),
            ("Area Code",      df_cust["area_code"].iloc[0]),
            ("Intl Plan",      df_cust["intl_plan"].iloc[0]),
            ("Voicemail Plan", df_cust["voice_mail_plan"].iloc[0]),
            ("Day Minutes",    f"{df_cust['total_day_minutes'].iloc[0]:.1f}"),
            ("Eve Minutes",    f"{df_cust['total_eve_minutes'].iloc[0]:.1f}"),
            ("Night Minutes",  f"{df_cust['total_night_minutes'].iloc[0]:.1f}"),
            ("Intl Minutes",   f"{df_cust['total_intl_minutes'].iloc[0]:.1f}"),
            ("Day Calls",      df_cust["total_day_calls"].iloc[0]),
            ("Intl Calls",     df_cust["total_intl_calls"].iloc[0]),
            ("Service Calls",  df_cust["number_customer_service_calls"].iloc[0]),
        ]
        for i, (lbl, val) in enumerate(fields):
            pc[i % 4].metric(lbl, val)

        prob, df_feat = predict(pipeline, df_cust)

        risk_label, risk_color, risk_emoji = "Low Risk", "#10b981", "🟢"
        for band, (lo, hi, col_, emoji) in RISK_BANDS.items():
            if lo <= prob < hi or (band == "High Risk" and prob >= 0.70):
                risk_label, risk_color, risk_emoji = band, col_, emoji
                break

        st.markdown("---")
        st.markdown("## 📈 Prediction Output")
        left, right = st.columns(2)

        with left:
            st.plotly_chart(gauge_chart(prob, dark_mode),
                            use_container_width=True)
            st.markdown(
                f'<div style="text-align:center">'
                f'<span class="risk-badge" '
                f'style="background:{risk_color}22;color:{risk_color};'
                f'border:2px solid {risk_color};">'
                f'{risk_emoji} {risk_label}</span></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<p style="text-align:center;color:#94a3b8;margin-top:8px;">'
                f'Churn Probability: <strong>{prob*100:.1f}%</strong></p>',
                unsafe_allow_html=True)
            st.markdown('<p class="section-header">Risk Meter</p>',
                        unsafe_allow_html=True)
            st.progress(min(prob, 1.0))
            r1, r2, r3 = st.columns(3)
            r1.markdown('<small style="color:#10b981">● Low (0–40%)</small>',
                        unsafe_allow_html=True)
            r2.markdown('<small style="color:#f59e0b">● Medium (40–70%)</small>',
                        unsafe_allow_html=True)
            r3.markdown('<small style="color:#ef4444">● High (70–100%)</small>',
                        unsafe_allow_html=True)

        with right:
            shap_df = None
            with st.spinner("Computing SHAP explanations…"):
                try:
                    shap_df = get_shap_values(
                        pipeline,
                        df_feat.to_json(),
                        tuple(num_cols), tuple(cat_cols),
                    )
                    st.plotly_chart(shap_bar_chart(shap_df, dark_mode),
                                    use_container_width=True)
                except Exception as exc:
                    st.warning(f"SHAP skipped: {exc}")

        st.markdown("---")
        st.markdown("## 🔍 Churn Insights & Explainability")
        il, ir = st.columns([1.1, 0.9])

        with il:
            st.markdown(
                '<p class="section-header">🧠 Why This Customer Might Churn</p>',
                unsafe_allow_html=True)
            if shap_df is not None:
                shown = 0
                for _, row_s in shap_df[shap_df["shap"] > 0].head(5).iterrows():
                    feat = row_s["feature"]
                    if feat in CHURN_REASONS:
                        lbl, det = CHURN_REASONS[feat]
                        st.markdown(
                            f'<div class="insight-row">'
                            f'<strong>{lbl}</strong><br>'
                            f'<small style="color:#94a3b8">{det}</small>'
                            f'</div>',
                            unsafe_allow_html=True)
                        shown += 1
                if shown == 0:
                    st.info("Risk is distributed across multiple factors "
                            "with no single dominant driver.")
            else:
                df_eng  = engineer_features(df_cust)
                reasons = []
                if df_eng["number_customer_service_calls"].iloc[0] >= 4:
                    reasons.append(("📞 Frequent support contacts",
                                    "4+ service calls — strong churn signal"))
                if df_eng["total_charge"].iloc[0] > 70:
                    reasons.append(("💸 High total bill",
                                    "Charges significantly above average"))
                if (df_cust["intl_plan"].iloc[0] == "yes" and
                        df_eng["total_intl_charge"].iloc[0] > 4):
                    reasons.append(("🌍 Costly international usage",
                                    "High international charges under current plan"))
                if df_cust["account_length"].iloc[0] < 50:
                    reasons.append(("📅 Short tenure",
                                    "Newer accounts churn more frequently"))
                for lbl, det in reasons:
                    st.markdown(
                        f'<div class="insight-row"><strong>{lbl}</strong><br>'
                        f'<small style="color:#94a3b8">{det}</small></div>',
                        unsafe_allow_html=True)
                if not reasons:
                    st.info("No dominant churn driver detected.")

        with ir:
            st.markdown('<p class="section-header">💼 Recommended Actions</p>',
                        unsafe_allow_html=True)
            for rec in RECOMMENDATIONS.get(risk_label, []):
                st.markdown(
                    f'<div class="insight-row">{rec}</div>',
                    unsafe_allow_html=True)

            st.markdown('<p class="section-header">📊 Usage Summary</p>',
                        unsafe_allow_html=True)
            df_eng = engineer_features(df_cust)
            for k, v in {
                "Total Monthly Charge": f"${df_eng['total_charge'].iloc[0]:.2f}",
                "Avg. Charge / Call":   f"${df_eng['charge_per_call'].iloc[0]:.3f}",
                "Total Minutes":        f"{df_eng['total_minutes'].iloc[0]:.0f} min",
                "Service Call Rate":    f"{df_eng['service_call_rate'].iloc[0]:.4f}",
            }.items():
                ca, cb = st.columns([2, 1])
                ca.write(k)
                cb.write(f"**{v}**")

        if shap_df is not None:
            with st.expander("📋 Full SHAP Feature Importance Table"):
                disp = shap_df.copy()
                disp["Feature Description"] = disp["feature"].map(
                    lambda f: FEATURE_LABELS.get(f, f))
                disp["Direction"] = disp["shap"].apply(
                    lambda v: "↑ Increases churn risk"
                    if v > 0 else "↓ Reduces churn risk")
                st.dataframe(
                    disp[["Feature Description", "shap", "Direction"]
                         ].rename(columns={"shap": "SHAP Value"}),
                    use_container_width=True)

    st.markdown("---")
    st.markdown(
        f'<p style="text-align:center;color:#64748b;font-size:0.82rem;">'
        f'Orange Telecom Churn Prediction System &nbsp;|&nbsp; '
        f'LightGBM · SHAP · Streamlit &nbsp;|&nbsp; '
        f'Model AUC: <strong>{auc_score:.4f}</strong></p>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
