
# dashboard/app.py
# Fraud Detection Operations Dashboard
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import joblib
import pickle
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>

div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
}

div[data-testid="metric-container"] label {
    color: #9ca3af !important;
}

div[data-testid="metric-container"] div {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ── Load Data & Model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("model.pkl")
    scaler   = joblib.load("scaler.pkl")
    with open("shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, explainer, feature_names

@st.cache_data
def load_data():
    return pd.read_csv("transactions.csv")

model, scaler, explainer, feature_names = load_model()
df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lock.png", width=80)
    st.title("🔐 Fraud Ops")
    st.markdown("---")
    page = st.radio("Navigation", ["📊 Overview", "🔍 Transaction Explorer", "🧠 SHAP Explainer"])
    st.markdown("---")
    st.subheader("Global Filters")
    risk_filter = st.multiselect(
        "Risk Tier",
        options=["🔴 Critical Risk", "🟡 Suspicious", "🟢 Clear"],
        default=["🔴 Critical Risk", "🟡 Suspicious", "🟢 Clear"]
    )
    prob_range = st.slider("Fraud Probability Range", 0.0, 1.0, (0.0, 1.0), 0.01)
    st.markdown("---")
    st.caption("Fraud Detection System v1.0")
    st.caption("Powered by LightGBM + SHAP")

# Apply filters
df_filtered = df[
    df["risk_tier"].isin(risk_filter) &
    df["fraud_prob"].between(*prob_range)
]

# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Fraud Detection Overview")
    st.markdown("Real-time fraud monitoring dashboard powered by LightGBM + SHAP")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total = len(df_filtered)
    fraud_count = df_filtered["isFraud"].sum()
    detection_rate = fraud_count / max(total, 1) * 100
    avg_fraud_amt = df_filtered[df_filtered["isFraud"]==1]["TransactionAmt"].mean() if fraud_count > 0 else 0

    col1.metric("📦 Total Transactions", f"{total:,}")
    col2.metric("🚨 Total Fraud Cases", f"{fraud_count:,}",
                delta=f"+{fraud_count - int(total*0.035):,} vs expected")
    col3.metric("📈 Detection Rate", f"{detection_rate:.2f}%")
    col4.metric("💰 Avg Fraud Amount", f"${avg_fraud_amt:,.2f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # Risk tier donut chart
    tier_counts = df_filtered["risk_tier"].value_counts().reset_index()
    tier_counts.columns = ["Risk Tier", "Count"]
    fig_donut = px.pie(
        tier_counts, values="Count", names="Risk Tier",
        hole=0.5, title="Risk Tier Distribution",
        color="Risk Tier",
        color_discrete_map={
            "🔴 Critical Risk": "#e74c3c",
            "🟡 Suspicious": "#f39c12",
            "🟢 Clear": "#2ecc71"
        }
    )
    fig_donut.update_layout(height=350)
    col_left.plotly_chart(fig_donut, use_container_width=True)

    # Fraud by hour
    if "HourOfDay" in df_filtered.columns:
        hourly = df_filtered.groupby("HourOfDay")["isFraud"].agg(["sum","count"]).reset_index()
        hourly["fraud_rate"] = hourly["sum"] / hourly["count"] * 100
        fig_hour = px.bar(
            hourly, x="HourOfDay", y="fraud_rate",
            title="Fraud Rate by Hour of Day",
            color="fraud_rate",
            color_continuous_scale="Reds"
        )
        fig_hour.update_layout(height=350)
        col_right.plotly_chart(fig_hour, use_container_width=True)

    # Fraud probability histogram
    st.subheader("Fraud Probability Distribution")
    fig_prob = px.histogram(
        df_filtered, x="fraud_prob", color="risk_tier",
        nbins=50, barmode="overlay", opacity=0.7,
        color_discrete_map={
            "🔴 Critical Risk": "#e74c3c",
            "🟡 Suspicious": "#f39c12",
            "🟢 Clear": "#2ecc71"
        },
        title="Distribution of Fraud Probability Scores",
        labels={"fraud_prob": "Fraud Probability", "count": "Transactions"}
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # Transaction amount scatter
    if "TransactionAmt" in df_filtered.columns and "HourOfDay" in df_filtered.columns:
        st.subheader("Transaction Amount vs Hour (coloured by Fraud Probability)")
        sample = df_filtered.sample(min(3000, len(df_filtered)))
        fig_scatter = px.scatter(
            sample, x="HourOfDay", y="TransactionAmt",
            color="fraud_prob", color_continuous_scale="RdYlGn_r",
            opacity=0.6, size_max=8,
            title="TransactionAmt vs HourOfDay (Fraud Probability)",
            labels={"fraud_prob": "Fraud Prob"},
            log_y=True
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 2 — TRANSACTION EXPLORER
# ════════════════════════════════════════════════════════════════
elif page == "🔍 Transaction Explorer":
    st.title("🔍 Transaction Explorer")
    st.markdown("Search and filter transactions with live risk scores.")

    col1, col2 = st.columns(2)
    txn_search = col1.text_input("🔎 Search TransactionID", placeholder="Enter TransactionID")
    sort_by = col2.selectbox("Sort By", ["fraud_prob", "TransactionAmt"], index=0)

    display_df = df_filtered.copy()

    if txn_search:
        try:
            txn_id = int(txn_search)
            display_df = display_df[display_df["TransactionID"] == txn_id]
        except ValueError:
            st.warning("Please enter a valid numeric TransactionID")

    display_df = display_df.sort_values(sort_by, ascending=False)

    # Show summary KPIs for filtered set
    c1, c2, c3 = st.columns(3)
    c1.metric("Filtered Transactions", f"{len(display_df):,}")
    c2.metric("Avg Fraud Prob", f"{display_df['fraud_prob'].mean():.3f}")
    c3.metric("Fraud Count", f"{display_df['isFraud'].sum():,}")

    # Table
    show_cols = ["TransactionID", "TransactionAmt", "fraud_prob", "risk_tier", "isFraud"]
    show_cols = [c for c in show_cols if c in display_df.columns]
    st.dataframe(
        display_df[show_cols].head(500).style
            .background_gradient(subset=["fraud_prob"], cmap="RdYlGn_r")
            .format({"fraud_prob": "{:.4f}", "TransactionAmt": "${:,.2f}"}),
        height=450,
        use_container_width=True
    )

    # Live risk score lookup
    st.subheader("Live Risk Score Lookup")
    lookup_id = st.number_input("Enter TransactionID for Risk Score", min_value=0, step=1)
    if st.button("🔍 Get Risk Score"):
        row = df[df["TransactionID"] == lookup_id]
        if len(row) > 0:
            prob = row["fraud_prob"].values[0]
            tier = row["risk_tier"].values[0]
            st.success(f"TransactionID {lookup_id}: Fraud Probability = **{prob:.4f}** | Risk Tier = **{tier}**")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Fraud Risk Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred"},
                    "steps": [
                        {"range": [0, 40],  "color": "#2ecc71"},
                        {"range": [40, 75], "color": "#f39c12"},
                        {"range": [75, 100],"color": "#e74c3c"}
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "value": prob*100}
                }
            ))
            gauge.update_layout(height=300)
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.error(f"TransactionID {lookup_id} not found in test set.")

# ════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP EXPLAINER
# ════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainer":
    st.title("🧠 SHAP Transaction Explainer")
    st.markdown("Enter a TransactionID to see a detailed SHAP-based explanation of the fraud prediction.")

    txn_id_input = st.number_input("TransactionID", min_value=0, step=1, value=0)

    if st.button("🔍 Explain This Transaction"):
        row = df[df["TransactionID"] == txn_id_input]
        if len(row) == 0:
            st.error("Transaction not found.")
        else:
            prob  = row["fraud_prob"].values[0]
            tier  = row["risk_tier"].values[0]
            label = row["isFraud"].values[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Fraud Probability", f"{prob:.4f}")
            col2.metric("Risk Tier", tier)
            col3.metric("Actual Label", "FRAUD" if label == 1 else "LEGITIMATE")

            # SHAP waterfall
            feat_cols = [c for c in feature_names if c in row.columns]
            X_row = row[feat_cols].values

            with st.spinner("Computing SHAP values..."):
                shap_vals = explainer.shap_values(X_row)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                    ev = explainer.expected_value[1]
                else:
                    sv = shap_vals[0]
                    ev = explainer.expected_value

            shap_exp = shap.Explanation(
                values=sv, base_values=ev,
                data=X_row[0], feature_names=feat_cols
            )

            fig, ax = plt.subplots(figsize=(12, 6))
            shap.waterfall_plot(shap_exp, max_display=15, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Plain-English explanation
            st.subheader("📝 Plain-English Explanation")
            feat_sv = sorted(zip(feat_cols, sv), key=lambda x: abs(x[1]), reverse=True)

            if prob >= 0.75:
                verdict = "🚨 This transaction is flagged as HIGH RISK."
            elif prob >= 0.40:
                verdict = "⚠️ This transaction is SUSPICIOUS and requires manual review."
            else:
                verdict = "✅ This transaction appears LEGITIMATE."

            explanation_lines = [verdict, ""]
            explanation_lines.append("**Key factors driving this prediction:**")
            for i, (feat, val) in enumerate(feat_sv[:5]):
                direction = "increases" if val > 0 else "decreases"
                explanation_lines.append(
                    f"{i+1}. **{feat}** — {direction} fraud risk (SHAP={val:+.4f})"
                )

            st.markdown(" ".join(explanation_lines))

            # Top features bar chart
            top10 = feat_sv[:10]
            fig2 = go.Figure(go.Bar(
                y=[f[0] for f in top10],
                x=[f[1] for f in top10],
                orientation="h",
                marker_color=["#e74c3c" if v > 0 else "#2ecc71" for _, v in top10]
            ))
            fig2.update_layout(
                title="Top 10 Feature SHAP Values",
                xaxis_title="SHAP Value",
                height=350
            )
            st.plotly_chart(fig2, use_container_width=True)
