import os
import json
import random
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

LEVELS = ["Baseline", "Blur", "SDInpaint", "Edge", "RAD"]
LEVEL_LABELS = {
    "Baseline": "Baseline (No Anonymization)",
    "Blur": "L1: Gaussian Blur",
    "Edge": "L4: Edge Silhouette",
    "RAD": "L3: RAD (Realistic Anonymization by Diffusion)",
    "SDInpaint": "L2: SD Inpaint",
}
LEVEL_COLORS = {
    "Baseline": "#6366f1",
    "Blur": "#3b82f6",
    "SDInpaint": "#f59e0b",
    "Edge": "#8b5cf6",
    "RAD": "#ef4444",
}
DATASET_DIRS = {
    "Baseline": os.path.join(RAW_DIR, "Market-1501"),
    "Blur": os.path.join(PROCESSED_DIR, "Market-1501-Blur"),
    "Edge": os.path.join(PROCESSED_DIR, "Market-1501-Edge"),
    "RAD": os.path.join(PROCESSED_DIR, "Market-1501-RAD"),
    "SDInpaint": os.path.join(PROCESSED_DIR, "Market-1501-SDInpaint"),
}

UTILITY_COLOR = "#3b82f6"
UTILITY_COLOR_LIGHT = "#93c5fd"
PRIVACY_COLOR = "#ef4444"


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def get_query_filenames():
    query_dir = os.path.join(RAW_DIR, "Market-1501", "query")
    if not os.path.isdir(query_dir):
        return []
    return sorted([f for f in os.listdir(query_dir) if f.endswith(".jpg") and not f.startswith("-1")])


def get_identities(filenames):
    ids = {}
    for f in filenames:
        pid = f.split("_")[0]
        ids.setdefault(pid, []).append(f)
    return ids


def build_unified_data(baseline_data, utility_data, privacy_data):
    rows = []
    for level in LEVELS:
        row = {"Level": level, "Label": LEVEL_LABELS[level]}
        if level == "Baseline" and baseline_data:
            b = baseline_data.get("Baseline", {})
            row["mAP"] = b.get("mAP")
            row["Rank-1"] = b.get("Rank-1")
            row["Rank-5"] = b.get("Rank-5")
            row["Rank-10"] = b.get("Rank-10")
            row["Rank-20"] = b.get("Rank-20")
        elif utility_data and level in utility_data:
            u = utility_data[level]
            row["mAP"] = u.get("mAP")
            row["Rank-1"] = u.get("Rank-1")
            row["Rank-5"] = u.get("Rank-5")
            row["Rank-10"] = u.get("Rank-10")
            row["Rank-20"] = u.get("Rank-20")

        if level == "Baseline":
            row["Privacy Leak (%)"] = 100.0
        elif privacy_data and level in privacy_data:
            row["Privacy Leak (%)"] = privacy_data[level].get("Top-1_Accuracy")

        rows.append(row)

    df = pd.DataFrame(rows)

    baseline_mAP = df.loc[df["Level"] == "Baseline", "mAP"].iloc[0] if not df.empty else None
    baseline_r1 = df.loc[df["Level"] == "Baseline", "Rank-1"].iloc[0] if not df.empty else None

    df["Privacy Protection (%)"] = 100.0 - df["Privacy Leak (%)"]
    if pd.notna(baseline_mAP) and baseline_mAP > 0:
        df["mAP % of Baseline"] = (df["mAP"] / baseline_mAP) * 100
        df["mAP Drop (%)"] = (1 - df["mAP"] / baseline_mAP) * 100
    if pd.notna(baseline_r1) and baseline_r1 > 0:
        df["R-1 Drop (%)"] = (1 - df["Rank-1"] / baseline_r1) * 100

    utility_cost = 100 - df["mAP"] * 100
    df["Effectiveness Index"] = df["Privacy Protection (%)"] / utility_cost.replace(0, np.nan)

    return df


def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        div[data-testid="stMetric"] {
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            color: #1e293b !important;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p,
        div[data-testid="stMetric"] label div,
        div[data-testid="stMetric"] label span {
            font-size: 0.8rem;
            color: #475569 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] div,
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] p,
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] span {
            font-size: 1.4rem;
            color: #1e293b !important;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
        .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(df):
    st.title("ANON4REID")
    st.caption(
        "Adversarial Threat Model & Privacy-Utility Benchmark  --  "
        "Measuring OSNet (Utility) degradation vs. MobileNetV2 (Privacy Leakage) "
        "across anonymization techniques on Market-1501."
    )
    st.divider()


def render_tradeoff_tab(df):
    # --- Main dual-axis trade-off chart ---
    st.subheader("Minimax Trade-off Curve")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Level"], y=df["mAP"],
        name="mAP (Utility)",
        mode="lines+markers",
        marker=dict(size=10, symbol="circle"),
        line=dict(color=UTILITY_COLOR, width=3),
        yaxis="y1",
        hovertemplate="<b>%{x}</b><br>mAP: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["Level"], y=df["Rank-1"],
        name="Rank-1 (Utility)",
        mode="lines+markers",
        marker=dict(size=10, symbol="square"),
        line=dict(color=UTILITY_COLOR_LIGHT, width=2, dash="dash"),
        yaxis="y1",
        hovertemplate="<b>%{x}</b><br>Rank-1: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["Level"], y=df["Privacy Leak (%)"],
        name="Privacy Leakage (Top-1 %)",
        mode="lines+markers",
        marker=dict(size=10, symbol="triangle-up", color=PRIVACY_COLOR),
        line=dict(color=PRIVACY_COLOR, width=3),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Leak: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(title="Utility (0-1)", range=[0, 1], side="left",
                   title_font=dict(color=UTILITY_COLOR), tickfont=dict(color=UTILITY_COLOR)),
        yaxis2=dict(title="Privacy Leakage (%)", range=[0, 105], side="right", overlaying="y",
                    title_font=dict(color=PRIVACY_COLOR), tickfont=dict(color=PRIVACY_COLOR)),
        xaxis=dict(title="Anonymization Technique"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=450, margin=dict(t=40, b=40),
        hovermode="x unified",
        font=dict(color="#1e293b"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Scatter: Privacy vs Utility ---
    st.subheader("Privacy vs. Utility Scatter")
    scatter_df = df[df["mAP"].notna() & df["Privacy Leak (%)"].notna()].copy()

    if not scatter_df.empty:
        max_mAP = scatter_df["mAP"].max()

        fig2 = go.Figure()

        fig2.add_shape(
            type="rect",
            x0=max_mAP * 0.4, x1=max_mAP * 1.15,
            y0=-5, y1=25,
            fillcolor="rgba(34,197,94,0.08)",
            line=dict(color="rgba(34,197,94,0.3)", width=1, dash="dot"),
            layer="below",
        )
        fig2.add_annotation(
            x=max_mAP * 0.77, y=3,
            text="Ideal Zone (High Utility, Low Leakage)",
            showarrow=False,
            font=dict(size=10, color="#16a34a"),
        )

        text_positions = []
        for _, r in scatter_df.iterrows():
            if r["Privacy Leak (%)"] > 50:
                text_positions.append("bottom right")
            elif r["mAP"] < 0.05:
                text_positions.append("top right")
            else:
                text_positions.append("top left")

        for i, (_, r) in enumerate(scatter_df.iterrows()):
            fig2.add_trace(go.Scatter(
                x=[r["mAP"]], y=[r["Privacy Leak (%)"]],
                mode="markers+text",
                marker=dict(size=14, color=LEVEL_COLORS[r["Level"]], line=dict(width=1.5, color="white")),
                text=[r["Level"]],
                textposition=text_positions[i],
                textfont=dict(size=12, color=LEVEL_COLORS[r["Level"]]),
                name=r["Level"],
                hovertemplate=(
                    f"<b>{r['Level']}</b><br>"
                    f"mAP: {r['mAP']:.4f}<br>"
                    f"Privacy Leak: {r['Privacy Leak (%)']:.2f}%<br>"
                    f"Protection: {r['Privacy Protection (%)']:.2f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        fig2.update_layout(
            template="plotly_white",
            xaxis=dict(title="Utility (mAP) -- higher is better",
                       range=[-0.03, max_mAP * 1.15]),
            yaxis=dict(title="Privacy Leakage (%) -- lower is better",
                       range=[-5, 110]),
            height=420, margin=dict(t=20, b=40),
            font=dict(color="#1e293b"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- Level Detail Cards ---
    st.subheader("Level Detail")
    selected = st.selectbox("Select anonymization level:", LEVELS, index=0, key="tradeoff_select")
    row = df[df["Level"] == selected].iloc[0]
    baseline_row = df[df["Level"] == "Baseline"].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)

    if pd.notna(row.get("mAP")):
        delta = None if selected == "Baseline" else f"{row['mAP'] - baseline_row['mAP']:+.4f}"
        c1.metric("mAP", f"{row['mAP']:.4f}", delta=delta)

    if pd.notna(row.get("Rank-1")):
        delta = None if selected == "Baseline" else f"{row['Rank-1'] - baseline_row['Rank-1']:+.4f}"
        c2.metric("Rank-1", f"{row['Rank-1']:.4f}", delta=delta)

    if pd.notna(row.get("Privacy Leak (%)")):
        delta = None if selected == "Baseline" else f"{row['Privacy Leak (%)'] - 100:+.2f}%"
        c3.metric("Privacy Leak", f"{row['Privacy Leak (%)']:.2f}%",
                  delta=delta, delta_color="inverse")

    if pd.notna(row.get("Privacy Protection (%)")):
        c4.metric("Privacy Protection", f"{row['Privacy Protection (%)']:.2f}%",
                  help="100% - Privacy Leak. Higher = better anonymization.")

    if pd.notna(row.get("Effectiveness Index")):
        c5.metric("Effectiveness Index", f"{row['Effectiveness Index']:.2f}",
                  help="Privacy Protection (%) / Utility Cost (%). Higher = more privacy gained per unit of utility lost.")



def render_visual_tab(df):
    st.subheader("Side-by-Side Anonymization Comparison")

    filenames = get_query_filenames()
    if not filenames:
        st.warning("No query images found in data/raw/Market-1501/query/")
        return

    identities = get_identities(filenames)
    identity_list = sorted(identities.keys())

    if st.session_state.get("_randomize_id"):
        st.session_state["_randomize_id"] = False
        st.session_state["visual_id"] = random.choice(identity_list)

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        selected_id = st.selectbox(
            "Select identity:",
            identity_list,
            format_func=lambda x: f"Person {x} ({len(identities[x])} images)",
            key="visual_id",
        )
    with col_ctrl2:
        st.write("")
        st.write("")
        if st.button("Random Identity", use_container_width=True):
            st.session_state["_randomize_id"] = True
            st.rerun()

    id_images = identities[selected_id]
    if len(id_images) > 1:
        img_idx = st.slider("Image variant:", 0, len(id_images) - 1, 0, key="img_slider")
    else:
        img_idx = 0

    chosen_file = id_images[img_idx]
    st.caption(f"Showing: `{chosen_file}`")

    cols = st.columns(len(LEVELS))
    for i, level in enumerate(LEVELS):
        dataset_dir = DATASET_DIRS[level]
        img_path = os.path.join(dataset_dir, "query", chosen_file)
        with cols[i]:
            st.markdown(f"**{LEVEL_LABELS[level]}**")
            if os.path.exists(img_path):
                st.image(Image.open(img_path), use_container_width=True)
            else:
                st.info("N/A")


def render_metrics_tab(df):
    # --- Rank Progression (CMC curves per level) ---
    st.subheader("Rank Progression (CMC Curves)")
    ranks = [1, 5, 10, 20]
    rank_labels = ["Rank-1", "Rank-5", "Rank-10", "Rank-20"]

    fig_cmc = go.Figure()
    for _, row in df.iterrows():
        if pd.isna(row.get("Rank-1")):
            continue
        vals = [row.get(m, 0) or 0 for m in rank_labels]
        fig_cmc.add_trace(go.Scatter(
            x=ranks, y=vals,
            name=row["Level"],
            mode="lines+markers",
            marker=dict(size=8),
            line=dict(color=LEVEL_COLORS[row["Level"]], width=2.5),
            hovertemplate=f"<b>{row['Level']}</b><br>Rank-%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig_cmc.update_layout(
        template="plotly_white",
        xaxis=dict(title="Rank (k)", tickvals=ranks, ticktext=[f"R-{r}" for r in ranks]),
        yaxis=dict(title="Accuracy", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=400, margin=dict(t=40, b=40),
        font=dict(color="#1e293b"),
    )
    st.plotly_chart(fig_cmc, use_container_width=True)

    st.divider()

    # --- Full data table with computed columns ---
    st.subheader("Full Results Table")
    table_cols = ["Level", "Label", "mAP", "Rank-1", "Rank-5", "Rank-10", "Rank-20",
                  "Privacy Leak (%)", "Privacy Protection (%)", "mAP % of Baseline", "Effectiveness Index"]
    available_cols = [c for c in table_cols if c in df.columns]
    display_df = df[available_cols].copy()

    format_dict = {}
    for col in ["mAP", "Rank-1", "Rank-5", "Rank-10", "Rank-20"]:
        if col in display_df.columns:
            format_dict[col] = "{:.4f}"
    for col in ["Privacy Leak (%)", "Privacy Protection (%)", "mAP % of Baseline"]:
        if col in display_df.columns:
            format_dict[col] = "{:.2f}"
    if "Effectiveness Index" in display_df.columns:
        format_dict["Effectiveness Index"] = "{:.2f}"

    styled = display_df.style.format(format_dict, na_rep="--").set_properties(**{"color": "#1e293b"})
    if "mAP" in display_df.columns and "Rank-1" in display_df.columns:
        styled = styled.background_gradient(subset=["mAP", "Rank-1"], cmap="Blues", vmin=0, vmax=1)
    if "Privacy Leak (%)" in display_df.columns:
        styled = styled.background_gradient(subset=["Privacy Leak (%)"], cmap="Reds", vmin=0, vmax=100)
    if "Privacy Protection (%)" in display_df.columns:
        styled = styled.background_gradient(subset=["Privacy Protection (%)"], cmap="Greens", vmin=0, vmax=100)

    st.dataframe(styled, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="ANON4REID: Privacy-Utility Trade-off",
        layout="wide",
    )
    inject_css()

    baseline_data = load_json("results_baseline.json")
    utility_data = load_json("results_utility.json")
    privacy_data = load_json("results_privacy.json")

    df = build_unified_data(baseline_data, utility_data, privacy_data)

    if df.empty or df["mAP"].isna().all():
        st.error(
            "No evaluation data found. Run the evaluation scripts first:\n\n"
            "1. `python src/baseline_osnet.py`\n"
            "2. `python src/evaluate_utility.py`\n"
            "3. `python src/train_attacker.py`\n"
            "4. `python src/evaluate_privacy.py`"
        )
        return

    render_header(df)

    tab1, tab2, tab3 = st.tabs([
        "Trade-off Analysis",
        "Visual Comparison",
        "Detailed Metrics",
    ])

    with tab1:
        render_tradeoff_tab(df)
    with tab2:
        render_visual_tab(df)
    with tab3:
        render_metrics_tab(df)


if __name__ == "__main__":
    main()
