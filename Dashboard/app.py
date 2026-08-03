"""
Shield Dashboard -- interactive early-warning system for university staff cuts.

Public Streamlit app built on the team's pipeline (jason_eda, Luigi_Modeling,
Luigi_RF_Interpretation). Loads the artifacts from train_and_export.py; the
final model is a Random Forest (test PR-AUC 0.345 on 2022).
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent      # the Dashboard/ folder, wherever it's launched from
ART = BASE / "artifacts"
DATA = BASE / "Data"
TARGET = "TARGET_STAFF_CUT"
VULN = "TARGET_VULNERABLE_TOTAL"

st.set_page_config(page_title="Shield Dashboard", layout="wide")


@st.cache_data
def panel():
    for p in (DATA / "master_panel_df.csv", Path("../Data/master_panel_df.csv")):
        if p.exists():
            return pd.read_csv(p, low_memory=False)
    return None

@st.cache_data
def art(name):
    p = ART / name
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def metrics():
    p = ART / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None

@st.cache_resource
def model_bundle():
    import joblib
    p = ART / "rf_model.joblib"
    return joblib.load(p) if p.exists() else None


M = metrics()
st.sidebar.title("Shield Dashboard")
st.sidebar.caption("Predicting instructional staff cuts at public four-year "
                   "universities from IPEDS data (2014-2024).")
page = st.sidebar.radio("Section", [
    "Overview", "Exploratory analysis", "Model comparison",
    "Hyperparameter tuning", "Random Forest interpretation",
    "Risk watchlist", "Risk scorer"])


# ============================ OVERVIEW ============================
if page == "Overview":
    st.title("Shield Dashboard")
    st.markdown(
        "An early-warning system that predicts whether a public four-year "
        "university will **cut instructional FTE staff by 5% or more over two "
        "years**, using only public IPEDS finance, enrollment, and staffing data. "
        "It is a **screening tool** that ranks institutions by risk -- not a "
        "decision-maker.")
    if M:
        c1, c2, c3 = st.columns(3)
        c1.metric("Random Forest test PR-AUC (2022)", f"{M['test_pr_auc']:.3f}",
                  f"{M['test_pr_auc'] / M['base_rate']:.2f}x base rate")
        c2.metric("Test ROC-AUC", f"{M['test_roc_auc']:.3f}")
        c3.metric("Base rate", f"{M['base_rate']:.1%}")
        st.caption(f"Cross-validation winner: {M['cv_winner']} "
                   f"(paired t-test vs {M['cv_runner_up']}, p = {M['paired_t_pvalue']:.3f}). "
                   "Random Forest was selected on the held-out 2022 test year.")
    else:
        st.info("Run train_and_export.py to generate artifacts, then reload.")


# ======================= EXPLORATORY ANALYSIS =======================
elif page == "Exploratory analysis":
    st.title("Exploratory analysis")
    df = panel()
    if df is None:
        st.warning("Place master_panel_df.csv in ./data to enable EDA.")
    else:
        st.subheader("Records by year")
        yc = df["YEAR"].value_counts().sort_index().reset_index()
        yc.columns = ["YEAR", "rows"]
        st.plotly_chart(px.bar(yc, x="YEAR", y="rows"), use_container_width=True)

        st.subheader("Target distribution")
        tgt = TARGET if TARGET in df.columns else VULN
        counts = df[tgt].dropna().astype(int).map({0: "Stable/Gained", 1: "Vulnerable"}).value_counts()
        st.plotly_chart(px.bar(counts, labels={"value": "count", "index": tgt}),
                        use_container_width=True)

        st.subheader("Staffing-variable correlations")
        staff = [c for c in ["SFTETOTL", "SALTOTL", "SFTEINST", "SFTEPSTC",
                             "SFTESRVC", "SALPROF", "SALASSC"] if c in df.columns]
        if len(staff) >= 2:
            C = df[staff].apply(pd.to_numeric, errors="coerce").corr()
            st.plotly_chart(px.imshow(C, text_auto=".2f", color_continuous_scale="RdBu_r",
                                      zmin=-1, zmax=1), use_container_width=True)

        st.subheader("Total staff vs total salary")
        if {"SFTETOTL", "SALTOTL"}.issubset(df.columns):
            tgt = VULN if VULN in df.columns else TARGET
            sc = df.dropna(subset=["SFTETOTL", "SALTOTL", tgt]).copy()
            sc["status"] = sc[tgt].astype(int).map({0: "Stable/Gained", 1: "Vulnerable"})
            fig = px.scatter(sc, x="SFTETOTL", y="SALTOTL", color="status",
                             log_x=True, log_y=True, opacity=0.45,
                             color_discrete_map={"Stable/Gained": "green", "Vulnerable": "red"})
            st.plotly_chart(fig, use_container_width=True)


# ========================= MODEL COMPARISON =========================
elif page == "Model comparison":
    st.title("Model comparison")
    comp = art("model_comparison.csv"); folds = art("fold_scores.csv")
    if comp is None:
        st.warning("Run train_and_export.py first.")
    else:
        st.subheader("10-fold cross-validated PR-AUC")
        if folds is not None:
            order = folds.median().sort_values().index
            fig = go.Figure()
            for m in order:
                fig.add_trace(go.Box(x=folds[m], name=m, orientation="h",
                                     marker_color="#7bc47f" if m == "Random Forest" else "#a8c4f0"))
            if M:
                fig.add_vline(x=M["base_rate"], line_dash="dash", line_color="red",
                              annotation_text="base rate")
            fig.update_layout(showlegend=False, xaxis_title="PR-AUC", height=420)
            st.plotly_chart(fig, use_container_width=True)
            if M:
                st.caption(f"Winner {M['cv_winner']} vs runner-up {M['cv_runner_up']}: "
                           f"paired t-test p = {M['paired_t_pvalue']:.3f} "
                           "(p < 0.05 means the gap is consistent across folds, not noise).")

        st.subheader("PR-AUC and ROC-AUC by split")
        st.dataframe(comp.set_index(comp.columns[0]), use_container_width=True)


# ===================== HYPERPARAMETER TUNING =====================
elif page == "Hyperparameter tuning":
    st.title("Hyperparameter tuning")
    bp = art("best_params.csv")
    if bp is None:
        st.warning("Run train_and_export.py first.")
    else:
        st.subheader("Best configuration per model")
        st.caption("Selected by 10-fold CV PR-AUC via grid search (refit on full training set).")
        st.dataframe(bp, use_container_width=True, hide_index=True)

        st.subheader("Random Forest tuning curves")
        c1, c2, c3 = st.columns(3)
        mtry, ntree, oob = art("rf_mtry_sweep.csv"), art("rf_ntree_sweep.csv"), art("rf_oob_curve.csv")
        if mtry is not None:
            c1.plotly_chart(px.line(mtry, x="max_features", y="cv_pr_auc", markers=True,
                                    title="mtry sweep"), use_container_width=True)
        if ntree is not None:
            c2.plotly_chart(px.line(ntree, x="n_estimators", y="cv_pr_auc", markers=True,
                                    title="trees vs CV PR-AUC"), use_container_width=True)
        if oob is not None:
            fig = px.line(oob, x="n_estimators", y="oob_pr_auc", markers=True,
                          title="OOB PR-AUC (plateaus ~300)")
            fig.add_vline(x=500, line_dash="dash", annotation_text="chosen: 500")
            c3.plotly_chart(fig, use_container_width=True)
        st.caption("More trees stop helping past ~300; 500 is chosen for a stable margin. "
                   "The OOB curve gives this 'for free' without a validation set.")


# =================== RANDOM FOREST INTERPRETATION ===================
elif page == "Random Forest interpretation":
    st.title("Random Forest interpretation")
    imp = art("feature_importances.csv")
    if imp is None:
        st.warning("Run train_and_export.py first.")
    else:
        st.subheader("Top feature importances (mean decrease in impurity)")
        top = imp.head(12).sort_values("importance")
        colors = ["#7bc47f" if "ENROLL" in f else "#a8c4f0" for f in top["feature"]]
        fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h",
                               marker_color=colors))
        fig.update_layout(xaxis_title="mean decrease in impurity", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Enrollment-derived features (green) rank among the strongest drivers, "
                   "supporting the project's emphasis on enrollment trajectory.")

        if M and M.get("confusion_matrix"):
            st.subheader("Confusion matrix (2022 test year)")
            cm = np.array(M["confusion_matrix"])
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            x=["pred no-cut", "pred cut"], y=["actual no-cut", "actual cut"])
            st.plotly_chart(fig, use_container_width=True)


# ========================= RISK WATCHLIST =========================
elif page == "Risk watchlist":
    st.title("Risk watchlist")
    preds = art("test_predictions.csv")
    if preds is None:
        st.warning("Run train_and_export.py first.")
    else:
        st.markdown("Institutions ranked by predicted two-year staff-cut risk on the "
                    "most recent scored year (2022) -- the provost-facing view: **where "
                    "to look first**, not a verdict.")

        id_cols = [c for c in ["UNITID", "YEAR"] if c in preds.columns]
        table = preds[id_cols + ["y_proba"]].copy() if id_cols else preds[["y_proba"]].copy()
        if TARGET in preds.columns:
            table["actual_outcome"] = preds[TARGET].map({0: "no cut", 1: "cut"})
        table = table.rename(columns={"y_proba": "risk_score"}).sort_values(
            "risk_score", ascending=False).reset_index(drop=True)
        table.index += 1

        c1, c2 = st.columns([1, 3])
        top_n = c1.slider("How many to flag", 10, 100, 25, step=5)
        thresh = c2.slider("Risk threshold", 0.0, 1.0, 0.50, 0.05)

        flagged = table[table["risk_score"] >= thresh]
        c1.metric("Institutions flagged", len(flagged))
        if "actual_outcome" in table.columns and len(flagged):
            hit = (flagged["actual_outcome"] == "cut").mean()
            c2.metric("Precision among flagged",
                      f"{hit:.0%}",
                      f"vs {(table['actual_outcome'] == 'cut').mean():.0%} base rate")

        st.subheader(f"Top {top_n} highest-risk institutions")
        show = table.head(top_n).copy()
        show["risk_score"] = show["risk_score"].round(3)
        st.dataframe(
            show.style.background_gradient(subset=["risk_score"], cmap="Reds"),
            use_container_width=True, height=520)

        st.download_button("Download full ranked list (CSV)",
                           table.to_csv().encode(), "risk_watchlist.csv", "text/csv")


# ========================== RISK SCORER ==========================
elif page == "Risk scorer":
    st.title("Interactive risk scorer")
    bundle = model_bundle(); preds = art("test_predictions.csv")
    if bundle is None or preds is None:
        st.warning("Run train_and_export.py first.")
    else:
        model, feats = bundle["model"], bundle["feature_names"]
        st.markdown("Pick an institution, then **stress-test** it by adjusting key "
                    "drivers to watch the two-year risk score respond.")

        id_cols = [c for c in ["UNITID", "YEAR"] if c in preds.columns]
        labels = (preds[id_cols].astype(str).agg(" - ".join, axis=1)
                  if id_cols else preds.index.astype(str))
        default = int(np.argmax(preds["y_proba"].values)) if "y_proba" in preds else 0
        pick = st.selectbox("Institution (2022 test set)", labels.tolist(), index=default)
        x = preds.iloc[labels.tolist().index(pick)][feats].astype(float).copy()

        st.subheader("Stress test (values are standardized; 0 = training-year average)")
        levers = [f for f in ["STATE_APPROP_SHARE", "ENROLL_YOY", "OPERATING_MARGIN",
                              "ADMIN_INTENSITY", "REVENUE_PER_STUDENT"] if f in feats]
        for c, f in zip(st.columns(len(levers)), levers):
            x[f] = c.slider(f, float(x[f]) - 2, float(x[f]) + 2, float(x[f]), 0.1)

        proba = float(model.predict_proba(x.values.reshape(1, -1))[:, 1][0])
        st.plotly_chart(go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100,
            title={"text": "Two-year staff-cut risk (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#2c7fb8"},
                   "steps": [{"range": [0, 33], "color": "#DFF0D8"},
                             {"range": [33, 66], "color": "#FCF8E3"},
                             {"range": [66, 100], "color": "#F2DEDE"}]})),
            use_container_width=True)

        with st.expander("Why this score? (SHAP)"):
            try:
                import shap
                sv = shap.TreeExplainer(model).shap_values(x.values.reshape(1, -1))
                sv = sv[1] if isinstance(sv, list) else sv
                contrib = pd.DataFrame({"feature": feats, "shap": np.ravel(sv)})
                contrib = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index).head(12)
                st.plotly_chart(px.bar(contrib.sort_values("shap"), x="shap", y="feature",
                                       orientation="h", color="shap",
                                       color_continuous_scale="RdBu_r"),
                                use_container_width=True)
            except Exception as e:
                st.info(f"SHAP unavailable: {e}")
