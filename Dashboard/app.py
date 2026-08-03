"""
Shield Dashboard -- interactive early-warning system for university staff cuts.

Public Streamlit app that mirrors the team's notebooks:
  jason_eda (EDA), Luigi_Modeling (7-model comparison + tuning),
  Luigi_RF_Interpretation (final Random Forest, importances, case profile).

The final model is Random Forest, selected on the 2022 test year (not the
cross-validation ranking), matching Luigi_Modeling cell 43.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent          # resolve everything relative to this file,
ART = BASE / "artifacts"              # so it works from any launch dir and on the cloud
DATA = BASE / "Data"
TARGET = "TARGET_STAFF_CUT"
VULN = "TARGET_VULNERABLE_TOTAL"

st.set_page_config(page_title="Shield Dashboard", layout="wide")


@st.cache_data
def panel():
    for p in (DATA / "master_panel_df.csv", BASE.parent / "Data" / "master_panel_df.csv"):
        if p.exists():
            return pd.read_csv(p, low_memory=False)
    return None

@st.cache_data
def art(name):
    p = ART / name
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def blob(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None

@st.cache_resource
def model_bundle():
    import joblib
    p = ART / "rf_model.joblib"
    return joblib.load(p) if p.exists() else None


M = blob("metrics.json")
st.sidebar.title("Shield Dashboard")
st.sidebar.caption("Predicting instructional staff cuts at public four-year "
                   "universities from IPEDS data (2014-2024).")
page = st.sidebar.radio("Section", [
    "Overview", "Model comparison",
    "Hyperparameter tuning", "Random Forest interpretation",
    "Risk scorer"])


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
                  f"{M['test_pr_auc'] / M['test_base_rate']:.2f}x base rate")
        c2.metric("Test ROC-AUC", f"{M['test_roc_auc']:.3f}")
        c3.metric("Base rate (2022)", f"{M['test_base_rate']:.1%}")
        st.caption(
            f"**Random Forest is the selected model, {M['selection_note']}.** "
            f"On cross-validation the top scorer was {M['cv_best_model']} "
            f"(paired t-test vs {M['cv_runner_up']}, p = {M['paired_t_pvalue']:.3f} -- "
            "not a significant gap), but Random Forest generalized best to the "
            "distribution-shifted 2022 test year, which is the basis for selection.")

        st.divider()
        st.header("Exploratory analysis")
        df = panel()
        if df is None:
            st.caption("Place master_panel_df.csv in ./Data to show the EDA charts.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Records by year")
                yc = df["YEAR"].value_counts().sort_index().reset_index()
                yc.columns = ["YEAR", "rows"]
                st.plotly_chart(px.bar(yc, x="YEAR", y="rows"), use_container_width=True)
            with col2:
                st.subheader("Target distribution")
                tgt = TARGET if TARGET in df.columns else VULN
                counts = (df[tgt].dropna().astype(int)
                          .map({0: "Stable/Gained", 1: "Vulnerable"}).value_counts())
                st.plotly_chart(px.bar(counts, labels={"value": "count", "index": tgt}),
                                use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Staffing-variable correlations")
                staff = [c for c in ["SFTETOTL", "SALTOTL", "SFTEINST", "SFTEPSTC",
                                     "SFTESRVC", "SALPROF", "SALASSC"] if c in df.columns]
                if len(staff) >= 2:
                    C = df[staff].apply(pd.to_numeric, errors="coerce").corr()
                    st.plotly_chart(px.imshow(C, text_auto=".2f", color_continuous_scale="RdBu_r",
                                              zmin=-1, zmax=1), use_container_width=True)
            with col4:
                st.subheader("Total staff vs total salary")
                if {"SFTETOTL", "SALTOTL"}.issubset(df.columns):
                    tgt = VULN if VULN in df.columns else TARGET
                    sc = df.dropna(subset=["SFTETOTL", "SALTOTL", tgt]).copy()
                    sc["status"] = sc[tgt].astype(int).map({0: "Stable/Gained", 1: "Vulnerable"})
                    fig = px.scatter(sc, x="SFTETOTL", y="SALTOTL", color="status",
                                     log_x=True, log_y=True, opacity=0.45,
                                     color_discrete_map={"Stable/Gained": "green",
                                                         "Vulnerable": "red"})
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run train_and_export.py to generate artifacts, then reload.")


# ========================= MODEL COMPARISON =========================
elif page == "Model comparison":
    st.title("Model comparison")
    comp = art("model_comparison.csv"); folds = art("fold_scores.csv")
    if comp is None or folds is None:
        st.warning("Run train_and_export.py first.")
    else:
        comp = comp.set_index(comp.columns[0])
        st.subheader("Cross-validated PR-AUC, with validation and test points")
        st.caption("Boxes are the 10 cross-validation folds on the training years. "
                   "The red diamond is 2021 validation, the blue triangle is 2022 test. "
                   "Random Forest (green) is chosen because its **test** point is highest, "
                   "even though other models score higher on cross-validation.")
        order = folds.median().sort_values().index.tolist()
        fig = go.Figure()
        for m in order:
            fig.add_trace(go.Box(x=folds[m], name=m, orientation="h", showlegend=False,
                                 marker_color="#7bc47f" if m == "Random Forest" else "#a8c4f0"))
            if m in comp.index:
                fig.add_trace(go.Scatter(x=[comp.loc[m, "valid_PR_AUC"]], y=[m], mode="markers",
                                         marker=dict(symbol="diamond", size=11, color="#d62728"),
                                         showlegend=False))
                fig.add_trace(go.Scatter(x=[comp.loc[m, "test_PR_AUC"]], y=[m], mode="markers",
                                         marker=dict(symbol="triangle-up", size=13, color="#1f77b4"),
                                         showlegend=False))
        if M:
            fig.add_vline(x=M["train_base_rate"], line_dash="dash", line_color="red",
                          annotation_text="baseline")
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="2021 validation",
                                 marker=dict(symbol="diamond", size=11, color="#d62728")))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="2022 test",
                                 marker=dict(symbol="triangle-up", size=13, color="#1f77b4")))
        fig.update_layout(xaxis_title="PR-AUC", height=460, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        if M:
            st.caption(f"Cross-validation top scorer {M['cv_best_model']} vs runner-up "
                       f"{M['cv_runner_up']}: paired t-test p = {M['paired_t_pvalue']:.3f} "
                       "(> 0.05, so the CV gap is not statistically significant).")

        st.subheader("PR-AUC and ROC-AUC by split")
        st.dataframe(comp, use_container_width=True)


# ===================== HYPERPARAMETER TUNING =====================
elif page == "Hyperparameter tuning":
    st.title("Hyperparameter tuning")
    bp = art("best_params.csv")
    if bp is None:
        st.warning("Run train_and_export.py first.")
    else:
        st.subheader("Best configuration per model")
        st.caption("Hyperparameters chosen by grid search; the score shown is each "
                   "model's PR-AUC on the held-out 2022 test set.")
        st.dataframe(bp, use_container_width=True, hide_index=True)

        st.subheader("Random Forest tuning curves (2022 test PR-AUC)")
        c1, c2 = st.columns(2)
        mtry, ntree = art("rf_mtry_sweep.csv"), art("rf_ntree_sweep.csv")
        if mtry is not None:
            c1.plotly_chart(px.line(mtry, x="max_features", y="test_pr_auc", markers=True,
                                    title="mtry sweep"), use_container_width=True)
        if ntree is not None:
            fig = px.line(ntree, x="n_estimators", y="test_pr_auc", markers=True,
                          title="trees vs test PR-AUC")
            fig.add_vline(x=500, line_dash="dash", annotation_text="chosen: 500")
            c2.plotly_chart(fig, use_container_width=True)
        st.caption("Both curves are scored on the 2022 test set. More trees stop helping "
                   "past ~300; 500 is chosen for a stable margin.")


# =================== RANDOM FOREST INTERPRETATION ===================
elif page == "Random Forest interpretation":
    st.title("Random Forest interpretation")
    imp = art("feature_importances.csv")
    if imp is None:
        st.warning("Run train_and_export.py first.")
    else:
        st.subheader("Top-12 feature importances (mean decrease in impurity)")
        top = imp.head(12).sort_values("importance")
        colors = ["#7bc47f" if "ENROLL" in f else "#a8c4f0" for f in top["feature"]]
        fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h",
                               marker_color=colors))
        fig.update_layout(xaxis_title="mean decrease in impurity", height=430)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Enrollment-derived features (green) rank among the strongest drivers, "
                   "supporting the project's emphasis on enrollment trajectory.")

        c1, c2 = st.columns(2)
        if M and M.get("confusion_matrix"):
            with c1:
                st.subheader("Confusion matrix (2022 test)")
                cm = np.array(M["confusion_matrix"])
                st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                          x=["pred no-cut", "pred cut"],
                                          y=["actual no-cut", "actual cut"]),
                                use_container_width=True)
        case, meta = art("case_profile.csv"), blob("case_meta.json")
        if case is not None:
            with c2:
                st.subheader("Highest-risk institution")
                if meta:
                    st.caption(f"Predicted P(cut) = {meta['predicted_proba']:.2f}, "
                               f"actual outcome = {'cut' if meta['actual'] else 'no cut'}"
                               + (f", UNITID {meta['UNITID']}" if "UNITID" in meta else ""))
                st.dataframe(case, use_container_width=True, hide_index=True)
                st.caption("Processed features are standardized on the training years: "
                           "0 = average institution, negative = below average.")


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

        st.subheader("Stress test (standardized values; 0 = training-year average)")
        levers = [f for f in ["STATE_APPROP_SHARE", "ENROLL_YOY", "OPERATING_MARGIN",
                              "ADMIN_INTENSITY", "REVENUE_PER_STUDENT"] if f in feats]
        for c, f in zip(st.columns(len(levers)), levers):
            x[f] = c.slider(f, float(x[f]) - 2, float(x[f]) + 2, float(x[f]), 0.1)

        proba = float(model.predict_proba(x.values.reshape(1, -1))[:, 1][0])
        st.plotly_chart(go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100,
            title={"text": "Two-year staff-cut risk (%)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2c7fb8"},
                   "steps": [{"range": [0, 33], "color": "#DFF0D8"},
                             {"range": [33, 66], "color": "#FCF8E3"},
                             {"range": [66, 100], "color": "#F2DEDE"}]})),
            use_container_width=True)
