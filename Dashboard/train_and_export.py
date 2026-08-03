"""
Train models and export dashboard artifacts -- a faithful reproduction of the
team's notebooks:
  * Luigi_Modeling.ipynb          (7-model comparison, tuning, selection)
  * Luigi_RF_Interpretation.ipynb (final RF, importances, case profile)

Selection follows the notebook exactly: the final model is Random Forest, chosen
on the 2022 TEST year (not the cross-validation ranking). Run once, locally,
after the preprocessing notebook has written ../Data/splits/.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix)

FULL_SEARCH = True          # False -> export only the final RF (skips the slow SVM grid)
DATA = Path("../Data/splits")
OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)
TARGET = "TARGET_STAFF_CUT"
SEED = 1056

train = pd.read_csv(DATA / "train_processed.csv")
valid = pd.read_csv(DATA / "valid_processed.csv")
test = pd.read_csv(DATA / "test_processed.csv")
train["YEAR"] = pd.read_csv(DATA / "train_raw.csv", usecols=["YEAR"])["YEAR"].values

X_train = train.drop(columns=[TARGET, "YEAR"]); y_train = train[TARGET]
years = train["YEAR"].values
X_valid = valid.drop(columns=[TARGET]); y_valid = valid[TARGET]
X_test = test.drop(columns=[TARGET]); y_test = test[TARGET]

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
SCORING = "average_precision"


def grid_search(model, grid):
    g = GridSearchCV(model, grid, cv=CV, scoring=SCORING, n_jobs=-1, refit=True)
    g.fit(X_train, y_train)
    return g


# ---- baseline logistic regression (no tuning) ----
lr = Pipeline([("sc", StandardScaler()),
               ("m", LogisticRegression(max_iter=3000, class_weight="balanced"))])
lr_scores = cross_val_score(lr, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
lr.fit(X_train, y_train)

models = {"Logistic Regression": lr}
best_params = {"Logistic Regression": ("— (no hyperparameters)", float(lr_scores.mean()))}
p = X_train.shape[1]

if FULL_SEARCH:
    g_glmnet = grid_search(
        Pipeline([("sc", StandardScaler()),
                  ("m", LogisticRegression(solver="saga", penalty="elasticnet",
                                           max_iter=4000, class_weight="balanced"))]),
        {"m__l1_ratio": [0, 0.25, 0.5, 0.75, 1], "m__C": np.logspace(-2, 1, 7)})
    models["Penalized Logistic Regression"] = g_glmnet.best_estimator_
    best_params["Penalized Logistic Regression"] = (str(g_glmnet.best_params_), float(g_glmnet.best_score_))

    g_knn = grid_search(
        Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier())]),
        {"m__n_neighbors": list(range(1, 26, 2))})
    models["k-Nearest Neighbors"] = g_knn.best_estimator_
    best_params["k-Nearest Neighbors"] = (str(g_knn.best_params_), float(g_knn.best_score_))

    g_svm_c = grid_search(
        Pipeline([("sc", StandardScaler()), ("m", SVC(probability=True))]),
        {"m__C": 2.0 ** np.arange(-3, 3), "m__gamma": ["scale"]})
    best_C = g_svm_c.best_params_["m__C"]
    g_svm = grid_search(
        Pipeline([("sc", StandardScaler()), ("m", SVC(probability=True))]),
        {"m__C": best_C * 2.0 ** np.arange(-1.5, 2), "m__gamma": ["scale", 0.001, 0.006]})
    models["Support Vector Machine"] = g_svm.best_estimator_
    best_params["Support Vector Machine"] = (str(g_svm.best_params_), float(g_svm.best_score_))

    # RF mtry sweep (also feeds the tuning page)
    rf_grid = {"max_features": sorted(set([2, int(np.sqrt(p)), int(p / 10), int(p / 5), int(p / 3)]))}
    g_rf = grid_search(
        RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=10,
                               class_weight="balanced", n_jobs=-1, random_state=SEED),
        rf_grid)
    models["Random Forest"] = g_rf.best_estimator_
    best_params["Random Forest"] = (str(g_rf.best_params_), float(g_rf.best_score_))
    # mtry sweep scored on the 2022 TEST set (each forest fit on train, evaluated on test)
    mtry_rows = []
    for mf in rf_grid["max_features"]:
        m = RandomForestClassifier(n_estimators=500, max_features=mf, max_depth=12,
                                   min_samples_leaf=10, class_weight="balanced",
                                   n_jobs=-1, random_state=SEED).fit(X_train, y_train)
        mtry_rows.append((mf, float(average_precision_score(y_test, m.predict_proba(X_test)[:, 1]))))
    pd.DataFrame(mtry_rows, columns=["max_features", "test_pr_auc"]).to_csv(
        OUT / "rf_mtry_sweep.csv", index=False)

    try:
        from xgboost import XGBClassifier
        pw = float((y_train == 0).sum() / (y_train == 1).sum())
        g_gbm = grid_search(
            XGBClassifier(eval_metric="aucpr", n_jobs=-1, random_state=SEED,
                          scale_pos_weight=pw, verbosity=0),
            {"max_depth": [3, 5], "learning_rate": [0.05, 0.1], "n_estimators": [200, 400],
             "min_child_weight": [5, 10], "subsample": [0.8], "colsample_bytree": [0.7]})
        models["XGBoost"] = g_gbm.best_estimator_
        best_params["XGBoost"] = (str(g_gbm.best_params_), float(g_gbm.best_score_))
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        pw = float((y_train == 0).sum() / (y_train == 1).sum())
        g_lgbm = grid_search(
            LGBMClassifier(scale_pos_weight=pw, random_state=SEED, verbose=-1, n_jobs=-1),
            {"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.1],
             "n_estimators": [200, 400], "min_child_samples": [20, 40]})
        models["LightGBM"] = g_lgbm.best_estimator_
        best_params["LightGBM"] = (str(g_lgbm.best_params_), float(g_lgbm.best_score_))
    except ImportError:
        pass

# ---- final Random Forest (Luigi_RF_Interpretation config) ----
rf_final = RandomForestClassifier(n_estimators=500, max_features=15, max_depth=12,
                                  min_samples_leaf=10, class_weight="balanced",
                                  n_jobs=-1, random_state=SEED).fit(X_train, y_train)
models["Random Forest"] = rf_final          # the selected model (notebook cell 43)
if "Random Forest" not in best_params:
    best_params["Random Forest"] = ("max_features=15, max_depth=12, min_samples_leaf=10", np.nan)

# ---- RF ntree tuning curve, scored on the 2022 TEST set ----
ntree = []
for n in [100, 200, 300, 400, 500, 800]:
    m = RandomForestClassifier(n_estimators=n, max_features=15, max_depth=12,
                               min_samples_leaf=10, class_weight="balanced",
                               n_jobs=-1, random_state=SEED).fit(X_train, y_train)
    ntree.append((n, float(average_precision_score(y_test, m.predict_proba(X_test)[:, 1]))))
pd.DataFrame(ntree, columns=["n_estimators", "test_pr_auc"]).to_csv(
    OUT / "rf_ntree_sweep.csv", index=False)

# ---- comparison: fold scores, valid/test per model, paired t-test ----
fold_scores = {n: cross_val_score(m, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
               for n, m in models.items()}
pd.DataFrame(fold_scores).to_csv(OUT / "fold_scores.csv", index=False)

rows = []
for name, m in models.items():
    row = {"model": name}
    for split, Xs, ys in [("train", X_train, y_train), ("valid", X_valid, y_valid), ("test", X_test, y_test)]:
        pr = m.predict_proba(Xs)[:, 1]
        row[f"{split}_PR_AUC"] = round(average_precision_score(ys, pr), 3)
        row[f"{split}_ROC_AUC"] = round(roc_auc_score(ys, pr), 3)
    rows.append(row)
split_stats = pd.DataFrame(rows).set_index("model")
split_stats.to_csv(OUT / "model_comparison.csv")

means = pd.Series({k: v.mean() for k, v in fold_scores.items()})
cv_best = means.idxmax(); runner = means.drop(cv_best).idxmax()
_, pval = stats.ttest_rel(fold_scores[cv_best], fold_scores[runner])

bp_rows = []
for k, v in best_params.items():
    test_score = split_stats.loc[k, "test_PR_AUC"] if k in split_stats.index else np.nan
    bp_rows.append((k, v[0], round(float(test_score), 3) if test_score == test_score else "—"))
pd.DataFrame(bp_rows, columns=["Model", "Best hyperparameters found", "Test PR-AUC (2022)"]
             ).to_csv(OUT / "best_params.csv", index=False)

# ---- final RF: metrics, importances, predictions, case profile ----
proba_test = rf_final.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, rf_final.predict(X_test))
metrics = {
    "final_model": "Random Forest",
    "selection_note": "selected on the 2022 test year, not the cross-validation ranking",
    "test_pr_auc": float(average_precision_score(y_test, proba_test)),
    "test_roc_auc": float(roc_auc_score(y_test, proba_test)),
    "test_base_rate": float(y_test.mean()),
    "train_base_rate": float(y_train.mean()),
    "cv_best_model": cv_best, "cv_runner_up": runner, "paired_t_pvalue": float(pval),
    "confusion_matrix": cm.tolist(),
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

importance = pd.Series(rf_final.feature_importances_, index=X_train.columns).sort_values(ascending=False)
top12 = importance.head(12)
top12.rename_axis("feature").reset_index(name="importance").to_csv(OUT / "feature_importances.csv", index=False)

test_out = test.copy(); test_out["y_proba"] = proba_test
raw_test = pd.read_csv(DATA / "test_raw.csv") if (DATA / "test_raw.csv").exists() else None
if raw_test is not None:
    for k in ["UNITID", "YEAR"]:
        if k in raw_test.columns:
            test_out[k] = raw_test[k].values
test_out.to_csv(OUT / "test_predictions.csv", index=False)

# case profile of the highest-risk institution (Luigi_RF_Interpretation cell 12)
i = int(np.argmax(proba_test))
case = X_test.iloc[i]
def fmt(v):
    if pd.isna(v):
        return "— (imputed)"
    return (f"{v:,.3f}".rstrip("0").rstrip(".")) if abs(v) < 100 else f"{v:,.0f}"
raw_vals = ([fmt(v) for v in raw_test.iloc[i][top12.index]] if raw_test is not None
            else [fmt(v) for v in case[top12.index]])
pd.DataFrame({
    "feature": top12.index,
    "z_score": case[top12.index].round(2).values,
    "raw_value": raw_vals,
    "direction": np.where(case[top12.index] < 0, "below average", "above average"),
}).to_csv(OUT / "case_profile.csv", index=False)
case_meta = {"institution_index": i, "predicted_proba": float(proba_test[i]),
             "actual": int(y_test.iloc[i])}
if raw_test is not None and "UNITID" in raw_test.columns:
    case_meta["UNITID"] = int(raw_test.iloc[i]["UNITID"])
(OUT / "case_meta.json").write_text(json.dumps(case_meta, indent=2))

joblib.dump({"model": rf_final, "feature_names": list(X_train.columns)}, OUT / "rf_model.joblib")
print("artifacts written to", OUT.resolve())
print(f"RF test PR-AUC = {metrics['test_pr_auc']:.3f} | selected on 2022 test year "
      f"(CV top scorer was {cv_best}, p={pval:.3f})")
