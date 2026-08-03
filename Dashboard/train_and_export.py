"""
Train the models and export every artifact the Shield Dashboard needs.

Faithful to the team's notebooks:
  * Luigi_Modeling.ipynb        -> 7-model comparison, PR-AUC CV, tuning
  * Luigi_RF_Interpretation.ipynb -> final Random Forest, importances, case profile

Run once, locally, after the preprocessing notebook has written ../Data/splits/.
Produces a self-contained ./artifacts folder the public app loads (no retraining
on the server). The full grid search -- SVM especially -- is the slow part;
set FULL_SEARCH = False to export only the final Random Forest and skip it.
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
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             confusion_matrix, precision_recall_curve, roc_curve)

FULL_SEARCH = True
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


# ---- the 7 models (as in Luigi_Modeling) ----
models, best_params, fold_scores = {}, {}, {}

lr = Pipeline([("sc", StandardScaler()),
               ("m", LogisticRegression(max_iter=3000, class_weight="balanced"))])
lr.fit(X_train, y_train)
models["Logistic Regression"] = lr
best_params["Logistic Regression"] = "-- (no hyperparameters)"

# The final Random Forest is always built (fast); the rest only under FULL_SEARCH.
p = X_train.shape[1]
if FULL_SEARCH:
    g_glmnet = grid_search(
        Pipeline([("sc", StandardScaler()),
                  ("m", LogisticRegression(solver="saga", penalty="elasticnet",
                                           max_iter=4000, class_weight="balanced"))]),
        {"m__l1_ratio": [0, 0.25, 0.5, 0.75, 1], "m__C": np.logspace(-2, 1, 7)})
    models["Penalized Logistic Regression"] = g_glmnet.best_estimator_
    best_params["Penalized Logistic Regression"] = str(g_glmnet.best_params_)

    g_knn = grid_search(
        Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier())]),
        {"m__n_neighbors": list(range(1, 26, 2))})
    models["k-Nearest Neighbors"] = g_knn.best_estimator_
    best_params["k-Nearest Neighbors"] = str(g_knn.best_params_)

    g_svm = grid_search(
        Pipeline([("sc", StandardScaler()), ("m", SVC(probability=True))]),
        {"m__C": 2.0 ** np.arange(-3, 3), "m__gamma": ["scale", 0.001, 0.006]})
    models["Support Vector Machine"] = g_svm.best_estimator_
    best_params["Support Vector Machine"] = str(g_svm.best_params_)

    try:
        from xgboost import XGBClassifier
        pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        g_gbm = grid_search(
            XGBClassifier(eval_metric="aucpr", n_jobs=-1, random_state=SEED,
                          scale_pos_weight=pos_weight, verbosity=0),
            {"max_depth": [3, 5], "learning_rate": [0.05, 0.1],
             "n_estimators": [200, 400], "min_child_weight": [5, 10],
             "subsample": [0.8], "colsample_bytree": [0.7]})
        models["XGBoost"] = g_gbm.best_estimator_
        best_params["XGBoost"] = str(g_gbm.best_params_)
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        g_lgbm = grid_search(
            LGBMClassifier(scale_pos_weight=pos_weight, random_state=SEED,
                           verbose=-1, n_jobs=-1),
            {"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.1],
             "n_estimators": [200, 400], "min_child_samples": [20, 40]})
        models["LightGBM"] = g_lgbm.best_estimator_
        best_params["LightGBM"] = str(g_lgbm.best_params_)
    except ImportError:
        pass

    # RF mtry sweep (for the tuning page)
    rf_grid = {"max_features": sorted(set([2, int(np.sqrt(p)), int(p / 10),
                                           int(p / 5), int(p / 3)]))}
    g_rf = grid_search(
        RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=10,
                               class_weight="balanced", n_jobs=-1, random_state=SEED),
        rf_grid)
    best_params["Random Forest"] = str(g_rf.best_params_)
    pd.DataFrame({"max_features": [pp["max_features"] for pp in g_rf.cv_results_["params"]],
                  "cv_pr_auc": g_rf.cv_results_["mean_test_score"]}
                 ).sort_values("max_features").to_csv(OUT / "rf_mtry_sweep.csv", index=False)

# ---- final Random Forest (Luigi_RF_Interpretation config) ----
rf = RandomForestClassifier(n_estimators=500, max_features=15, max_depth=12,
                            min_samples_leaf=10, class_weight="balanced",
                            n_jobs=-1, random_state=SEED)
rf.fit(X_train, y_train)
models["Random Forest"] = rf
best_params.setdefault("Random Forest", "max_features=15, max_depth=12, min_samples_leaf=10")

# ---- ntree + OOB tuning curves ----
ntree = []
for n in [100, 200, 300, 400, 500, 800]:
    m = RandomForestClassifier(n_estimators=n, max_depth=12, min_samples_leaf=10,
                               class_weight="balanced", n_jobs=-1, random_state=SEED)
    ntree.append((n, float(cross_val_score(m, X_train, y_train, cv=CV, scoring=SCORING).mean())))
pd.DataFrame(ntree, columns=["n_estimators", "cv_pr_auc"]).to_csv(OUT / "rf_ntree_sweep.csv", index=False)

oob = []
for n in [25, 50, 100, 200, 300, 400, 500, 625]:
    m = RandomForestClassifier(n_estimators=n, max_features=15, max_depth=12,
                               min_samples_leaf=10, class_weight="balanced",
                               oob_score=True, n_jobs=-1, random_state=SEED).fit(X_train, y_train)
    op = m.oob_decision_function_[:, 1]; ok = ~np.isnan(op)
    oob.append((n, float(average_precision_score(y_train[ok], op[ok]))))
pd.DataFrame(oob, columns=["n_estimators", "oob_pr_auc"]).to_csv(OUT / "rf_oob_curve.csv", index=False)

# ---- model comparison: fold PR-AUC + split stats + paired t-test ----
for name, m in models.items():
    fold_scores[name] = cross_val_score(m, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
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
best_name = means.idxmax(); runner = means.drop(best_name).idxmax()
_, pval = stats.ttest_rel(fold_scores[best_name], fold_scores[runner])

pd.DataFrame([(k, best_params.get(k, "-"), round(fold_scores[k].mean(), 3)) for k in models],
             columns=["Model", "Best hyperparameters found", "CV PR-AUC"]
             ).to_csv(OUT / "best_params.csv", index=False)

# ---- final RF: metrics, curves, importances, predictions, top-risk case ----
proba_test = rf.predict_proba(X_test)[:, 1]
prec, rec, _ = precision_recall_curve(y_test, proba_test)
fpr, tpr, _ = roc_curve(y_test, proba_test)
cm = confusion_matrix(y_test, rf.predict(X_test))
metrics = {
    "final_model": "Random Forest",
    "test_pr_auc": float(average_precision_score(y_test, proba_test)),
    "test_roc_auc": float(roc_auc_score(y_test, proba_test)),
    "base_rate": float(y_test.mean()),
    "cv_winner": best_name, "cv_runner_up": runner, "paired_t_pvalue": float(pval),
    "confusion_matrix": cm.tolist(),
    "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
    "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importance.head(20).rename_axis("feature").reset_index(name="importance").to_csv(
    OUT / "feature_importances.csv", index=False)

test_out = test.copy(); test_out["y_proba"] = proba_test
if (DATA / "test_raw.csv").exists():
    raw = pd.read_csv(DATA / "test_raw.csv")
    for k in ["UNITID", "YEAR"]:
        if k in raw.columns:
            test_out[k] = raw[k].values
test_out.to_csv(OUT / "test_predictions.csv", index=False)

joblib.dump({"model": rf, "feature_names": list(X_train.columns)}, OUT / "rf_model.joblib")
print("artifacts written to", OUT.resolve())
print(f"RF test PR-AUC = {metrics['test_pr_auc']:.3f} | CV winner = {best_name} (p={pval:.3f})")
