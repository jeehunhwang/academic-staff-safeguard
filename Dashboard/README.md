# Shield Dashboard

Interactive early-warning system predicting instructional staff cuts at public
four-year universities from IPEDS data. Built on the team's pipeline:
jason_eda (EDA), Luigi_Modeling (7-model comparison + tuning), and
Luigi_RF_Interpretation (final Random Forest). Final model: Random Forest,
test PR-AUC 0.345 on 2022 (baseline 0.181).

## Build the artifacts (run once, locally)

1. Run the preprocessing notebook so it writes `../Data/splits/*_processed.csv`
   and `*_raw.csv`.
2. Generate model + dashboard artifacts:
   ```bash
   pip install -r requirements.txt
   python train_and_export.py
   ```
   Writes `./artifacts/`: model, model_comparison, fold_scores, best_params,
   RF tuning curves (mtry / ntree / OOB), feature importances, metrics, and
   test predictions. The full 7-model grid search is slow (SVM most of all);
   set `FULL_SEARCH = False` in the script to export only the Random Forest.
3. Copy `master_panel_df.csv` into `./data/` for the EDA page.

## Run locally
```bash
streamlit run app.py
```

## Publish for free (Streamlit Community Cloud)

1. Push this folder to a **public GitHub repo**, including `artifacts/` and
   `data/` (committed, so the hosted app can read them).
2. At https://share.streamlit.io , sign in with GitHub and point it at `app.py`.
3. It builds from `requirements.txt` and returns a public URL; pushes redeploy.

### Notes
- Keep committed files under ~100 MB. A 500-tree RF `.joblib` can be large;
  use `joblib.dump(..., compress=3)` or fewer trees if needed.
- All inputs are public IPEDS data, so there are no hosting privacy constraints.
- Pages: Overview, Exploratory analysis, Model comparison, Hyperparameter
  tuning, Random Forest interpretation, Risk scorer (live scoring + SHAP).
