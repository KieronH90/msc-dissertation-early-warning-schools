# MSc Dissertation: Early Warning System for Secondary School Performance

This repository contains the code, model artefacts and reproducibility materials for my MSc Data Science dissertation:

**Developing an Early Warning System for Secondary School Performance: Evaluating Machine Learning Models Against Persistence Baselines Using Time-Lagged Census Data**

## Project Summary

This project develops an interpretable early warning framework for forecasting institutional performance in English secondary schools using time-lagged census data.

The modelling strategy compares:

- Naive persistence baseline
- OLS delta regression
- XGBoost delta regression
- XGBoost underperformance classifier
- XGBoost overperformance classifier

The project uses out-of-time validation to simulate realistic forward-looking deployment.

## Evaluation Design

- Data period: 2016–2023
- Training period: 2017–2019
- Excluded target years: 2020–2021 due to Centre Assessed Grades and Teacher Assessed Grades
- Test period: 2022–2023
- Primary regression target: year-on-year change in Progress 8
- Operational risk threshold: Progress 8 ≤ −0.5

## Repository Structure

- `notebooks/` — Notebook pipeline used for data processing, feature engineering, model training and evaluation
- `app/` — Streamlit dashboard prototype
- `models/` — Final trained model artefacts and feature lists
- `data/processed/` — Final processed modelling artefacts, where included
- `outputs/figures/` — Figures used in the dissertation
- `outputs/tables/` — Result tables used in the dissertation
- `docs/` — Dissertation PDF and supporting documentation

## Notebook Execution Order

The notebooks should be run in the following order:

1. `00_data_setup.ipynb`
2. `01_data_ingestion.ipynb`
3. `02_academy_mapper.ipynb`
4. `03_feature_engineering.ipynb`
5. `04_model_training.ipynb`
6. `05_check_features.ipynb`
7. `06_final_evaluation.ipynb`

## Dashboard

The Streamlit dashboard prototype is located at:

`app/app.py`

The dashboard is intended as a decision-support prototype. It presents model outputs, risk probabilities, local explanation and scenario exploration. It should not be interpreted as an automated decision system.

To run the dashboard locally, install the required packages and run:

`streamlit run app/app.py`

Some file paths may need to be updated depending on the local environment.

## Model Artefacts

The final model artefacts are stored in the `models/` folder:

- `delta_xgb_model.joblib`
- `delta_model_features.joblib`
- `delta_ols_model.joblib`
- `risk_xgb_classifier.joblib`
- `risk_model_features.joblib`
- `overperf_xgb_classifier.joblib`
- `overperf_model_features.joblib`

These artefacts correspond to the final modelling pipeline described in the dissertation.

## Data Notes

The project uses school-level institutional data derived from publicly available Department for Education datasets.

Where included, processed modelling artefacts are stored in:

`data/processed/`

The key processed files are:

- `school_panel_final.parquet`
- `model_input.parquet`
- `urn_mapping.pkl`

No individual pupil-level data is included in this repository.

## Reproducibility Notes

The original project was developed in a local Windows environment. Some file paths may need to be updated before running the notebooks or dashboard on another machine.

The 2020–2021 target years are excluded because Centre Assessed Grades and Teacher Assessed Grades introduced a structural break in the Progress 8 outcome distribution.

The notebook pipeline is intended to document the full workflow from data setup through to final evaluation.

## Ethical and Governance Note

This project uses institutional-level data only. Model outputs are intended for research and decision-support demonstration only.

The dashboard should not be interpreted as an automated decision-making system. Its outputs are intended to support professional judgement, not replace it.

## Dissertation

The final dissertation PDF is stored in:

`docs/`