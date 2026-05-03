# ==========================================================
# app.py — Trust School Priority Tool
# ==========================================================
#
# Purpose:
# Provide an interactive Streamlit-based decision-support
# prototype for exploring school-level risk estimates,
# reviewing key model drivers, and testing plausible
# what-if scenarios.
#
# Required inputs:
# - data/processed/school_panel_final.parquet
# - models/risk_xgb_classifier.joblib
# - models/risk_model_features.joblib
# - models/overperf_xgb_classifier.joblib
# - models/overperf_model_features.joblib
# - models/delta_xgb_model.joblib
# - models/delta_model_features.joblib
#
# Optional input:
# - data/processed/master_data_cache.pkl
#   Used only for richer school-name lookup if available.
#
# Outputs:
# - Interactive dashboard interface
# - Leadership briefing text export
#
# Role in the pipeline:
# This script is the deployment / prototype layer of the
# dissertation. It operationalises the trained models within
# an interpretable interface for school-level exploration,
# risk triage, and scenario testing.
#
# Reproducibility note:
# This public repository version uses relative paths based on
# the repository structure. It expects the app to be run from
# the project repository containing data/, models/, and app/.
#
# Methodological note:
# This tool is designed as a decision-support prototype rather
# than an automated decision system. It presents:
# - underperformance risk probabilities
# - overperformance probabilities
# - SHAP-based local explanation
# - optional projected P8 bands
#
# Scenario note:
# The scenario module estimates model sensitivity to
# hypothetical feature adjustments and should therefore be
# interpreted as exploratory simulation rather than causal
# intervention prediction.
#
# Governance note:
# The dashboard is intended to support professional judgement,
# not replace it. Risk estimates are probabilistic and should
# be interpreted alongside contextual knowledge.
# ==========================================================

import os
import glob
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap


# ==========================================================
# CONFIG
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FOLDER = DATA_DIR

PANEL_PATH = DATA_DIR / "school_panel_final.parquet"
MASTER_CACHE_PATH = DATA_DIR / "master_data_cache.pkl"

UNDER_MODEL_PATH = MODEL_DIR / "risk_xgb_classifier.joblib"
UNDER_FEATURES_PATH = MODEL_DIR / "risk_model_features.joblib"

OVER_MODEL_PATH = MODEL_DIR / "overperf_xgb_classifier.joblib"
OVER_FEATURES_PATH = MODEL_DIR / "overperf_model_features.joblib"

DELTA_MODEL_PATH = MODEL_DIR / "delta_xgb_model.joblib"
DELTA_FEATURES_PATH = MODEL_DIR / "delta_model_features.joblib"

GREEN_MAX = 0.40
AMBER_MAX = 0.60
P8_MAE_BAND = 0.21


# ==========================================================
# STREAMLIT PAGE SETUP
# ==========================================================

st.set_page_config(
    page_title="Trust School Priority Tool",
    page_icon="🧭",
    layout="wide"
)


# ==========================================================
# STYLING
# ==========================================================

st.markdown(
    """
    <style>
      .kicker { font-size: 1.0rem; opacity: 0.95; }
      .muted { opacity: 0.88; }
      .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,0.10);
        background: rgba(0,0,0,0.03);
        font-size: 0.95rem;
        margin-right: 8px;
        margin-bottom: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def clamp(v: float, lo: float, hi: float) -> float:
    if pd.isna(v):
        return v
    return float(max(lo, min(hi, v)))


def band_prob(p: float) -> str:
    if p < GREEN_MAX:
        return "LOW"
    if p < AMBER_MAX:
        return "MEDIUM"
    return "HIGH"


def rag_colour(level: str) -> str:
    return {
        "LOW": "🟢",
        "MEDIUM": "🟠",
        "HIGH": "🔴"
    }.get(level, "⚪")


def build_X(row: pd.Series, feature_list: list[str]) -> pd.DataFrame:
    X = pd.DataFrame([{f: row.get(f, np.nan) for f in feature_list}])
    return X.apply(pd.to_numeric, errors="coerce")


def predict_prob(model, X: pd.DataFrame) -> float:
    return float(model.predict_proba(X)[0, 1])


def _read_csv_minimal(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)

    df.columns = [c.upper().strip() for c in df.columns]
    return df


@st.cache_data
def build_urn_name_lookup(panel_urns: np.ndarray) -> dict[int, str]:
    """
    Build a URN -> school name lookup by scanning available
    raw / processed school information and spine files.

    This improves usability of the dashboard by showing
    human-readable labels rather than URNs alone.
    """
    urn_set = set(int(u) for u in panel_urns if pd.notna(u))
    mapping: dict[int, str] = {}

    patterns = [
        str(RAW_FOLDER / "*school_information*.csv"),
        str(PROCESSED_FOLDER / "*school_information*.csv"),
        str(RAW_FOLDER / "*spine*.csv"),
        str(PROCESSED_FOLDER / "*spine*.csv"),
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    possible_name_cols = [
        "SCHOOLNAME",
        "ESTABLISHMENTNAME",
        "NAME",
        "SCHNAME",
        "ESTABLISHMENT_NAME"
    ]

    possible_urn_cols = [
        "URN",
        "SCHOOL_URN"
    ]

    for f in sorted(set(files)):
        try:
            df = _read_csv_minimal(f)
            urn_col = next((c for c in possible_urn_cols if c in df.columns), None)
            name_col = next((c for c in possible_name_cols if c in df.columns), None)

            if not urn_col or not name_col:
                continue

            tmp = df[[urn_col, name_col]].copy()
            tmp.rename(
                columns={urn_col: "URN_FINAL", name_col: "SCHOOL_NAME"},
                inplace=True
            )

            tmp["URN_FINAL"] = pd.to_numeric(tmp["URN_FINAL"], errors="coerce")
            tmp = tmp.dropna(subset=["URN_FINAL"])
            tmp["URN_FINAL"] = tmp["URN_FINAL"].astype(int)
            tmp = tmp[tmp["URN_FINAL"].isin(urn_set)]
            tmp["SCHOOL_NAME"] = tmp["SCHOOL_NAME"].astype(str).str.strip()
            tmp = tmp[tmp["SCHOOL_NAME"].ne("")]

            for u, n in zip(tmp["URN_FINAL"], tmp["SCHOOL_NAME"]):
                if u not in mapping:
                    mapping[u] = n

        except Exception:
            continue

    if os.path.exists(MASTER_CACHE_PATH):
        try:
            with open(MASTER_CACHE_PATH, "rb") as fh:
                cache = pickle.load(fh)

            for key in ["school_info", "spine"]:
                if key in cache and isinstance(cache[key], pd.DataFrame):
                    df = cache[key].copy()
                    df.columns = [c.upper().strip() for c in df.columns]

                    urn_col = next((c for c in possible_urn_cols if c in df.columns), None)
                    name_col = next((c for c in possible_name_cols if c in df.columns), None)

                    if not urn_col or not name_col:
                        continue

                    tmp = df[[urn_col, name_col]].copy()
                    tmp.rename(
                        columns={urn_col: "URN_FINAL", name_col: "SCHOOL_NAME"},
                        inplace=True
                    )

                    tmp["URN_FINAL"] = pd.to_numeric(tmp["URN_FINAL"], errors="coerce")
                    tmp = tmp.dropna(subset=["URN_FINAL"])
                    tmp["URN_FINAL"] = tmp["URN_FINAL"].astype(int)
                    tmp = tmp[tmp["URN_FINAL"].isin(urn_set)]
                    tmp["SCHOOL_NAME"] = tmp["SCHOOL_NAME"].astype(str).str.strip()
                    tmp = tmp[tmp["SCHOOL_NAME"].ne("")]

                    for u, n in zip(tmp["URN_FINAL"], tmp["SCHOOL_NAME"]):
                        if u not in mapping:
                            mapping[u] = n

        except Exception:
            pass

    return mapping


def display_label(u: int, name_map: dict[int, str]) -> str:
    nm = name_map.get(int(u), "")
    return f"{nm} ({u})" if nm else f"URN {u}"


def parse_urn_from_display(s: str) -> int:
    if s.startswith("URN "):
        return int(s.split("URN ")[1].strip())
    return int(s.rsplit("(", 1)[1].replace(")", "").strip())


def theme_for_feature(f: str) -> str:
    """
    Group technical feature names into interpretable thematic
    categories for action recommendations and user-facing
    explanation.
    """
    fu = f.upper()

    if "ABSENCE" in fu:
        return "Attendance"
    if "PTR" in fu or "TEACHER" in fu:
        return "Workforce / Capacity"
    if "FSM" in fu:
        return "Disadvantage"
    if "SEN" in fu or "EHCP" in fu:
        return "SEND"
    if fu.startswith("PNUM") or fu.startswith("NUM"):
        return "Pupil cohort / Context"

    return "Other"


def top3_drivers(explainer, X: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """
    Extract the top three local SHAP drivers for the selected
    school. These form the technical basis of the explanation
    and recommendation logic shown in the dashboard.
    """
    sv = explainer.shap_values(X)

    if isinstance(sv, list) and len(sv) == 2:
        vals = sv[1][0]
    else:
        arr = np.array(sv)
        vals = arr[0, :, 1] if arr.ndim == 3 else arr[0, :]

    df = pd.DataFrame({
        "feature": feature_list,
        "abs": np.abs(vals),
        "signed": vals
    })

    return df.sort_values("abs", ascending=False).head(3).reset_index(drop=True)


@st.cache_data
def compute_feature_benchmarks(
    panel: pd.DataFrame,
    features: list[str]
) -> dict[str, dict[str, float]]:
    """
    Compute percentile-based benchmark ranges for selected
    features. These are used to cap scenario sliders and
    prevent unrealistic user inputs.
    """
    benchmarks = {}

    for f in features:
        if f in panel.columns:
            s = pd.to_numeric(panel[f], errors="coerce")

            if s.notna().sum() > 200:
                benchmarks[f] = {
                    "p05": float(s.quantile(0.05)),
                    "p25": float(s.quantile(0.25)),
                    "p50": float(s.quantile(0.50)),
                    "p75": float(s.quantile(0.75)),
                    "p95": float(s.quantile(0.95)),
                }

    return benchmarks


def apply_basic_levers(
    X: pd.DataFrame,
    absence_delta_pp: float,
    ptr_delta: float
) -> pd.DataFrame:
    """
    Apply simple scenario changes to attendance and pupil-teacher
    ratio. These are deterministic feature edits used for
    what-if simulation only.
    """
    X2 = X.copy()

    if "ABSENCE_RATE" in X2.columns and pd.notna(X2.at[0, "ABSENCE_RATE"]):
        X2.at[0, "ABSENCE_RATE"] = max(
            0.0,
            float(X2.at[0, "ABSENCE_RATE"]) - float(absence_delta_pp)
        )

    if "PTR" in X2.columns and pd.notna(X2.at[0, "PTR"]):
        X2.at[0, "PTR"] = max(
            1.0,
            float(X2.at[0, "PTR"]) - float(ptr_delta)
        )

    return X2


def model_score(model, X: pd.DataFrame, mode: str) -> float:
    """
    Helper used by scenario grid search.

    mode='min_prob' : minimise classifier probability
    mode='max_pred' : maximise regressor prediction indirectly
                      by minimising negative predicted value
    """
    if mode == "min_prob":
        if not hasattr(model, "predict_proba"):
            raise TypeError("mode='min_prob' requires predict_proba().")
        return float(model.predict_proba(X)[0, 1])

    if mode == "max_pred":
        return -float(model.predict(X)[0])

    raise ValueError(f"Unknown mode: {mode}")


def apply_send_intervention_grid(
    model,
    X: pd.DataFrame,
    strength: float,
    drivers_df: pd.DataFrame,
    benchmarks: dict[str, dict[str, float]],
    mode: str,
    max_send_features: int = 6,
    grid_points: int = 41,
    eps: float = 1e-6
) -> tuple[pd.DataFrame, str | None, float | None, float | None, float]:
    """
    Explore plausible SEND-related feature adjustments within
    empirical percentile bounds and apply the most beneficial
    change according to the selected model objective.

    This is a model-sensitivity routine, not a causal
    intervention estimator.
    """
    X2 = X.copy()
    base_score = model_score(model, X2, mode)

    candidates = []

    for _, r in drivers_df.iterrows():
        f = str(r["feature"])

        if ("SEN" in f.upper() or "EHCP" in f.upper()) and f in X2.columns:
            candidates.append(f)

    if len(candidates) < max_send_features:
        for f in X2.columns:
            fu = f.upper()

            if ("SEN" in fu or "EHCP" in fu) and f not in candidates:
                candidates.append(f)

            if len(candidates) >= max_send_features:
                break

    if not candidates:
        return X2, None, None, None, 0.0

    best_feat = None
    best_value = None
    best_score = base_score

    for feat in candidates:
        if feat not in benchmarks:
            continue

        cur = pd.to_numeric(X2.at[0, feat], errors="coerce")

        if pd.isna(cur):
            continue

        lo = benchmarks[feat].get("p05", np.nan)
        hi = benchmarks[feat].get("p95", np.nan)

        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < eps:
            continue

        grid = np.linspace(lo, hi, grid_points)

        for v in grid:
            Xt = X2.copy()
            Xt.at[0, feat] = float(v)
            s = model_score(model, Xt, mode)

            if s < best_score:
                best_score = s
                best_feat = feat
                best_value = float(v)

    if best_feat is None or best_value is None:
        fallback_feat = candidates[0]
        cur = pd.to_numeric(X2.at[0, fallback_feat], errors="coerce")

        return X2, fallback_feat, (float(cur) if pd.notna(cur) else None), None, 0.0

    cur = float(pd.to_numeric(X2.at[0, best_feat], errors="coerce"))
    applied = float(cur + float(strength) * (best_value - cur))
    X2.at[0, best_feat] = applied

    if mode == "min_prob":
        base_prob = float(model.predict_proba(X)[0, 1])
        new_prob = float(model.predict_proba(X2)[0, 1])
        delta_display = (new_prob - base_prob) * 100.0
    else:
        base_pred = float(model.predict(X)[0])
        new_pred = float(model.predict(X2)[0])
        delta_display = new_pred - base_pred

    return X2, best_feat, cur, best_value, float(delta_display)


def risk_circle_component(p: float, label: str) -> None:
    """
    Render a simple RAG probability label.
    """
    p = float(np.clip(p, 0.0, 1.0))
    level = band_prob(p)

    st.markdown(f"**{label}**")
    st.markdown(f"{rag_colour(level)} **{level}** ({p * 100:.0f}%)")


def action_playbook(theme: str) -> dict:
    """
    Return simple user-facing action suggestions linked to the
    dominant explanatory theme. These are heuristic prompts for
    discussion, not direct recommendations from a causal model.
    """
    theme = theme.strip()

    if theme == "Attendance":
        return {
            "why": "Attendance is directly actionable and strongly associated with outcomes.",
            "actions": [
                "Target the highest-absence pupils first (pastoral + family engagement).",
                "Weekly attendance huddles: identify barriers and assign named adult support.",
                "Tighten first-day response and rapid reintegration after absence."
            ],
            "owner": "Attendance lead / Pastoral"
        }

    if theme == "Workforce / Capacity":
        return {
            "why": "Capacity affects teaching quality and ability to deliver targeted support.",
            "actions": [
                "Stabilise staffing in core subjects and high-need year groups.",
                "Protect intervention time for small groups (timetable + staffing).",
                "Review TA/specialist deployment for maximum impact."
            ],
            "owner": "Headteacher / HR / Ops"
        }

    if theme == "SEND":
        return {
            "why": "Identification and provision quality can shift outcomes for vulnerable learners.",
            "actions": [
                "SENCO-led provision audit: needs → plan → delivery → review.",
                "Check support is consistent in classrooms, not only on paper.",
                "Train staff on adaptive teaching and structured scaffolding."
            ],
            "owner": "SENCO / Inclusion"
        }

    if theme == "Disadvantage":
        return {
            "why": "Disadvantage is a key contextual risk factor; mitigation is targeted support.",
            "actions": [
                "Prioritise tutoring and high-impact interventions for disadvantaged pupils.",
                "Improve homework access through study clubs, devices, or quiet spaces.",
                "Review progress half-termly and adjust intensity quickly."
            ],
            "owner": "Pupil Premium lead"
        }

    return {
        "why": "This factor influences the forecast, but may be less directly controllable.",
        "actions": [
            "Use the Evidence tab to review underlying driver feature(s).",
            "Prioritise attendance, SEND provision, and staffing where possible."
        ],
        "owner": "Leadership team"
    }


def make_briefing_text(
    school_name: str,
    urn: int,
    year_latest: float,
    p_under_base: float,
    p_under_scn: float,
    p_over_base: float,
    p_over_scn: float,
    att_before: float | None,
    att_after: float | None,
    ptr_before: float | None,
    ptr_after: float | None,
    send_feat_used: str | None,
    send_before: float | None,
    send_after: float | None,
    primary_focus: str,
) -> str:
    """
    Build a plain-text briefing summary for export.
    """
    lines = []

    lines.append("Trust School Priority Briefing (Decision Support)")
    lines.append(f"School: {school_name} (URN {urn})")
    lines.append(f"Latest year in panel: {int(year_latest) if pd.notna(year_latest) else 'Unknown'}")
    lines.append("")
    lines.append("Summary (model estimates)")
    lines.append(f"- Priority estimate (before): {p_under_base * 100:.0f}% [{band_prob(p_under_base)}]")
    lines.append(f"- Priority estimate (after scenario): {p_under_scn * 100:.0f}% [{band_prob(p_under_scn)}]")
    lines.append(f"- Chance of exceeding expected progress (before): {p_over_base * 100:.0f}%")
    lines.append(f"- Chance of exceeding expected progress (after scenario): {p_over_scn * 100:.0f}%")
    lines.append("")
    lines.append("Scenario inputs (before to after)")

    if att_before is not None and att_after is not None and pd.notna(att_before) and pd.notna(att_after):
        lines.append(f"- Attendance: {att_before:.1f}% to {att_after:.1f}%")

    if ptr_before is not None and ptr_after is not None and pd.notna(ptr_before) and pd.notna(ptr_after):
        lines.append(f"- PTR: {ptr_before:.2f} to {ptr_after:.2f}")

    if send_feat_used and send_before is not None and send_after is not None and pd.notna(send_before) and pd.notna(send_after):
        lines.append(f"- SEND driver adjusted ({send_feat_used}): {send_before:.2f} to {send_after:.2f}")

    lines.append("")
    lines.append(f"Recommended focus area: {primary_focus}")
    lines.append("")
    lines.append("Note: This is a what-if tool. It estimates sensitivity, not guaranteed causal impact.")

    return "\n".join(lines)


# ==========================================================
# LOAD MODELS AND DATA
# ==========================================================

@st.cache_resource
def load_models():
    """
    Load trained model artefacts and construct SHAP explainers.

    Required models:
    - underperformance classifier
    - overperformance classifier

    Optional:
    - delta regressor and its feature list
    """
    required = [
        (UNDER_MODEL_PATH, "underperformance model"),
        (UNDER_FEATURES_PATH, "underperformance feature list"),
        (OVER_MODEL_PATH, "overperformance model"),
        (OVER_FEATURES_PATH, "overperformance feature list"),
    ]

    for path, label in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}: {path}")

    under_model = joblib.load(UNDER_MODEL_PATH)
    under_feats = joblib.load(UNDER_FEATURES_PATH)
    under_expl = shap.TreeExplainer(under_model)

    over_model = joblib.load(OVER_MODEL_PATH)
    over_feats = joblib.load(OVER_FEATURES_PATH)
    over_expl = shap.TreeExplainer(over_model)

    delta_model = None
    delta_feats = None

    if os.path.exists(DELTA_MODEL_PATH) and os.path.exists(DELTA_FEATURES_PATH):
        try:
            delta_model = joblib.load(DELTA_MODEL_PATH)
            delta_feats = joblib.load(DELTA_FEATURES_PATH)
        except Exception:
            delta_model = None
            delta_feats = None

    return (
        under_model,
        under_feats,
        under_expl,
        over_model,
        over_feats,
        over_expl,
        delta_model,
        delta_feats
    )


@st.cache_data
def load_panel():
    """
    Load the final modelling panel and enforce core identifier
    types for dashboard use.
    """
    if not os.path.exists(PANEL_PATH):
        raise FileNotFoundError(f"Missing panel file: {PANEL_PATH}")

    df = pd.read_parquet(PANEL_PATH)

    df["URN_FINAL"] = pd.to_numeric(df["URN_FINAL"], errors="coerce")
    df = df.dropna(subset=["URN_FINAL"])
    df["URN_FINAL"] = df["URN_FINAL"].astype(int)
    df["YEAR_START"] = pd.to_numeric(df.get("YEAR_START", np.nan), errors="coerce")

    return df


(
    under_model,
    under_feats,
    under_expl,
    over_model,
    over_feats,
    over_expl,
    delta_model,
    delta_feats
) = load_models()

panel = load_panel()

urns = np.array(sorted(panel["URN_FINAL"].unique().tolist()), dtype=int)
name_map = build_urn_name_lookup(urns)
display_options = [display_label(u, name_map) for u in urns]

bench_under = compute_feature_benchmarks(panel, under_feats)


# ==========================================================
# HEADER
# ==========================================================

st.markdown("## 🧭 Trust School Priority Tool")
st.markdown(
    "<div class='kicker'>A decision-support prototype to help prioritise support and test plausible scenarios. "
    "<span class='muted'>Not a guarantee; use alongside professional judgement.</span></div>",
    unsafe_allow_html=True
)


# ==========================================================
# SAFE DEFAULTS
# ==========================================================

absence_delta_pp = 0.0
ptr_delta = 0.0
send_strength = float(st.session_state.get("send_strength", 0.0))
selected_urn = None


# ==========================================================
# SIDEBAR INPUTS
# ==========================================================

with st.sidebar:
    st.markdown("### School")

    search = st.text_input(
        "Search school name or URN",
        value="",
        key="search_school"
    )

    filtered = display_options

    if search.strip():
        q = search.strip().lower()
        filtered = [opt for opt in display_options if q in opt.lower()]

        if not filtered:
            st.warning("No matches. Showing full list.")
            filtered = display_options

    selected_display = st.selectbox(
        "Select school",
        filtered,
        index=0,
        key="selected_school"
    )

    selected_urn = parse_urn_from_display(selected_display)

    school_rows_tmp = panel[panel["URN_FINAL"] == int(selected_urn)].sort_values("YEAR_START")
    latest_tmp = school_rows_tmp.iloc[-1]

    abs_base = pd.to_numeric(latest_tmp.get("ABSENCE_RATE", np.nan), errors="coerce")
    att_base = clamp(100.0 - abs_base, 0.0, 100.0) if pd.notna(abs_base) else np.nan
    ptr_base = pd.to_numeric(latest_tmp.get("PTR", np.nan), errors="coerce")

    st.markdown("---")
    st.markdown("### Scenario builder (what-if)")

    att_min, att_max = 80.0, 100.0

    if "ABSENCE_RATE" in bench_under:
        lo_abs = bench_under["ABSENCE_RATE"].get("p05", np.nan)
        hi_abs = bench_under["ABSENCE_RATE"].get("p95", np.nan)

        if np.isfinite(lo_abs) and np.isfinite(hi_abs):
            att_min = float(clamp(100.0 - hi_abs, 0.0, 100.0))
            att_max = float(clamp(100.0 - lo_abs, 0.0, 100.0))

            if att_min > att_max:
                att_min, att_max = att_max, att_min

    ptr_min, ptr_max = 10.0, 35.0

    if "PTR" in bench_under:
        lo_ptr = bench_under["PTR"].get("p05", np.nan)
        hi_ptr = bench_under["PTR"].get("p95", np.nan)

        if np.isfinite(lo_ptr) and np.isfinite(hi_ptr):
            ptr_min = float(max(1.0, lo_ptr))
            ptr_max = float(max(ptr_min + 0.5, hi_ptr))

    last_urn = st.session_state.get("last_urn", None)

    if last_urn != selected_urn:
        if pd.notna(att_base):
            st.session_state["att_target"] = float(clamp(att_base, att_min, att_max))
        else:
            st.session_state.pop("att_target", None)

        if pd.notna(ptr_base):
            st.session_state["ptr_target"] = float(clamp(ptr_base, ptr_min, ptr_max))
        else:
            st.session_state.pop("ptr_target", None)

        st.session_state["send_strength"] = 0.0
        st.session_state["last_urn"] = selected_urn
        st.session_state["do_reset"] = False

    if st.session_state.get("do_reset", False):
        if pd.notna(att_base):
            st.session_state["att_target"] = float(clamp(att_base, att_min, att_max))
        else:
            st.session_state.pop("att_target", None)

        if pd.notna(ptr_base):
            st.session_state["ptr_target"] = float(clamp(ptr_base, ptr_min, ptr_max))
        else:
            st.session_state.pop("ptr_target", None)

        st.session_state["send_strength"] = 0.0
        st.session_state["do_reset"] = False

    if pd.notna(att_base):
        if "att_target" not in st.session_state or not np.isfinite(st.session_state["att_target"]):
            st.session_state["att_target"] = float(clamp(att_base, att_min, att_max))

        st.session_state["att_target"] = float(
            clamp(st.session_state["att_target"], att_min, att_max)
        )

    if pd.notna(ptr_base):
        if "ptr_target" not in st.session_state or not np.isfinite(st.session_state["ptr_target"]):
            st.session_state["ptr_target"] = float(clamp(ptr_base, ptr_min, ptr_max))

        st.session_state["ptr_target"] = float(
            clamp(st.session_state["ptr_target"], ptr_min, ptr_max)
        )

    if "send_strength" not in st.session_state or not np.isfinite(st.session_state["send_strength"]):
        st.session_state["send_strength"] = 0.0

    st.session_state["send_strength"] = float(
        clamp(st.session_state["send_strength"], 0.0, 1.0)
    )

    if pd.notna(att_base):
        att_target = st.slider(
            "Attendance target (%)",
            min_value=float(att_min),
            max_value=float(att_max),
            step=0.5,
            key="att_target",
        )
    else:
        st.info("Attendance not available for this school.")
        att_target = np.nan

    if pd.notna(ptr_base):
        ptr_target = st.slider(
            "PTR target (pupils per teacher)",
            min_value=float(ptr_min),
            max_value=float(ptr_max),
            step=0.5,
            key="ptr_target",
        )
    else:
        st.info("PTR not available for this school.")
        ptr_target = np.nan

    st.markdown("**SEND support intensity (what-if)**")

    b1, b2, b3 = st.columns(3)

    if b1.button("Minor", use_container_width=True):
        st.session_state["send_strength"] = 0.25
        st.rerun()

    if b2.button("Moderate", use_container_width=True):
        st.session_state["send_strength"] = 0.60
        st.rerun()

    if b3.button("Intensive", use_container_width=True):
        st.session_state["send_strength"] = 1.00
        st.rerun()

    send_strength = st.slider(
        "Fine-tune SEND intensity",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="send_strength",
    )

    absence_delta_pp = 0.0

    if pd.notna(att_base) and pd.notna(att_target) and pd.notna(abs_base):
        abs_target = 100.0 - float(att_target)
        absence_delta_pp = max(0.0, float(abs_base) - float(abs_target))

    ptr_delta = 0.0

    if pd.notna(ptr_base) and pd.notna(ptr_target):
        ptr_delta = max(0.0, float(ptr_base) - float(ptr_target))

    affects_att = "ABSENCE_RATE" in under_feats
    affects_ptr = "PTR" in under_feats

    st.caption(
        f"Model sensitivity: Attendance={'Yes' if affects_att else 'No'} | "
        f"PTR={'Yes' if affects_ptr else 'No'}"
    )

    if st.button("Reset scenario", use_container_width=True):
        st.session_state["do_reset"] = True
        st.rerun()


# ==========================================================
# MAIN SCORING LOGIC
# ==========================================================

school_rows = panel[panel["URN_FINAL"] == int(selected_urn)].sort_values("YEAR_START")
latest = school_rows.iloc[-1]

school_name = name_map.get(int(selected_urn), f"URN {selected_urn}")
year_latest = latest.get("YEAR_START", np.nan)

X_under_base = build_X(latest, under_feats)
X_over_base = build_X(latest, over_feats)

p_under_base = predict_prob(under_model, X_under_base)
p_over_base = predict_prob(over_model, X_over_base)

drivers = top3_drivers(under_expl, X_under_base, under_feats)

X_under_scn = apply_basic_levers(X_under_base, absence_delta_pp, ptr_delta)
X_over_scn = apply_basic_levers(X_over_base, absence_delta_pp, ptr_delta)

X_under_scn, send_feat_used, send_cur, send_best, send_delta_display = apply_send_intervention_grid(
    under_model,
    X_under_scn,
    send_strength,
    drivers,
    bench_under,
    mode="min_prob"
)

if (
    send_feat_used is not None
    and send_best is not None
    and send_feat_used in X_over_scn.columns
):
    cur_over = pd.to_numeric(X_over_scn.at[0, send_feat_used], errors="coerce")

    if pd.notna(cur_over):
        applied = float(cur_over + send_strength * (send_best - float(cur_over)))
        X_over_scn.at[0, send_feat_used] = applied

p_under_scn = predict_prob(under_model, X_under_scn)
p_over_scn = predict_prob(over_model, X_over_scn)

abs_before = (
    pd.to_numeric(X_under_base.get("ABSENCE_RATE", pd.Series([np.nan])).iloc[0], errors="coerce")
    if "ABSENCE_RATE" in X_under_base.columns
    else np.nan
)

abs_after = (
    pd.to_numeric(X_under_scn.get("ABSENCE_RATE", pd.Series([np.nan])).iloc[0], errors="coerce")
    if "ABSENCE_RATE" in X_under_scn.columns
    else np.nan
)

att_before = clamp(100.0 - abs_before, 0.0, 100.0) if pd.notna(abs_before) else np.nan
att_after = clamp(100.0 - abs_after, 0.0, 100.0) if pd.notna(abs_after) else np.nan

ptr_before = (
    pd.to_numeric(X_under_base.get("PTR", pd.Series([np.nan])).iloc[0], errors="coerce")
    if "PTR" in X_under_base.columns
    else np.nan
)

ptr_after = (
    pd.to_numeric(X_under_scn.get("PTR", pd.Series([np.nan])).iloc[0], errors="coerce")
    if "PTR" in X_under_scn.columns
    else np.nan
)

send_before = np.nan
send_after = np.nan

if (
    send_feat_used is not None
    and send_feat_used in X_under_base.columns
    and send_feat_used in X_under_scn.columns
):
    send_before = pd.to_numeric(X_under_base.at[0, send_feat_used], errors="coerce")
    send_after = pd.to_numeric(X_under_scn.at[0, send_feat_used], errors="coerce")


# ==========================================================
# OPTIONAL P8 FORECAST BAND
# ==========================================================

def forecast_p8_band(
    delta_model,
    delta_feats,
    latest_row: pd.Series,
    X_delta_scn: pd.DataFrame | None
):
    lag = pd.to_numeric(latest_row.get("TARGET_P8_LAG1", np.nan), errors="coerce")

    if pd.isna(lag):
        lag = pd.to_numeric(latest_row.get("TARGET_P8", np.nan), errors="coerce")

    if delta_model is None or delta_feats is None or pd.isna(lag):
        return None, None, None, None

    Xb = build_X(latest_row, delta_feats)

    try:
        d_hat = float(delta_model.predict(Xb)[0])
    except Exception:
        return None, None, None, None

    p8_hat = float(lag + d_hat)
    band = (p8_hat - P8_MAE_BAND, p8_hat + P8_MAE_BAND)

    if X_delta_scn is not None:
        try:
            d_hat_s = float(delta_model.predict(X_delta_scn)[0])
            p8_hat_s = float(lag + d_hat_s)
            band_s = (p8_hat_s - P8_MAE_BAND, p8_hat_s + P8_MAE_BAND)

            return p8_hat, band, p8_hat_s, band_s

        except Exception:
            pass

    return p8_hat, band, None, None


p8_base = band_base = p8_scn = band_scn = None

if delta_model is not None and delta_feats is not None:
    bench_delta = compute_feature_benchmarks(panel, delta_feats)
    X_delta_base = build_X(latest, delta_feats)
    X_delta_scn = apply_basic_levers(X_delta_base, absence_delta_pp, ptr_delta)

    X_delta_scn, _, _, _, _ = apply_send_intervention_grid(
        delta_model,
        X_delta_scn,
        send_strength,
        drivers,
        bench_delta,
        mode="max_pred"
    )

    p8_base, band_base, p8_scn, band_scn = forecast_p8_band(
        delta_model,
        delta_feats,
        latest,
        X_delta_scn
    )

theme_counts = pd.Series(
    [theme_for_feature(str(r["feature"])) for _, r in drivers.iterrows()]
).value_counts()

primary_focus = theme_counts.index[0] if len(theme_counts) else "Improvement"


# ==========================================================
# MAIN USER INTERFACE
# ==========================================================

st.markdown(
    f"<span class='pill'><b>{school_name}</b> (URN {int(selected_urn)})</span>"
    f"<span class='pill'>Latest year: <b>{int(year_latest) if pd.notna(year_latest) else 'Unknown'}</b></span>",
    unsafe_allow_html=True
)

tab_overview, tab_actions, tab_evidence = st.tabs(
    ["Overview", "Actions", "Evidence"]
)


with tab_overview:
    level_base = band_prob(p_under_base)
    level_scn = band_prob(p_under_scn)

    risk_change_pp = (p_under_scn - p_under_base) * 100.0
    over_change_pp = (p_over_scn - p_over_base) * 100.0

    with st.container(border=True):
        st.markdown(f"## {rag_colour(level_base)} Priority level: **{level_base}**")
        st.write(
            f"**{school_name}** is estimated to be **{level_base} priority** for next-year support "
            f"(based on patterns in similar schools and prior outcomes)."
        )
        st.info(f"**Recommended focus area:** {primary_focus}")
        st.caption("This is decision support, not a guarantee. Use alongside local knowledge.")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown("### Current outlook")
            risk_circle_component(
                p_under_base,
                "Priority estimate (likelihood of underperformance)"
            )
            st.metric("Priority estimate", f"{p_under_base * 100:.0f}%")
            st.metric("Chance of exceeding expected progress", f"{p_over_base * 100:.0f}%")
            st.caption(f"Status: {rag_colour(level_base)} {level_base}")

    with c2:
        with st.container(border=True):
            st.markdown("### With your scenario")
            risk_circle_component(
                p_under_scn,
                "Priority estimate (scenario)"
            )
            st.metric(
                "Priority estimate (scenario)",
                f"{p_under_scn * 100:.0f}%",
                delta=f"{risk_change_pp:+.0f}pp",
                delta_color="inverse"
            )
            st.metric(
                "Chance of exceeding expected progress (scenario)",
                f"{p_over_scn * 100:.0f}%",
                delta=f"{over_change_pp:+.0f}pp",
            )
            st.caption(f"Status: {rag_colour(level_scn)} {level_scn}")

    with st.container(border=True):
        st.markdown("### Optional: Projected P8 band (next year)")

        if band_base is not None:
            cc1, cc2 = st.columns(2, gap="large")

            with cc1:
                st.markdown("**Before**")
                st.write(f"Projected band: **{band_base[0]:.2f} to {band_base[1]:.2f}**")

            with cc2:
                st.markdown("**After scenario**")

                if band_scn is not None:
                    st.write(f"Projected band: **{band_scn[0]:.2f} to {band_scn[1]:.2f}**")
                else:
                    st.write("Not available.")

            st.caption(f"Band reflects ±{P8_MAE_BAND:.2f} around the forecast.")

        else:
            st.write("Not available. Delta model is missing, or the selected row has no valid lagged P8 value.")

    briefing = make_briefing_text(
        school_name=school_name,
        urn=int(selected_urn),
        year_latest=year_latest,
        p_under_base=p_under_base,
        p_under_scn=p_under_scn,
        p_over_base=p_over_base,
        p_over_scn=p_over_scn,
        att_before=(float(att_before) if pd.notna(att_before) else None),
        att_after=(float(att_after) if pd.notna(att_after) else None),
        ptr_before=(float(ptr_before) if pd.notna(ptr_before) else None),
        ptr_after=(float(ptr_after) if pd.notna(ptr_after) else None),
        send_feat_used=send_feat_used,
        send_before=(float(send_before) if pd.notna(send_before) else None),
        send_after=(float(send_after) if pd.notna(send_after) else None),
        primary_focus=primary_focus,
    )

    st.download_button(
        "Download leadership briefing (txt)",
        data=briefing,
        file_name=f"priority_briefing_{int(selected_urn)}.txt",
        mime="text/plain"
    )


with tab_actions:
    st.markdown("## Recommended actions this term")
    st.caption(
        "Suggestions are based on what is most associated with this school’s forecast. "
        "Tailor using professional judgement."
    )

    themes = []

    for _, r in drivers.iterrows():
        t = theme_for_feature(str(r["feature"]))

        if t not in themes:
            themes.append(t)

        if len(themes) >= 3:
            break

    cols = st.columns(3, gap="large")

    for i, t in enumerate(themes):
        play = action_playbook(t)

        with cols[i]:
            with st.container(border=True):
                st.markdown(f"### {i + 1}. {t}")
                st.write(f"**Why this matters here:** {play['why']}")
                st.write("**Suggested actions:**")

                for a in play["actions"]:
                    st.write(f"• {a}")

                st.caption(f"Likely owner: {play['owner']}")


with tab_evidence:
    with st.container(border=True):
        st.markdown("### Evidence")
        st.caption("For dissertation/examiner transparency and technical checks.")
        st.write(f"Underperformance model features: **{len(under_feats)}**")
        st.write(f"Overperformance model features: **{len(over_feats)}**")
        st.write(f"Delta model available: **{'Yes' if delta_model is not None else 'No'}**")

    with st.expander("Show top driver features"):
        st.dataframe(drivers, use_container_width=True)

    with st.expander("Show latest school rows"):
        st.dataframe(school_rows.tail(8), use_container_width=True)

    with st.expander("Show key notes"):
        st.write("- Priority bands: LOW < 40%, MEDIUM 40–60%, HIGH ≥ 60%.")
        st.write("- Attendance/PTR sliders are capped to p05–p95 ranges to avoid unrealistic scenarios.")
        st.write("- Scenarios show model sensitivity, not guaranteed causality.")


# ==========================================================
# OUTPUTS / USAGE SUMMARY
# ==========================================================
#
# This application provides:
# - school-level risk triage
# - scenario testing
# - SHAP-based explanation
# - leadership briefing export
#
# It functions as a prototype decision-support interface for
# the dissertation rather than a production deployment tool.
# ==========================================================
