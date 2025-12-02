import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# to search epitopes
DATA_PATH = "cleaned_final.csv"  # <- replace with your real file name

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ---------------------------
# 1. Load model + metadata
# ---------------------------
MODEL_PATH = "rf_tuned_model.joblib"
THR_PATH = "rf_tuned_threshold.json"
FEAT_PATH = "rf_feature_columns.json"

@st.cache_resource
def load_model_and_meta():
    model = joblib.load(MODEL_PATH)
    with open(THR_PATH, "r") as f:
        thr = json.load(f)["threshold"]
    with open(FEAT_PATH, "r") as f:
        feature_cols = json.load(f)["feature_columns"]
    return model, thr, feature_cols

rf_model, DECISION_THR, FEATURE_COLS = load_model_and_meta()

SRC_PLACEHOLDER  = "Select source organism"
MHC_PLACEHOLDER  = "Select MHC context"
RESP_PLACEHOLDER = "Select response type"


# prefixes for one-hot columns (must match training)
SRC_PREFIX  = "Source Organism_"
MHC_PREFIX  = "MHC Present_mode_"
RESP_PREFIX = "Response_measured_mode_"

# derive available options from feature names (N-1 dummies; one implicit baseline)
SRC_OPTIONS  = sorted([c[len(SRC_PREFIX):]  for c in FEATURE_COLS if c.startswith(SRC_PREFIX)])
MHC_OPTIONS  = sorted([c[len(MHC_PREFIX):]  for c in FEATURE_COLS if c.startswith(MHC_PREFIX)])
RESP_OPTIONS = sorted([c[len(RESP_PREFIX):] for c in FEATURE_COLS if c.startswith(RESP_PREFIX)])

BASE_SRC  = "Mycobacterium tuberculosis"   # fill exact string as in your data
BASE_RESP = "IFNg release"                 # fill exact string as in your data
# BASE_MHC  = "HLA-DRB1*04:01"              # <-- EXAMPLE, replace with real one


SRC_OPTIONS_UI  = [SRC_PLACEHOLDER, BASE_SRC] + SRC_OPTIONS
MHC_OPTIONS_UI  = [MHC_PLACEHOLDER] + MHC_OPTIONS
RESP_OPTIONS_UI = [RESP_PLACEHOLDER, BASE_RESP] + RESP_OPTIONS


# ---------------------------
# 2. Feature engineering helpers
#    (must match training logic)
# ---------------------------

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

KD = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}

AA_MW = {
    'A': 89.09,  'C': 121.15, 'D': 133.10, 'E': 147.13, 'F': 165.19,
    'G': 75.07,  'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
}

AA_COMP_COLS = [f"AA_freq_{aa}" for aa in AA_LIST]
PHYSCHEM_COLS = [
    "seq_len_calc",
    "hydro_mean",
    "hydro_std",
    "mol_wt",
    "net_charge_approx",
    "aromatic_frac"
]

def aa_composition(seq: str):
    seq = seq.upper()
    L = len(seq)
    counts = {aa: seq.count(aa) for aa in AA_LIST}
    return [counts[aa] / L for aa in AA_LIST]

def physchem_features(seq: str):
    seq = seq.upper()
    L = len(seq)
    hydro_vals = [KD.get(a, 0.0) for a in seq]
    hydro_mean = float(np.mean(hydro_vals))
    hydro_std  = float(np.std(hydro_vals))
    mw = float(sum(AA_MW.get(a, 0.0) for a in seq))
    basic  = seq.count('K') + seq.count('R') + seq.count('H')
    acidic = seq.count('D') + seq.count('E')
    net_charge = float(basic - acidic)
    aromatic_count = seq.count('F') + seq.count('Y') + seq.count('W')
    aromatic_frac = float(aromatic_count / L)
    return [L, hydro_mean, hydro_std, mw, net_charge, aromatic_frac]

def make_feature_row(
    seq: str,
    src_choice: str,
    mhc_choice: str,
    resp_choice: str
) -> pd.DataFrame:
    """
    Build one-row feature DataFrame matching FEATURE_COLS.
    Includes:
      - sequence-derived numeric features
      - one-hot encoding for selected Source Organism / MHC / Response

    Baseline categories (BASE_SRC, BASE_MHC, BASE_RESP) are represented
    by all-zero dummy columns for that group, as in training.
    """
    seq = seq.strip().upper()

    # numeric features from sequence
    aac  = aa_composition(seq)
    phys = physchem_features(seq)
    num_feats = dict(zip(AA_COMP_COLS + PHYSCHEM_COLS, aac + phys))

    # start with all zeros for all features
    row = pd.DataFrame([[0.0] * len(FEATURE_COLS)], columns=FEATURE_COLS)

    # fill numeric columns
    for k, v in num_feats.items():
        if k in row.columns:
            row.at[0, k] = v

    # --- Source Organism one-hot ---
    # baseline (BASE_SRC) = all zeros, so do nothing
    if src_choice not in [SRC_PLACEHOLDER, BASE_SRC]:
        # one-hot others
        for col in FEATURE_COLS:
            if col.startswith(SRC_PREFIX):
                row.at[0, col] = 1.0 if col == SRC_PREFIX + src_choice else 0.0

    # --- MHC Present_mode one-hot ---
    if mhc_choice not in [MHC_PLACEHOLDER]:
        for col in FEATURE_COLS:
            if col.startswith(MHC_PREFIX):
                row.at[0, col] = 1.0 if col == MHC_PREFIX + mhc_choice else 0.0

    # --- Response_measured_mode one-hot ---
    if resp_choice not in [RESP_PLACEHOLDER, BASE_RESP]:
        for col in FEATURE_COLS:
            if col.startswith(RESP_PREFIX):
                row.at[0, col] = 1.0 if col == RESP_PREFIX + resp_choice else 0.0

    return row

# ---------------------------
# 3. Simple validators
# ---------------------------

VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

def validate_sequence(seq: str):
    seq = seq.strip().upper()
    if len(seq) == 0:
        return False, "Sequence is empty."
    if not VALID_AA_RE.match(seq):
        return False, "Sequence contains invalid characters. Use standard one-letter amino-acid codes."
    if not (13 <= len(seq) <= 25):
        return False, "Sequence length must be between 13 and 25 amino acids (MHC II-like)."
    return True, ""

# ---------------------------
# 4. Streamlit UI
# ---------------------------

st.set_page_config(
    page_title="MTB Epitope Immunogenicity Predictor",
    layout="wide"
)

st.title("MTB Epitope Immunogenicity Predictor")
st.caption("Machine learning–based prediction of T-cell immunogenicity for *Mycobacterium tuberculosis* peptides.")

st.markdown("---")

# Layout: two columns – left inputs, right outputs
col_left, col_right = st.columns([1.1, 1.3])

# ---------- LEFT: INPUTS ----------
with col_left:
    st.subheader("Single Peptide Prediction")

    seq_input = st.text_area(
        "Peptide sequence (13–25 aa)",
        height=90,
        placeholder="Enter MTB peptide sequence, e.g. QWERTYASDFGHKL"
    )

    st.markdown("**Experimental / biological context (optional):**")

    src_choice = st.selectbox(
        "Source organism",
        options=SRC_OPTIONS_UI
    )

    mhc_choice = st.selectbox(
        "MHC context",
        options=MHC_OPTIONS_UI
    )

    resp_choice = st.selectbox(
        "Response measured",
        options=RESP_OPTIONS_UI
    )


    run_single = st.button("Predict Immunogenicity", type="primary")

# ---------- RIGHT: OUTPUTS ----------
with col_right:
    st.subheader("Results")

    if run_single:
        is_valid, msg = validate_sequence(seq_input)
        if not is_valid:
            st.error(msg)
        else:
            feats = make_feature_row(seq_input, src_choice, mhc_choice, resp_choice)
            proba = rf_model.predict_proba(feats)[:, 1][0]
            label = int(proba >= DECISION_THR)
            label_str = "Immunogenic" if label == 1 else "Non-immunogenic"

            # Color-coded result
            if label == 1:
                st.success(f"Predicted: **{label_str}**")
            else:
                st.error(f"Predicted: **{label_str}**")

            st.write(f"Model probability (immunogenic): **{proba:.3f}**")
            st.write(f"Decision threshold used: **{DECISION_THR:.2f}**")

            st.markdown("**Interpretation (high level):**")
            st.markdown(
                "- Prediction is based on sequence composition and physicochemical properties "
                "(hydrophobicity, charge, molecular weight, aromaticity), together with the selected "
                "experimental context (source organism, MHC restriction, response measured).\n"
                "- The model was trained on curated MTB epitopes from IEDB and evaluated on a held-out test set.\n"
                "- One category per context variable is encoded as a baseline, which appears here as "
                "'baseline/unspecified'.\n"
                "- Values close to the threshold should be interpreted with caution."
            )

st.markdown("---")
st.subheader("Model summary")

st.markdown("---")
st.subheader("MTB epitope dataset browser (for testing)")

try:
    df_all = load_data()
except Exception as e:
    st.error(f"Could not load dataset from '{DATA_PATH}': {e}")
    df_all = None

if df_all is not None:
    # Basic columns we want to filter on
    col_qual = "Qualitative Measurement"
    col_resp = "Response measured"
    col_mhc  = "MHC Present"
    col_src  = "Source Organism"

    # ---- Filter widgets ----
    with st.expander("Filter & search epitopes", expanded=True):

        # 1) Free-text search (any column)
        search_text = st.text_input(
            "Search text (matches any column, case-insensitive)",
            value=""
        )

        # 2) Categorical filters
        qual_vals = sorted(df_all[col_qual].dropna().unique())
        resp_vals = sorted(df_all[col_resp].dropna().unique())
        mhc_vals  = sorted(df_all[col_mhc].dropna().unique())
        src_vals  = sorted(df_all[col_src].dropna().unique())

        sel_qual = st.multiselect(
            "Qualitative Measurement",
            options=qual_vals,
            default=qual_vals
        )

        sel_resp = st.multiselect(
            "Response measured",
            options=resp_vals,
            default=resp_vals
        )

        sel_mhc = st.multiselect(
            "MHC Present",
            options=mhc_vals,
            default=mhc_vals
        )

        sel_src = st.multiselect(
            "Source Organism",
            options=src_vals,
            default=src_vals
        )

    # ---- Apply filters ----
    df_view = df_all.copy()

    # categorical filters
    if sel_qual:
        df_view = df_view[df_view[col_qual].isin(sel_qual)]
    if sel_resp:
        df_view = df_view[df_view[col_resp].isin(sel_resp)]
    if sel_mhc:
        df_view = df_view[df_view[col_mhc].isin(sel_mhc)]
    if sel_src:
        df_view = df_view[df_view[col_src].isin(sel_src)]

    # text search across all columns (case-insensitive)
    if search_text.strip():
        pattern = search_text.strip().lower()
        mask = df_view.apply(
            lambda row: row.astype(str).str.lower().str.contains(pattern).any(),
            axis=1
        )
        df_view = df_view[mask]

    st.caption(f"Showing {len(df_view)} of {len(df_all)} rows")
    st.dataframe(df_view, use_container_width=True)


st.markdown(
    """
- **Model:** Random Forest (tuned)  
- **Decision threshold:** 0.45  
- **Test set performance (held-out MTB epitopes):**  
  - Accuracy: **0.975**  
  - Precision (positives): **0.903**  
  - Recall (positives): **0.613**  
  - F1-score: **0.730**  
  - ROC–AUC: **0.867**  
"""
)
