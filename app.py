import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

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

def make_feature_row(seq: str) -> pd.DataFrame:
    """Build one-row feature DataFrame matching FEATURE_COLS."""
    seq = seq.strip().upper()
    # numeric features
    aac = aa_composition(seq)
    phys = physchem_features(seq)
    num_feats = dict(zip(AA_COMP_COLS + PHYSCHEM_COLS, aac + phys))

    # start with all-zero row
    row = pd.DataFrame([0.0] * len(FEATURE_COLS), index=FEATURE_COLS).T

    # fill numeric columns we actually compute
    for k, v in num_feats.items():
        if k in row.columns:
            row.at[row.index[0], k] = v

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

    run_single = st.button("Predict Immunogenicity", type="primary")

    st.markdown("---")
    st.subheader("Batch Prediction (optional)")
    uploaded_file = st.file_uploader(
        "Upload CSV with column 'Sequence' for batch prediction",
        type=["csv"],
        help="Only the 'Sequence' column is required. Other columns will be ignored."
    )
    run_batch = st.button("Run Batch Prediction")

# ---------- RIGHT: OUTPUTS ----------
with col_right:
    st.subheader("Results")

    # Single prediction result
    if run_single:
        is_valid, msg = validate_sequence(seq_input)
        if not is_valid:
            st.error(msg)
        else:
            feats = make_feature_row(seq_input)
            proba = rf_model.predict_proba(feats)[:, 1][0]
            label = int(proba >= DECISION_THR)
            label_str = "Immunogenic" if label == 1 else "Non-immunogenic"

            st.success(f"Predicted: **{label_str}**")
            st.write(f"Model probability (immunogenic): **{proba:.3f}**")
            st.write(f"Decision threshold used: **{DECISION_THR:.2f}**")

            st.markdown("**Interpretation (high level):**")
            st.markdown(
                "- Prediction is based on sequence composition and physicochemical properties "
                "(hydrophobicity, charge, molecular weight, aromaticity).\n"
                "- The model was trained on curated MTB epitopes from IEDB "
                "and evaluated on a held-out test set.\n"
                "- Values close to the threshold should be interpreted with caution."
            )

    # Batch prediction result
    if run_batch:
        if uploaded_file is None:
            st.error("Please upload a CSV file first.")
        else:
            try:
                df_in = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df_in = None

            if df_in is not None:
                if "Sequence" not in df_in.columns:
                    st.error("CSV must contain a 'Sequence' column.")
                else:
                    st.info(f"Processing {len(df_in)} peptides...")
                    preds = []
                    probs = []
                    for seq in df_in["Sequence"].astype(str):
                        ok, _ = validate_sequence(seq)
                        if not ok:
                            preds.append(None)
                            probs.append(None)
                            continue
                        feats = make_feature_row(seq)
                        p = rf_model.predict_proba(feats)[:, 1][0]
                        probs.append(p)
                        preds.append(1 if p >= DECISION_THR else 0)

                    df_out = df_in.copy()
                    df_out["Predicted_label"] = preds
                    df_out["Predicted_label_str"] = df_out["Predicted_label"].map(
                        {1: "Immunogenic", 0: "Non-immunogenic", None: "Invalid sequence"}
                    )
                    df_out["Probability_immunogenic"] = probs

                    st.success("Batch prediction complete.")
                    st.dataframe(df_out.head(15))

                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download full predictions as CSV",
                        data=csv_bytes,
                        file_name="mtb_epitope_predictions.csv",
                        mime="text/csv"
                    )

st.markdown("---")
st.subheader("Model summary")

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
