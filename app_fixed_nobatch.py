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
        return False, "Please enter a peptide between 13 and 25 amino acids long (13–25 letters)."
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
        "Peptide sequence",
        height=90,
        placeholder="Enter MTB peptide sequence, e.g. QWERTYASDFGHKL"
    )

    st.caption("Enter a peptide sequence **13–25 amino acids** long using one-letter codes (e.g., `QWERTYASDFGHKL`).")

    run_single = st.button("Predict Immunogenicity", type="primary")

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

            # Convert probability to a user-friendly percentage
            prob_pct = float(proba * 100)
            cutoff_pct = float(DECISION_THR * 100)

            st.markdown(f"**Model score:** {prob_pct:.1f}% (estimated probability of being immunogenic)")
            st.caption(f"Scores at or above **{cutoff_pct:.0f}%** are labelled *Immunogenic*; below this are *Non-immunogenic*.")

            # Visual probability bar (0–100)
            bar_color = "#16a34a" if label == 1 else "#dc2626"  # green / red
            st.markdown(
                f"""
<div style="margin-top:0.25rem;margin-bottom:0.75rem;">
  <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#6b7280;">
    <span>0</span><span>100</span>
  </div>
  <div style="background:#e5e7eb;border-radius:10px;height:18px;overflow:hidden;">
    <div style="width:{prob_pct:.1f}%;background:{bar_color};height:18px;"></div>
  </div>
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown("**Recommendations for Lab Validation**")
            if label == 1:
                st.success("High-priority candidate for ELISpot or Intracellular Cytokine Staining (ICS) validation.")
            else:
                st.warning("Low-priority. Consider screening alternate MHC-II contexts if MTb-specific response is still suspected.")
st.markdown("---")
st.subheader("Model information")

col_ms, col_int = st.columns(2)

with col_ms:
    st.markdown(
        f"""
- **Model:** Random Forest (tuned)  
- **Classification cutoff:** {DECISION_THR*100:.0f}%  
- **Test set performance (held-out MTB epitopes):**  
  - Accuracy: **0.975**  
  - Precision (positives): **0.903**  
  - Recall (positives): **0.613**  
  - F1-score: **0.730**  
  - ROC–AUC: **0.867**  
"""
    )

with col_int:
    st.markdown("**Interpretation (high level):**")
    st.markdown(
        """
- Predictions are based on amino-acid composition and physicochemical properties (e.g., hydrophobicity, charge, molecular weight, aromaticity).
- The model was trained on curated MTB epitopes from IEDB and evaluated on a held-out test set.
- Results close to the cutoff should be interpreted with caution.
"""
    )
