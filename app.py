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

SRC_PLACEHOLDER  = "Select source organism"
MHC_PLACEHOLDER  = "Select MHC context"
RESP_PLACEHOLDER = "Select response type"

SRC_PREFIX  = "Source Organism_"
MHC_PREFIX  = "MHC Present_mode_"
RESP_PREFIX = "Response_measured_mode_"

SRC_OPTIONS  = sorted([c[len(SRC_PREFIX):]  for c in FEATURE_COLS if c.startswith(SRC_PREFIX)])
MHC_OPTIONS  = sorted([c[len(MHC_PREFIX):]  for c in FEATURE_COLS if c.startswith(MHC_PREFIX)])
RESP_OPTIONS = sorted([c[len(RESP_PREFIX):] for c in FEATURE_COLS if c.startswith(RESP_PREFIX)])

BASE_SRC  = "Mycobacterium tuberculosis"   
BASE_RESP = "IFNg release"                 

SRC_OPTIONS_UI  = [SRC_PLACEHOLDER, BASE_SRC] + SRC_OPTIONS
MHC_OPTIONS_UI  = [MHC_PLACEHOLDER] + MHC_OPTIONS
RESP_OPTIONS_UI = [RESP_PLACEHOLDER, BASE_RESP] + RESP_OPTIONS

# ---------------------------
# 2. Feature engineering helpers
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
PHYSCHEM_COLS = ["seq_len_calc", "hydro_mean", "hydro_std", "mol_wt", "net_charge_approx", "aromatic_frac"]

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

def make_feature_row(seq, src_choice, mhc_choice, resp_choice):
    seq = seq.strip().upper()
    aac  = aa_composition(seq)
    phys = physchem_features(seq)
    num_feats = dict(zip(AA_COMP_COLS + PHYSCHEM_COLS, aac + phys))
    row = pd.DataFrame([[0.0] * len(FEATURE_COLS)], columns=FEATURE_COLS)
    for k, v in num_feats.items():
        if k in row.columns:
            row.at[0, k] = v
    if src_choice not in [SRC_PLACEHOLDER, BASE_SRC]:
        for col in FEATURE_COLS:
            if col.startswith(SRC_PREFIX):
                row.at[0, col] = 1.0 if col == SRC_PREFIX + src_choice else 0.0
    if mhc_choice not in [MHC_PLACEHOLDER]:
        for col in FEATURE_COLS:
            if col.startswith(MHC_PREFIX):
                row.at[0, col] = 1.0 if col == MHC_PREFIX + mhc_choice else 0.0
    if resp_choice not in [RESP_PLACEHOLDER, BASE_RESP]:
        for col in FEATURE_COLS:
            if col.startswith(RESP_PREFIX):
                row.at[0, col] = 1.0 if col == RESP_PREFIX + resp_choice else 0.0
    return row

VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
def validate_sequence(seq: str):
    seq = seq.strip().upper()
    if len(seq) == 0:
        return False, "Sequence is empty."
    if not VALID_AA_RE.match(seq):
        return False, "Sequence contains invalid characters. Use standard one-letter amino-acid codes."
    if not (13 <= len(seq) <= 25):
        return False, "Please enter a peptide between 13 and 25 amino acids (standard length for MHC-II epitopes)."
    return True, ""

# ---------------------------
# 3. UI Helper Functions (The Toggle Features)
# ---------------------------

def append_aa(aa_char):
    """Callback to append a character to the session state text."""
    if "seq_input_key" not in st.session_state:
        st.session_state.seq_input_key = ""
    st.session_state.seq_input_key += aa_char

def render_aa_buttons():
    """Renders a grid of buttons to append amino acids."""
    st.markdown("**Quick Add Amino Acid:**")
    
    # Define AA data for tooltips
    aa_data = [
        ('A', 'Alanine'), ('C', 'Cysteine'), ('D', 'Aspartic Acid'), ('E', 'Glutamic Acid'),
        ('F', 'Phenylalanine'), ('G', 'Glycine'), ('H', 'Histidine'), ('I', 'Isoleucine'),
        ('K', 'Lysine'), ('L', 'Leucine'), ('M', 'Methionine'), ('N', 'Asparagine'),
        ('P', 'Proline'), ('Q', 'Glutamine'), ('R', 'Arginine'), ('S', 'Serine'),
        ('T', 'Threonine'), ('V', 'Valine'), ('W', 'Tryptophan'), ('Y', 'Tyrosine')
    ]
    
    # Create 5 columns for the buttons
    cols = st.columns(5)
    for i, (code, name) in enumerate(aa_data):
        col_idx = i % 5
        with cols[col_idx]:
            st.button(
                code, 
                key=f"btn_{code}", 
                help=name, 
                on_click=append_aa, 
                args=(code,),
                use_container_width=True
            )

def render_aa_reference_table():
    """Renders an expandable table of AA properties."""
    with st.expander("Amino Acid Reference & Properties"):
        data = [
            {"Code": "A", "3-Letter": "Ala", "Name": "Alanine", "Type": "Nonpolar, aliphatic"},
            {"Code": "C", "3-Letter": "Cys", "Name": "Cysteine", "Type": "Polar, uncharged"},
            {"Code": "D", "3-Letter": "Asp", "Name": "Aspartic Acid", "Type": "Acidic, negatively charged"},
            {"Code": "E", "3-Letter": "Glu", "Name": "Glutamic Acid", "Type": "Acidic, negatively charged"},
            {"Code": "F", "3-Letter": "Phe", "Name": "Phenylalanine", "Type": "Aromatic, nonpolar"},
            {"Code": "G", "3-Letter": "Gly", "Name": "Glycine", "Type": "Nonpolar, aliphatic"},
            {"Code": "H", "3-Letter": "His", "Name": "Histidine", "Type": "Basic, positively charged"},
            {"Code": "I", "3-Letter": "Ile", "Name": "Isoleucine", "Type": "Nonpolar, aliphatic"},
            {"Code": "K", "3-Letter": "Lys", "Name": "Lysine", "Type": "Basic, positively charged"},
            {"Code": "L", "3-Letter": "Leu", "Name": "Leucine", "Type": "Nonpolar, aliphatic"},
            {"Code": "M", "3-Letter": "Met", "Name": "Methionine", "Type": "Nonpolar, sulfur-containing"},
            {"Code": "N", "3-Letter": "Asn", "Name": "Asparagine", "Type": "Polar, uncharged"},
            {"Code": "P", "3-Letter": "Pro", "Name": "Proline", "Type": "Nonpolar, cyclic"},
            {"Code": "Q", "3-Letter": "Gln", "Name": "Glutamine", "Type": "Polar, uncharged"},
            {"Code": "R", "3-Letter": "Arg", "Name": "Arginine", "Type": "Basic, positively charged"},
            {"Code": "S", "3-Letter": "Ser", "Name": "Serine", "Type": "Polar, uncharged"},
            {"Code": "T", "3-Letter": "Thr", "Name": "Threonine", "Type": "Polar, uncharged"},
            {"Code": "V", "3-Letter": "Val", "Name": "Valine", "Type": "Nonpolar, aliphatic"},
            {"Code": "W", "3-Letter": "Trp", "Name": "Tryptophan", "Type": "Aromatic, nonpolar"},
            {"Code": "Y", "3-Letter": "Tyr", "Name": "Tyrosine", "Type": "Aromatic, polar"},
        ]
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.set_page_config(page_title="MTB Epitope Immunogenicity Predictor", layout="wide")

st.title("MTB Epitope Immunogenicity Predictor")
st.caption("Machine learning–based prediction of T-cell immunogenicity for *Mycobacterium tuberculosis* peptides.")

st.markdown("---")

col_left, col_right = st.columns([1.1, 1.3])

# --- LEFT COLUMN ---
with col_left:
    st.subheader("Single Peptide Prediction")

    # Ensure session state is initialized for the text input
    if "seq_input_key" not in st.session_state:
        st.session_state.seq_input_key = ""

    # The text area is bound to session state so buttons can update it
    seq_input = st.text_area(
        "Peptide sequence",
        key="seq_input_key",
        height=90,
        placeholder="Enter MTB peptide sequence...",
        help="Peptides should be within the 13–25 amino acid range."
    )

    # ==========================================
    # TOGGLE FEATURE: AA INPUT BUTTONS
    # Comment out the line below to hide buttons
    # ==========================================
    render_aa_buttons()
    # ==========================================

    st.markdown("---")
    st.markdown("**Experimental / biological context (optional):**")

    src_choice = st.selectbox("Source organism", options=SRC_OPTIONS_UI)
    mhc_choice = st.selectbox("MHC context", options=MHC_OPTIONS_UI)
    resp_choice = st.selectbox("Response measured", options=RESP_OPTIONS_UI)

    run_single = st.button("Predict Immunogenicity", type="primary")

# --- RIGHT COLUMN ---
with col_right:
    st.subheader("Results")

    if run_single:
        # Retrieve the current value from session state or the variable
        current_seq = st.session_state.seq_input_key 
        is_valid, msg = validate_sequence(current_seq)
        
        if not is_valid:
            st.error(msg)
        else:
            feats = make_feature_row(current_seq, src_choice, mhc_choice, resp_choice)
            proba = rf_model.predict_proba(feats)[:, 1][0]
            proba_pct = proba * 100
            cutoff_pct = DECISION_THR * 100
            label = int(proba >= DECISION_THR)
            
            label_str = "Immunogenic" if label == 1 else "Non-immunogenic"

            if label == 1:
                st.success(f"Predicted: **{label_str}**")
            else:
                st.error(f"Predicted: **{label_str}**")

            st.markdown(f"**Model Score: {proba_pct:.1f}%**")
            st.progress(proba) 

            st.markdown(f"**Classification cutoff: {cutoff_pct:.0f}%**", 
                        help="Peptides scoring above this percentage are considered likely to trigger a biological immune response.")

            st.markdown("---")
            st.subheader("Recommendations for Lab Validation")
            if label == 1:
                st.info("💡 **High-priority candidate** for ELISpot or Intracellular Cytokine Staining (ICS) validation.")
            else:
                st.warning("⚠️ **Low-priority.** Consider screening alternate MHC-II contexts if MTb-specific response is still suspected.")

st.markdown("---")

# --- BOTTOM SECTION ---
col_info_1, col_info_2 = st.columns(2)

with col_info_1:
    st.subheader("Model Summary")
    st.markdown(
        f"""
    - **Algorithm:** Tuned Random Forest Classifier  
    - **Cutoff:** {DECISION_THR:.2f} (Scores above this label as Positive)
    - **Test Set Performance:**
      - Accuracy: **0.975**
      - Precision: **0.903**
      - Recall: **0.613**
      - ROC–AUC: **0.867**
    """
    )

with col_info_2:
    st.subheader("Interpretation (High Level)")
    st.markdown(
        """
        - **Biophysical Basis:** Prediction is derived from sequence composition and physicochemical properties such as hydrophobicity, net charge, and aromaticity.
        - **Biological Context:** The model factors in selected MHC restriction and experimental response types.
        - **Training Data:** Built using curated *Mycobacterium tuberculosis* epitopes from the IEDB database.
        - **Note:** Values very close to the cutoff should be treated with additional biological caution.
        """
    )

# ==========================================
# TOGGLE FEATURE: AA REFERENCE TABLE
# Comment out the line below to hide the table
# ==========================================
render_aa_reference_table()
# ==========================================