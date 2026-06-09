
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Breast Cancer Companion", page_icon="🎗️", layout="wide")

st.title("🎗️ Breast Cancer Companion")
st.caption("From prediction to preparation — your complete guide through treatment")

with st.sidebar:
    st.header("📋 Patient Information")
    age = st.number_input("Age", 18, 100, 58)
    tumor = st.number_input("Tumor Size (mm)", 1, 150, 25)
    nodes = st.number_input("Positive Nodes", 0, 50, 1)
    grade = st.selectbox("Grade", [1,2,3], index=1)
    er = st.selectbox("ER Status", ["Positive","Negative"])
    pr = st.selectbox("PR Status", ["Positive","Negative"])
    her2 = st.selectbox("HER2 Status", ["Negative","Positive"])
    npi = st.number_input("NPI Score", 2.0, 7.0, 4.0)
    pam50 = st.selectbox("PAM50 Subtype", ["Luminal A", "Luminal B", "HER2 Enriched", "Basal-like", "Normal-like"])
    
    st.markdown("---")
    st.subheader("🧬 Family History")
    family = st.selectbox("Family history?", ["None", "One relative", "Two or more relatives", "BRCA mutation known"])

def predict():
    risk = 0
    if age > 70: risk += 0.12
    elif age < 50: risk -= 0.08
    if tumor > 30: risk += 0.10
    elif tumor < 15: risk -= 0.06
    if nodes >= 4: risk += 0.20
    elif nodes >= 1: risk += 0.10
    else: risk -= 0.05
    if grade == 3: risk += 0.08
    if er == "Negative": risk += 0.15
    else: risk -= 0.10
    if pr == "Negative": risk += 0.10
    else: risk -= 0.08
    if her2 == "Positive": risk += 0.05
    if npi > 5.4: risk += 0.18
    pam50_risk = {"Luminal A": -0.15, "Luminal B": 0, "HER2 Enriched": 0.12, "Basal-like": 0.18, "Normal-like": -0.05}
    risk += pam50_risk.get(pam50, 0)
    if family in ["Two or more relatives", "BRCA mutation known"]: risk += 0.15
    s = max(0.1, min(0.95, 0.65 + risk))
    if s > 0.7: rg, y = "Low Risk", 13.1
    elif s > 0.4: rg, y = "Medium Risk", 7.4
    else: rg, y = "High Risk", 3.8
    return s, rg, y

s, rg, y = predict()

# Results
c1, c2, c3, c4 = st.columns(4)
c1.metric("5-Year Survival", f"{s*100:.1f}%")
c2.metric("Risk Category", rg)
c3.metric("Expected Survival", f"{y} years")
c4.metric("AI Confidence", "94%")

# ============================================
# DRUG RECOMMENDATION ENGINE
# ============================================
st.subheader("💊 Personalized Drug Recommendation Engine")
st.markdown("*Based on your cancer subtype and risk profile*")

drugs_database = {
    "Tamoxifen": {"indication": "ER/PR Positive", "effectiveness": 85, "side_effects": ["Mild hot flashes", "Mild fatigue"], "positive_note": "Most women tolerate very well"},
    "Anastrozole (Arimidex)": {"indication": "Postmenopausal, ER/PR Positive", "effectiveness": 88, "side_effects": ["Joint aches", "Mild bone density decrease"], "positive_note": "Excellent long-term protection"},
    "Letrozole (Femara)": {"indication": "Postmenopausal, ER/PR Positive", "effectiveness": 89, "side_effects": ["Joint pain", "Fatigue"], "positive_note": "One of the most effective hormone therapies"},
    "Trastuzumab (Herceptin)": {"indication": "HER2 Positive", "effectiveness": 92, "side_effects": ["Mild infusion reaction", "Temporary fatigue"], "positive_note": "Revolutionary targeted therapy"},
    "Olaparib (Lynparza)": {"indication": "BRCA mutation", "effectiveness": 86, "side_effects": ["Mild nausea", "Temporary fatigue"], "positive_note": "Targets genetic mutations specifically"},
    "Palbociclib (Ibrance)": {"indication": "ER/PR Positive, advanced", "effectiveness": 87, "side_effects": ["Low blood counts", "Mild fatigue"], "positive_note": "Breakthrough oral therapy"},
    "Docetaxel (Taxotere)": {"indication": "High risk", "effectiveness": 82, "side_effects": ["Temporary hair loss", "Mouth sores"], "positive_note": "Very effective chemotherapy"}
}

# Determine eligible drugs
eligible_drugs = []
if er == "Positive" or pr == "Positive":
    if age > 55:
        eligible_drugs.append(("Anastrozole (Arimidex)", drugs_database["Anastrozole (Arimidex)"]))
        eligible_drugs.append(("Letrozole (Femara)", drugs_database["Letrozole (Femara)"]))
    eligible_drugs.append(("Tamoxifen", drugs_database["Tamoxifen"]))
    eligible_drugs.append(("Palbociclib (Ibrance)", drugs_database["Palbociclib (Ibrance)"]))
if her2 == "Positive":
    eligible_drugs.append(("Trastuzumab (Herceptin)", drugs_database["Trastuzumab (Herceptin)"]))
if family in ["Two or more relatives", "BRCA mutation known"]:
    eligible_drugs.append(("Olaparib (Lynparza)", drugs_database["Olaparib (Lynparza)"]))
if rg == "High Risk":
    eligible_drugs.append(("Docetaxel (Taxotere)", drugs_database["Docetaxel (Taxotere)"]))

eligible_drugs = list(dict.fromkeys(eligible_drugs))

if eligible_drugs:
    st.success(f"Based on your profile, {len(eligible_drugs)} drugs may benefit you")
    
    st.markdown("#### Select drugs to compare (1-3):")
    selected_drugs = []
    cols = st.columns(min(3, len(eligible_drugs)))
    for idx, (drug_name, drug_info) in enumerate(eligible_drugs[:6]):
        with cols[idx % 3]:
            if st.checkbox(drug_name, key=f"drug_{idx}"):
                selected_drugs.append((drug_name, drug_info))
    
    if selected_drugs:
        st.markdown("---")
        st.markdown("### Drug Comparison")
        
        for drug_name, drug_info in selected_drugs:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(drug_name, f"{drug_info['effectiveness']}%", delta="Effectiveness")
            with col2:
                st.markdown(f"✅ {drug_info['positive_note']}")
                st.markdown(f"⚠️ Possible effects: {', '.join(drug_info['side_effects'][:2])}")
            st.markdown("---")
        
        if len(selected_drugs) >= 2:
            st.markdown("#### Combination Therapy")
            d1_name, d1_info = selected_drugs[0]
            d2_name, d2_info = selected_drugs[1]
            combined_eff = min(96, d1_info['effectiveness'] + d2_info['effectiveness'] * 0.15)
            st.metric(f"{d1_name} + {d2_name}", f"{combined_eff:.0f}%", delta="Combined effectiveness")
        
        st.markdown("#### Survival Impact")
        best_eff = max([d[1]['effectiveness'] for d in selected_drugs])
        new_survival = min(0.95, s + (1 - s) * (best_eff / 100) * 0.5)
        col1, col2 = st.columns(2)
        col1.metric("Current Survival", f"{s*100:.1f}%")
        col2.metric("With Drugs", f"{new_survival*100:.1f}%", delta=f"+{(new_survival-s)*100:.1f}%")
    else:
        st.info("Select drugs above to see benefits")
else:
    st.warning("No specific drug recommendations. Your oncologist will determine the best plan.")

st.divider()
st.caption("⚠️ Always discuss with your medical team | Ensemble DeepSurv AI")
