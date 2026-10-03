"""
Aman: AI-Powered Pharmaceutical Supply Chain Integrity
Streamlit Prototype for Presight Innovation Challenge

Models trained at startup on synthetic data with non-linear interactions.
Benford's Law and MKT are real formulas. Random seed fixed for reproducibility.
"""

import math
import random
import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import comtradeapicall

random.seed(10)
np.random.seed(10)

st.set_page_config(page_title="Aman - Pharma Supply Chain Integrity", layout="wide")

# ============================================================================
# PORT COORDINATES [longitude, latitude]
# ============================================================================

PORTS = {
    "Mumbai, India": [72.8777, 19.0760],
    "Shanghai, China": [121.4737, 31.2304],
    "Singapore": [103.8198, 1.3521],
    "Hong Kong": [114.1694, 22.3193],
    "Istanbul, Turkey": [28.9784, 41.0082],
    "Cairo, Egypt": [31.2357, 30.0444],
    "Jebel Ali, UAE": [55.0617, 25.0118],
    "Abu Dhabi, UAE": [54.3773, 24.4539],
    "Dubai Airport, UAE": [55.3657, 25.2532],
    "Sharjah, UAE": [55.4209, 25.3463],
}

# ============================================================================
# REAL FORMULAS
# ============================================================================

def document_integrity_check(declared_values):
    if len(declared_values) < 10:
        return 0.0
    expected = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
    leading = [int(str(int(abs(v)))[0]) for v in declared_values]
    n = len(leading)
    observed = {d: leading.count(d) / n for d in range(1, 10)}
    deviation = sum(abs(observed[d] - expected[d]) for d in range(1, 10))
    return min(deviation / 1.0, 1.0)


def cold_chain_integrity_check(temp_log, safe_threshold_c=8.0):
    if not temp_log:
        return 0.0
    Ea = 83144
    R = 8.314462
    kelvin = [t + 273.15 for t in temp_log]
    sum_exp = sum(math.exp(-Ea / (R * t)) for t in kelvin)
    avg_exp = sum_exp / len(kelvin)
    mkt_c = (-Ea / (R * math.log(avg_exp))) - 273.15
    if mkt_c <= safe_threshold_c:
        return 0.0
    return min((mkt_c - safe_threshold_c) / 5.0, 1.0)


def route_integrity_check(planned_hours, actual_hours):
    deviation = abs(actual_hours - planned_hours) / planned_hours
    return min(deviation / 0.5, 1.0)


def custody_integrity_check(openings):
    return min(openings * 0.6, 1.0)

@st.cache_data(ttl=86400)
def fetch_real_uae_pharma_imports(api_key):
    """Fetches real UAE pharmaceutical import values (HS Chapter 30) from UN Comtrade."""
    try:
        df = comtradeapicall.getFinalData(
            api_key,
            typeCode='C',
            freqCode='A',
            clCode='HS',
            period='2023',
            reporterCode='784',   # UAE
            cmdCode='30',         # HS Chapter 30 = Pharmaceuticals
            flowCode='M',         # Imports
            partnerCode='0',      # World
            maxRecords=500,
        )
        if df is None or df.empty:
            return []
        values = df['primaryValue'].dropna().tolist()
        return [v for v in values if v > 0]
    except Exception:
        return []


def benford_check_real(real_values):
    if len(real_values) < 10:
        return None
    expected = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
    leading = [int(str(int(abs(v)))[0]) for v in real_values if int(abs(v)) > 0]
    n = len(leading)
    observed = {d: leading.count(d) / n for d in range(1, 10)}
    deviation = sum(abs(observed[d] - expected[d]) for d in range(1, 10))
    return {"deviation": round(deviation, 4), "n_records": n}
# ============================================================================
# SYNTHETIC VALUE GENERATORS
# ============================================================================

def generate_natural_values(n=100):
    values = []
    for _ in range(n):
        magnitude = random.choice([10, 100, 1000, 10000])
        coeff = 10 ** random.uniform(0, 1)
        values.append(round(coeff * magnitude, 2))
    return values


def generate_suspicious_values(n=100):
    return [round(random.uniform(8000, 9999), 2) for _ in range(n)]


def generate_mildly_off_values(n=100):
    """Values that are slightly off Benford but not extreme."""
    values = []
    for _ in range(n):
        magnitude = random.choice([100, 1000])
        coeff = random.uniform(3, 7)  # narrow band, mild Benford deviation
        values.append(round(coeff * magnitude, 2))
    return values


# ============================================================================
# MODEL TRAINING (at startup, cached)
# ============================================================================

@st.cache_resource
def train_models():
    rng = np.random.default_rng(42)

    # --- Supplier risk model (XGBoost) ---
    # Features: [violations, years_active, shipment_volume]
    n_suppliers = 800
    violations = rng.integers(0, 10, n_suppliers)
    years_active = rng.integers(1, 30, n_suppliers)
    volume = rng.integers(10, 1000, n_suppliers)

    X_supplier = np.column_stack([violations, years_active, volume])

    # Non-linear label: risk grows with violations, shrinks with years active,
    # grows with volume, plus interaction between violations and newness
    risk_signal = (
        violations * 0.35
        + (30 - years_active) * 0.02
        + np.log1p(volume) * 0.08
        + violations * (30 - years_active) * 0.005
        + rng.normal(0, 0.4, n_suppliers)
    )
    y_supplier = (risk_signal > 1.5).astype(int)

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_supplier, y_supplier, test_size=0.2, random_state=42
    )

    xgb_model = XGBClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", verbosity=0,
    )
    xgb_model.fit(Xs_train, ys_train)

    supplier_acc = accuracy_score(ys_test, xgb_model.predict(Xs_test))
    supplier_auc = roc_auc_score(ys_test, xgb_model.predict_proba(Xs_test)[:, 1])

    # --- Fusion model (Logistic Regression) ---
    # Features: [doc, supp, cold, route, cust]
    n_shipments = 1500
    doc = rng.uniform(0, 1, n_shipments)
    supp = rng.uniform(0, 1, n_shipments)
    cold = rng.uniform(0, 1, n_shipments)
    route = rng.uniform(0, 1, n_shipments)
    cust = rng.uniform(0, 1, n_shipments)

    
    # DECOUPLED LABELS: ground truth comes from a latent "true risk" process,
    # not from the same features the model trains on. This forces the model
    # to INFER, not invert a known formula.
    #
    # Latent true risk (unobserved by the model) — different structure than the features
    latent_risk = (
        0.30 * rng.beta(2, 5, n_shipments)     # base population risk
        + 0.25 * rng.binomial(1, 0.15, n_shipments)  # random "incident" flag
        + 0.20 * rng.exponential(0.5, n_shipments).clip(0, 2)  # occasional severe event
    )

    # Observed features are CORRUPTED, NOISY proxies of the latent risk
    # (in reality, sensors and paperwork imperfectly reflect what's happening)
    doc_obs = np.clip(latent_risk + rng.normal(0, 0.25, n_shipments), 0, 1)
    supp_obs = np.clip(latent_risk * 0.8 + rng.normal(0, 0.3, n_shipments), 0, 1)
    cold_obs = np.clip(latent_risk * 1.2 + rng.normal(0, 0.35, n_shipments), 0, 1)
    route_obs = np.clip(latent_risk * 0.5 + rng.normal(0, 0.4, n_shipments), 0, 1)
    cust_obs = np.clip(latent_risk * 0.7 + rng.normal(0, 0.3, n_shipments), 0, 1)

    X_fusion = np.column_stack([doc_obs, supp_obs, cold_obs, route_obs, cust_obs])

    # Label: did the shipment actually turn out to be a problem?
    # Depends on LATENT risk (with noise), not on the observed features directly
    y_fusion = (latent_risk + rng.normal(0, 0.25, n_shipments) > 0.55).astype(int)

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        X_fusion, y_fusion, test_size=0.2, random_state=42
    )

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(Xf_train, yf_train)

    fusion_acc = accuracy_score(yf_test, log_model.predict(Xf_test))
    fusion_auc = roc_auc_score(yf_test, log_model.predict_proba(Xf_test)[:, 1])

        # Precision@k — the metric that matters for triage
    y_proba = log_model.predict_proba(Xf_test)[:, 1]
    order = np.argsort(-y_proba)
    k10 = max(1, int(0.1 * len(order)))   # top 10%
    k20 = max(1, int(0.2 * len(order)))   # top 20%
    precision_at_10 = yf_test[order[:k10]].mean()
    precision_at_20 = yf_test[order[:k20]].mean()

    return {
        "xgb": xgb_model,
        "log": log_model,
        "supplier_acc": supplier_acc,
        "supplier_auc": supplier_auc,
        "fusion_acc": fusion_acc,
        "fusion_auc": fusion_auc,
        "precision_at_10": precision_at_10,
        "precision_at_20": precision_at_20,
    }


models = train_models()
xgb_model = models["xgb"]
log_model = models["log"]


# ============================================================================
# PIPELINE
# ============================================================================

def supplier_risk_score(violations, years_active, volume):
    features = np.array([[violations, years_active, volume]])
    return float(xgb_model.predict_proba(features)[0][1])


def fuse_risk_scores(doc, supp, cold, route, cust):
    features = np.array([[doc, supp, cold, route, cust]])
    return float(log_model.predict_proba(features)[0][1])


def get_flag_reasons(doc, supp, cold, route, cust, threshold=0.5):
    reasons = []
    if doc > threshold: reasons.append("Document anomaly")
    if supp > threshold: reasons.append("Supplier history")
    if cold > threshold: reasons.append("Cold-chain breach")
    if route > threshold: reasons.append("Route deviation")
    if cust > threshold: reasons.append("Custody breach")
    return reasons if reasons else ["None"]


# ============================================================================
# SHIPMENT DATABASE (natural gradient of risk)
# ============================================================================

if "shipments" not in st.session_state:
    st.session_state.shipments = [
        # Clean shipments
        {"id": "SH-1001", "note": "Clean shipment", "origin": "Mumbai, India",
         "destination": "Jebel Ali, UAE", "violations": 0, "years_active": 18, "volume": 400,
         "values": generate_natural_values(), "temp_log": [4,5,4,5,6,5,4],
         "planned": 48, "actual": 47, "openings": 0},

        {"id": "SH-1002", "note": "Clean shipment", "origin": "Singapore",
         "destination": "Jebel Ali, UAE", "violations": 0, "years_active": 22, "volume": 600,
         "values": generate_natural_values(), "temp_log": [4,4,5,5,4,4,5],
         "planned": 48, "actual": 46, "openings": 0},

        # Mild concerns
        {"id": "SH-1003", "note": "Minor route delay", "origin": "Istanbul, Turkey",
         "destination": "Jebel Ali, UAE", "violations": 0, "years_active": 15, "volume": 350,
         "values": generate_natural_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 56, "openings": 0},

        {"id": "SH-1004", "note": "Slightly elevated supplier risk", "origin": "Mumbai, India",
         "destination": "Sharjah, UAE", "violations": 1, "years_active": 8, "volume": 500,
         "values": generate_natural_values(), "temp_log": [5,6,5,6,5,6,5],
         "planned": 48, "actual": 49, "openings": 0},

        # Moderate concerns
        {"id": "SH-1005", "note": "Mild cold-chain excursion", "origin": "Singapore",
         "destination": "Abu Dhabi, UAE", "violations": 0, "years_active": 12, "volume": 450,
         "values": generate_natural_values(), "temp_log": [4,5,6,9,10,7,5],
         "planned": 48, "actual": 50, "openings": 0},

        {"id": "SH-1006", "note": "Slight document anomaly", "origin": "Shanghai, China",
         "destination": "Jebel Ali, UAE", "violations": 1, "years_active": 10, "volume": 700,
         "values": generate_mildly_off_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 48, "openings": 0},

        {"id": "SH-1007", "note": "Container opened once", "origin": "Hong Kong",
         "destination": "Dubai Airport, UAE", "violations": 0, "years_active": 14, "volume": 300,
         "values": generate_natural_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 47, "openings": 1},

        # High concerns
        {"id": "SH-1008", "note": "Cold-chain breach", "origin": "Cairo, Egypt",
         "destination": "Jebel Ali, UAE", "violations": 1, "years_active": 9, "volume": 550,
         "values": generate_natural_values(), "temp_log": [4,5,6,15,18,16,5],
         "planned": 48, "actual": 48, "openings": 0},

        {"id": "SH-1009", "note": "Fabricated paperwork", "origin": "Shanghai, China",
         "destination": "Jebel Ali, UAE", "violations": 2, "years_active": 6, "volume": 800,
         "values": generate_suspicious_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 49, "openings": 0},

        {"id": "SH-1010", "note": "Major route deviation", "origin": "Istanbul, Turkey",
         "destination": "Jebel Ali, UAE", "violations": 1, "years_active": 11, "volume": 500,
         "values": generate_natural_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 84, "openings": 0},

                {"id": "SH-1011", "note": "Two mild issues", "origin": "Mumbai, India",
         "destination": "Jebel Ali, UAE", "violations": 1, "years_active": 9, "volume": 500,
         "values": generate_mildly_off_values(), "temp_log": [5,6,5,6,5,6,5],
         "planned": 48, "actual": 52, "openings": 0},

        {"id": "SH-1012", "note": "Moderate cold + mild route", "origin": "Singapore",
         "destination": "Abu Dhabi, UAE", "violations": 1, "years_active": 10, "volume": 600,
         "values": generate_natural_values(), "temp_log": [4,6,8,11,12,9,5],
         "planned": 48, "actual": 58, "openings": 0},

        {"id": "SH-1013", "note": "Mild custody + document issues", "origin": "Hong Kong",
         "destination": "Dubai Airport, UAE", "violations": 2, "years_active": 7, "volume": 700,
         "values": generate_mildly_off_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 50, "openings": 1},

        # Worst case
        {"id": "SH-1014", "note": "Multiple red flags", "origin": "Cairo, Egypt",
         "destination": "Jebel Ali, UAE", "violations": 4, "years_active": 3, "volume": 900,
         "values": generate_suspicious_values(), "temp_log": [4,20,22,19,18,17,5],
         "planned": 48, "actual": 90, "openings": 2},
    ]

# ============================================================================
# HEADER + SIDEBAR
# ============================================================================

st.title("Aman: Pharmaceutical Supply Chain Integrity")
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))
st.markdown("*AI-powered risk scoring for customs inspection prioritization*")
st.divider()

st.sidebar.header("Controls")

st.sidebar.subheader("Model Performance")
st.sidebar.metric("Precision @ Top 10", f"{models['precision_at_10']:.1%}")
st.sidebar.metric("Precision @ Top 20", f"{models['precision_at_20']:.1%}")
st.sidebar.metric("Fusion Model AUC", f"{models['fusion_auc']:.2f}")
st.sidebar.caption("Precision@k = of the top-k flagged shipments, how many were real problems?")

st.sidebar.divider()
st.sidebar.subheader("Flag Threshold")
flag_threshold = st.sidebar.slider("Flag if score above", 0.0, 1.0, 0.5, 0.05)

st.sidebar.divider()
st.sidebar.subheader("Real Data Validation")
api_key = st.sidebar.text_input("UN Comtrade API key", type="password")

if api_key:
    real_values = fetch_real_uae_pharma_imports(api_key)
    if real_values:
        result = benford_check_real(real_values)
        if result:
            st.sidebar.success(f"Validated on {result['n_records']} real records")
            st.sidebar.metric("Benford deviation (real)", result["deviation"])
        else:
            st.sidebar.warning("Not enough records.")
    else:
        st.sidebar.warning("No data — check key.")

st.sidebar.divider()
st.sidebar.subheader("Add New Shipment")

with st.sidebar.form("add_shipment"):
    new_id = st.text_input("Shipment ID", "SH-1012")
    new_note = st.text_input("Scenario note", "Custom entry")
    new_origin = st.selectbox("Origin", list(PORTS.keys()), index=0)
    new_dest = st.selectbox("Destination", list(PORTS.keys()), index=6)
    new_violations = st.number_input("Supplier violations", 0, 10, 0)
    new_years = st.number_input("Supplier years active", 1, 50, 15)
    new_volume = st.number_input("Shipment volume", 10, 2000, 400)
    new_openings = st.number_input("Unauthorized openings", 0, 10, 0)
    new_planned = st.number_input("Planned hours", 1, 200, 48)
    new_actual = st.number_input("Actual hours", 1, 200, 48)
    value_type = st.selectbox("Declared values", ["Natural (Benford-compliant)", "Suspicious (fabricated)", "Mildly off"])
    temp_type = st.selectbox("Temperature log", ["Normal (2-8°C)", "Mild excursion (9-12°C)", "Breach (spike to 20°C)"])

    submitted = st.form_submit_button("Add Shipment")

    if submitted:
        if "Natural" in value_type:
            vals = generate_natural_values()
        elif "Suspicious" in value_type:
            vals = generate_suspicious_values()
        else:
            vals = generate_mildly_off_values()

        if "Normal" in temp_type:
            temps = [4,5,4,5,4,5,4]
        elif "Mild" in temp_type:
            temps = [4,5,6,9,10,7,5]
        else:
            temps = [4,20,22,19,18,17,5]

        st.session_state.shipments.append({
            "id": new_id, "note": new_note,
            "origin": new_origin, "destination": new_dest,
            "violations": new_violations, "years_active": new_years, "volume": new_volume,
            "values": vals, "temp_log": temps,
            "planned": new_planned, "actual": new_actual, "openings": new_openings,
        })
        st.session_state["flash"] = f"Added {new_id}"
        st.rerun()

if st.sidebar.button("Reset to default shipments"):
    st.session_state.shipments = st.session_state.shipments[:14]
    st.sidebar.success("Reset")

# ============================================================================
# RUN PIPELINE
# ============================================================================

def run_pipeline(s):
    doc = document_integrity_check(s["values"])
    supp = supplier_risk_score(s["violations"], s["years_active"], s["volume"])
    cold = cold_chain_integrity_check(s["temp_log"])
    route = route_integrity_check(s["planned"], s["actual"])
    cust = custody_integrity_check(s["openings"])
    fused = fuse_risk_scores(doc, supp, cold, route, cust)
    reasons = get_flag_reasons(doc, supp, cold, route, cust, flag_threshold)
    return {
        "Shipment": s["id"],
        "Scenario": s["note"],
        "Origin": s["origin"],
        "Destination": s["destination"],
        "Risk Score": round(fused, 3),
        "Flagged": "YES" if fused >= flag_threshold else "No",
        "Reasons": ", ".join(reasons),
        "Document": round(doc, 2),
        "Supplier": round(supp, 2),
        "Cold-Chain": round(cold, 2),
        "Route": round(route, 2),
        "Custody": round(cust, 2),
    }


results = [run_pipeline(s) for s in st.session_state.shipments]
df = pd.DataFrame(results).sort_values("Risk Score", ascending=False).reset_index(drop=True)

# ============================================================================
# METRICS
# ============================================================================

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Shipments", len(df))
col2.metric("Flagged for Inspection", len(df[df["Flagged"] == "YES"]))
col3.metric("Average Risk Score", round(df["Risk Score"].mean(), 3))
col4.metric("Highest Risk", df["Risk Score"].max())

st.divider()

# ============================================================================
# MAP
# ============================================================================

st.subheader("Global Shipment Routes")

arc_data = []
for _, row in df.iterrows():
    origin_coords = PORTS.get(row["Origin"])
    dest_coords = PORTS.get(row["Destination"])
    if origin_coords and dest_coords:
        color = [220, 50, 50] if row["Flagged"] == "YES" else [50, 180, 90]
        arc_data.append({
            "origin": origin_coords, "destination": dest_coords,
            "color": color, "shipment": row["Shipment"],
            "risk": row["Risk Score"],
            "origin_name": row["Origin"],
            "dest_name": row["Destination"],
        })

arc_df = pd.DataFrame(arc_data)

if not arc_df.empty:
    arc_layer = pdk.Layer(
        "ArcLayer", data=arc_df,
        get_source_position="origin", get_target_position="destination",
        get_source_color="color", get_target_color="color",
        get_width=3, get_height=0.3, pickable=True,
    )
    view_state = pdk.ViewState(latitude=20, longitude=70, zoom=2, pitch=0)
    st.pydeck_chart(pdk.Deck(
        layers=[arc_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>{shipment}</b><br/>{origin_name} → {dest_name}<br/>Risk: {risk}",
            "style": {"backgroundColor": "white", "color": "black"},
        },
    ))
    st.caption("Red arcs = flagged shipments | Green arcs = cleared shipments")

st.divider()

# ============================================================================
# RANKED TABLE
# ============================================================================

st.subheader("Shipment Risk Ranking (highest risk first)")

def color_risk(val):
    if val >= flag_threshold:
        return "background-color: #ffcccc"
    elif val >= flag_threshold * 0.6:
        return "background-color: #fff4cc"
    return "background-color: #ccffcc"

styled = df.style.map(color_risk, subset=["Risk Score"])
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ============================================================================
# WHAT THE MODEL LEARNED
# ============================================================================

st.subheader("What the Model Learned")

feature_names = ["Document", "Supplier", "Cold-Chain", "Route", "Custody"]
coefficients = log_model.coef_[0]

importance_df = pd.DataFrame({
    "Signal": feature_names,
    "Learned Weight": [round(c, 3) for c in coefficients],
}).sort_values("Learned Weight", ascending=False)

st.dataframe(importance_df, use_container_width=True, hide_index=True)
st.caption("Learned weights from the fusion model — higher = stronger predictor of actual problems.")

st.divider()

st.subheader("Document Integrity — Validated on Real UN Comtrade Data")

if api_key:
    real_values = fetch_real_uae_pharma_imports(api_key)
    result = benford_check_real(real_values) if real_values else None
    if result:
        st.write(
            f"Real UAE pharmaceutical import declarations (HS Chapter 30, 2023): "
            f"**{result['n_records']} records**, Benford deviation = **{result['deviation']}**."
        )
        st.caption(
            "This is the same Benford's Law function used on synthetic data, "
            "now validated against real declared trade values from UN Comtrade."
        )
    else:
        st.info("Enter your UN Comtrade API key in the sidebar to load real data.")
else:
    st.info("Enter your UN Comtrade API key in the sidebar to validate on real data.")

# ============================================================================
# BREAKDOWN
# ============================================================================

st.subheader("Risk Factor Breakdown")

selected = st.selectbox("Select a shipment to inspect", df["Shipment"].tolist())
row = df[df["Shipment"] == selected].iloc[0]

col_a, col_b = st.columns([1, 2])

with col_a:
    st.metric("Fused Risk Score", row["Risk Score"])
    if row["Flagged"] == "YES":
        st.error(f"FLAGGED: {row['Reasons']}")
    else:
        st.success("Cleared for standard processing")

with col_b:
    factor_df = pd.DataFrame({
        "Check": ["Document", "Supplier", "Cold-Chain", "Route", "Custody"],
        "Risk": [row["Document"], row["Supplier"], row["Cold-Chain"], row["Route"], row["Custody"]],
    })
    st.bar_chart(factor_df.set_index("Check"))

st.divider()

st.caption(
    "Prototype for Presight Innovation Challenge | "
    "Benford's Law + MKT are real formulas | "
    "XGBoost + Logistic Regression trained on synthetic data with non-linear interactions"
)
