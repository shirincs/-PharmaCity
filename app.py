import math
import random
import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Aman - Pharma Supply Chain Integrity", layout="wide")

# ============================================================================
# PORT COORDINATES
# ============================================================================

PORTS = {
    "Mumbai, India": [19.0760, 72.8777],
    "Shanghai, China": [31.2304, 121.4737],
    "Singapore": [1.3521, 103.8198],
    "Hong Kong": [22.3193, 114.1694],
    "Istanbul, Turkey": [41.0082, 28.9784],
    "Cairo, Egypt": [30.0444, 31.2357],
    "Jebel Ali, UAE": [25.0118, 55.0617],
    "Abu Dhabi, UAE": [24.4539, 54.3773],
    "Dubai Airport, UAE": [25.2532, 55.3657],
    "Sharjah, UAE": [25.3463, 55.4209],
}

# ============================================================================
# REAL FORMULAS (unchanged)
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


# ============================================================================
# SYNTHETIC TRAINING DATA + MODEL TRAINING (at startup)
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


@st.cache_resource
def train_models():
    """
    Trains XGBoost (supplier risk) and Logistic Regression (fusion)
    on synthetic historical data at startup.
    Cached so it only runs once per session.
    """
    rng = np.random.default_rng(42)

    # --- Supplier risk training data ---
    # Features: [violations, years_active, shipment_volume]
    n_suppliers = 500
    violations = rng.integers(0, 10, n_suppliers)
    years_active = rng.integers(1, 30, n_suppliers)
    volume = rng.integers(10, 1000, n_suppliers)

    X_supplier = np.column_stack([violations, years_active, volume])

    # Label: supplier flagged if violations high OR (new AND high volume)
    # Add noise so it's not trivially separable
    noise = rng.normal(0, 0.5, n_suppliers)
    y_supplier = ((violations * 0.4) + (10 - years_active) * 0.05 + noise > 1.2).astype(int)

    xgb_model = XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    )
    xgb_model.fit(X_supplier, y_supplier)

    # --- Fusion training data ---
    # Features: [doc_risk, supplier_risk, cold_risk, route_risk, custody_risk]
    n_shipments = 800
    doc = rng.uniform(0, 1, n_shipments)
    supp = rng.uniform(0, 1, n_shipments)
    cold = rng.uniform(0, 1, n_shipments)
    route = rng.uniform(0, 1, n_shipments)
    cust = rng.uniform(0, 1, n_shipments)

    X_fusion = np.column_stack([doc, supp, cold, route, cust])

    # Label: shipment was actually a problem if weighted combination + noise > threshold
    combined = (
        doc * 0.20 + supp * 0.20 + cold * 0.25 + route * 0.15 + cust * 0.20
    )
    noise = rng.normal(0, 0.08, n_shipments)
    y_fusion = ((combined + noise) > 0.45).astype(int)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_fusion, y_fusion)

    return xgb_model, log_model


xgb_model, log_model = train_models()


# ============================================================================
# PIPELINE
# ============================================================================

def supplier_risk_score(violations, years_active=10, volume=100):
    """Uses trained XGBoost model to predict probability of supplier being risky."""
    features = np.array([[violations, years_active, volume]])
    prob = xgb_model.predict_proba(features)[0][1]
    return float(prob)


def fuse_risk_scores(doc, supp, cold, route, cust):
    """Uses trained Logistic Regression to combine signals into one score."""
    features = np.array([[doc, supp, cold, route, cust]])
    prob = log_model.predict_proba(features)[0][1]
    return float(prob)


def get_flag_reasons(doc, supp, cold, route, cust, threshold=0.5):
    reasons = []
    if doc > threshold: reasons.append("Document anomaly")
    if supp > threshold: reasons.append("Supplier history")
    if cold > threshold: reasons.append("Cold-chain breach")
    if route > threshold: reasons.append("Route deviation")
    if cust > threshold: reasons.append("Custody breach")
    return reasons if reasons else ["None"]


# ============================================================================
# SESSION STATE
# ============================================================================

if "shipments" not in st.session_state:
    st.session_state.shipments = [
        {"id": "SH-1001", "note": "Clean shipment", "origin": "Mumbai, India",
         "destination": "Jebel Ali, UAE", "violations": 0,
         "values": generate_natural_values(), "temp_log": [4,5,4,5,6,5,4],
         "planned": 48, "actual": 47, "openings": 0},
        {"id": "SH-1002", "note": "Fabricated paperwork + risky supplier", "origin": "Shanghai, China",
         "destination": "Jebel Ali, UAE", "violations": 3,
         "values": generate_suspicious_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 49, "openings": 0},
        {"id": "SH-1003", "note": "Cold-chain breach mid-transit", "origin": "Singapore",
         "destination": "Abu Dhabi, UAE", "violations": 0,
         "values": generate_natural_values(), "temp_log": [4,5,6,15,18,16,5],
         "planned": 48, "actual": 48, "openings": 0},
        {"id": "SH-1004", "note": "Major unexplained route delay", "origin": "Istanbul, Turkey",
         "destination": "Jebel Ali, UAE", "violations": 0,
         "values": generate_natural_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 84, "openings": 0},
        {"id": "SH-1005", "note": "Container opened twice", "origin": "Hong Kong",
         "destination": "Dubai Airport, UAE", "violations": 0,
         "values": generate_natural_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 47, "openings": 2},
        {"id": "SH-1006", "note": "Worst case - multiple red flags", "origin": "Cairo, Egypt",
         "destination": "Jebel Ali, UAE", "violations": 4,
         "values": generate_suspicious_values(), "temp_log": [4,20,22,19,18,17,5],
         "planned": 48, "actual": 90, "openings": 1},
        {"id": "SH-1007", "note": "Mild - low risk overall", "origin": "Mumbai, India",
         "destination": "Sharjah, UAE", "violations": 1,
         "values": generate_natural_values(), "temp_log": [5,6,5,6,5,6,5],
         "planned": 48, "actual": 50, "openings": 0},
        {"id": "SH-1008", "note": "Clean shipment", "origin": "Singapore",
         "destination": "Jebel Ali, UAE", "violations": 0,
         "values": generate_natural_values(), "temp_log": [4,4,5,5,4,4,5],
         "planned": 48, "actual": 46, "openings": 0},
    ]

# ============================================================================
# HEADER + SIDEBAR
# ============================================================================

st.title("Aman: Pharmaceutical Supply Chain Integrity")
st.markdown("*AI-powered risk scoring for customs inspection prioritization*")
st.divider()

st.sidebar.header("Controls")
st.sidebar.subheader("Flag Threshold")
flag_threshold = st.sidebar.slider("Flag if score above", 0.0, 1.0, 0.5, 0.05)

st.sidebar.divider()
st.sidebar.subheader("Add New Shipment")

with st.sidebar.form("add_shipment"):
    new_id = st.text_input("Shipment ID", "SH-1009")
    new_note = st.text_input("Scenario note", "Custom entry")
    new_origin = st.selectbox("Origin", list(PORTS.keys()), index=0)
    new_dest = st.selectbox("Destination", list(PORTS.keys()), index=6)
    new_violations = st.number_input("Supplier violations", 0, 10, 0)
    new_openings = st.number_input("Unauthorized openings", 0, 10, 0)
    new_planned = st.number_input("Planned hours", 1, 200, 48)
    new_actual = st.number_input("Actual hours", 1, 200, 48)
    value_type = st.selectbox("Declared values", ["Natural (Benford-compliant)", "Suspicious (fabricated)"])
    temp_type = st.selectbox("Temperature log", ["Normal (2-8°C)", "Breach (spike to 20°C)"])

    submitted = st.form_submit_button("Add Shipment")

    if submitted:
        vals = generate_natural_values() if "Natural" in value_type else generate_suspicious_values()
        temps = [4,5,4,5,4,5,4] if "Normal" in temp_type else [4,20,22,19,18,17,5]
        st.session_state.shipments.append({
            "id": new_id, "note": new_note,
            "origin": new_origin, "destination": new_dest,
            "violations": new_violations, "values": vals, "temp_log": temps,
            "planned": new_planned, "actual": new_actual, "openings": new_openings,
        })
        st.sidebar.success(f"Added {new_id}")

if st.sidebar.button("Reset to default shipments"):
    st.session_state.shipments = st.session_state.shipments[:8]
    st.sidebar.success("Reset")

# ============================================================================
# RUN PIPELINE
# ============================================================================

def run_pipeline(s):
    doc = document_integrity_check(s["values"])
    supp = supplier_risk_score(s["violations"])
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
            "color": color, "shipment": row["Shipment"], "risk": row["Risk Score"],
        })

arc_df = pd.DataFrame(arc_data)

if not arc_df.empty:
    arc_layer = pdk.Layer(
        "ArcLayer", data=arc_df,
        get_source_position="origin", get_target_position="destination",
        get_source_color="color", get_target_color="color",
        get_width=4, pickable=True,
    )
    port_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([{"name": n, "coords": c} for n, c in PORTS.items()]),
        get_position="coords", get_radius=30000,
        get_fill_color=[30, 100, 200, 180], pickable=True,
    )
    view_state = pdk.ViewState(latitude=20, longitude=70, zoom=2.5, pitch=40)
    st.pydeck_chart(pdk.Deck(
        layers=[arc_layer, port_layer],
        initial_view_state=view_state,
        tooltip={"text": "{shipment}\nRisk: {risk}"},
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

styled = df.style.applymap(color_risk, subset=["Risk Score"])
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

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
    "XGBoost + Logistic Regression trained on synthetic data at startup"
)
