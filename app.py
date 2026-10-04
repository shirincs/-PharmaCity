"""
PharmaCity: Pharmaceutical Supply Chain Integrity
Prototype for Presight Innovation Challenge

Benford's Law and MKT are real formulas. The registry is SIMULATED.
Clean shipments' declared values are resampled from real World Bank WITS data.
Models trained on synthetic data. Random seed fixed for reproducibility.

"""

import math
import random
import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

random.seed(10)
np.random.seed(10)

st.set_page_config(page_title="PharmaCity - Pharma Supply Chain Integrity", layout="wide")

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
# FORMULAS
# ============================================================================

BENFORD_NORMAL = 0.29
BENFORD_MAX = 1.5


def document_integrity_check(declared_values):
    """Benford's Law half of the document check: are the declared values plausible?"""
    if len(declared_values) < 10:
        return 0.0
    expected = {d: math.log10(1 + 1 / d) for d in range(1, 10)} # Formula
    leading = [int(str(int(abs(v)))[0]) for v in declared_values] 
    n = len(leading)
    observed = {d: leading.count(d) / n for d in range(1, 10)} 
    deviation = sum(abs(observed[d] - expected[d]) for d in range(1, 10)) 
    return min(max((deviation - BENFORD_NORMAL) / (BENFORD_MAX - BENFORD_NORMAL), 0.0), 1.0)


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
# REGISTRY CHECK: the retrieval step of the proposal's RAG design
# ============================================================================

REGISTRY = [
    {"license_id": "LIC-1001", "manufacturer": "Aldara Pharmaceuticals", "status": "Active"},
    {"license_id": "LIC-1002", "manufacturer": "Brightwell Biotech", "status": "Active"},
    {"license_id": "LIC-1003", "manufacturer": "Cedarline Medical", "status": "Active"},
    {"license_id": "LIC-1004", "manufacturer": "Dunmore Healthcare", "status": "Expired"},
    {"license_id": "LIC-1005", "manufacturer": "Evergreen Generics", "status": "Active"},
    {"license_id": "LIC-1006", "manufacturer": "Fairhaven Labs", "status": "Active"},
    {"license_id": "LIC-1007", "manufacturer": "Glenfield Pharma", "status": "Active"},
    {"license_id": "LIC-1008", "manufacturer": "Harborview Therapeutics", "status": "Suspended"},
]


@st.cache_resource
def build_registry_index():
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform([r["manufacturer"] for r in REGISTRY])
    return vectorizer, matrix


def registry_integrity_check(manufacturer, license_id, min_similarity=0.6):
    """Registry half of the document check. Returns (risk 0-1, explanation).

    Severity order (a judgment call): unknown manufacturer (1.0) > wrong or unknown
    license (0.8) > expired or suspended license (0.7) > small name mismatch.
    """
    vectorizer, matrix = build_registry_index()
    sims = cosine_similarity(vectorizer.transform([manufacturer]), matrix)[0]
    best = int(sims.argmax())
    entry, sim = REGISTRY[best], float(sims[best])

    if sim < min_similarity:
        return 1.0, f"no registry entry resembles '{manufacturer}'"

    claimed = license_id.strip().upper()
    if claimed != entry["license_id"]:
        owner = next((r["manufacturer"] for r in REGISTRY if r["license_id"] == claimed), None)
        if owner:
            return 0.8, f"license {claimed} belongs to {owner}, not {entry['manufacturer']}"
        return 0.8, f"license {claimed} not found in registry"

    if entry["status"] != "Active":
        return 0.7, f"license {claimed} is {entry['status'].lower()}"

    return max(0.0, round(1 - sim, 2)), f"verified ({entry['manufacturer']}, {claimed}, active)"


# ============================================================================
# REAL DATA (World Bank WITS Excel file)
# ============================================================================

@st.cache_data
def load_real_uae_pharma_imports():
    """Loads real UAE pharmaceutical import values from the WITS Excel file."""
    try:
        df = pd.read_excel("WITS-By-HS6Product.xlsx", sheet_name="By-HS6Product")
        value_col = None
        for col in df.columns:
            if "Trade Value" in col:
                value_col = col
                break
        if value_col is None:
            return []
        values = df[value_col].dropna().tolist()
        return [float(v) for v in values if v > 0]
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
    """Declared values of a legitimate shipment: resampled from real WITS trade values
    (synthetic log-uniform fallback if the file is missing)."""
    real = load_real_uae_pharma_imports()
    if len(real) >= 10:
        return random.choices(real, k=n)
    values = []
    for _ in range(n):
        magnitude = random.choice([10, 100, 1000, 10000])
        coeff = 10 ** random.uniform(0, 1)
        values.append(round(coeff * magnitude, 2))
    return values


def generate_suspicious_values(n=100):
    return [round(random.uniform(8000, 9999), 2) for _ in range(n)]


def generate_mildly_off_values(n=100):
    values = []
    for _ in range(n):
        magnitude = random.choice([100, 1000])
        coeff = random.uniform(3, 7)
        values.append(round(coeff * magnitude, 2))
    return values


# ============================================================================
# MODEL TRAINING (at startup, cached)
# ============================================================================

@st.cache_resource
def train_models():
    rng = np.random.default_rng(42)

    # --- Supplier risk model (XGBoost) ---
    n_suppliers = 800
    violations = rng.integers(0, 10, n_suppliers)
    years_active = rng.integers(1, 30, n_suppliers)
    volume = rng.integers(10, 1000, n_suppliers)

    X_supplier = np.column_stack([violations, years_active, volume])

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
    n_shipments = 1500

    latent_risk = (
        0.30 * rng.beta(2, 5, n_shipments)
        + 0.25 * rng.binomial(1, 0.15, n_shipments)
        + 0.20 * rng.exponential(0.5, n_shipments).clip(0, 2)
    )

    doc_obs = np.clip(latent_risk + rng.normal(0, 0.25, n_shipments), 0, 1)
    supp_obs = np.clip(latent_risk * 0.8 + rng.normal(0, 0.3, n_shipments), 0, 1)
    cold_obs = np.clip(latent_risk * 1.2 + rng.normal(0, 0.35, n_shipments), 0, 1)
    route_obs = np.clip(latent_risk * 0.5 + rng.normal(0, 0.4, n_shipments), 0, 1)
    cust_obs = np.clip(latent_risk * 0.7 + rng.normal(0, 0.3, n_shipments), 0, 1)

    X_fusion = np.column_stack([doc_obs, supp_obs, cold_obs, route_obs, cust_obs])

    y_fusion = (latent_risk + rng.normal(0, 0.25, n_shipments) > 0.55).astype(int)

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        X_fusion, y_fusion, test_size=0.2, random_state=42
    )

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(Xf_train, yf_train)

    fusion_acc = accuracy_score(yf_test, log_model.predict(Xf_test))
    fusion_auc = roc_auc_score(yf_test, log_model.predict_proba(Xf_test)[:, 1])

    y_proba = log_model.predict_proba(Xf_test)[:, 1]
    order = np.argsort(-y_proba)
    k10 = max(1, int(0.1 * len(order)))
    k20 = max(1, int(0.2 * len(order)))
    precision_at_10 = yf_test[order[:k10]].mean()
    precision_at_20 = yf_test[order[:k20]].mean()
    base_rate = yf_test.mean()

    return {
        "xgb": xgb_model,
        "log": log_model,
        "supplier_acc": supplier_acc,
        "supplier_auc": supplier_auc,
        "fusion_acc": fusion_acc,
        "fusion_auc": fusion_auc,
        "precision_at_10": precision_at_10,
        "precision_at_20": precision_at_20,
        "base_rate": base_rate,
    }


models = train_models()
xgb_model = models["xgb"]
log_model = models["log"]

def build_model_report():
    """Builds a downloadable text report of all model metrics and diagnostics."""
    lines = []
    lines.append("=" * 60)
    lines.append("PharmaCity — Model & Validation Report")
    lines.append("=" * 60)
    lines.append("")

    lines.append("FUSION MODEL PERFORMANCE")
    lines.append("-" * 60)
    lines.append(f"Precision @ top 10% inspected:  {models['precision_at_10']:.3f}")
    lines.append(f"Precision @ top 20% inspected:  {models['precision_at_20']:.3f}")
    lines.append(f"Fusion Model AUC:               {models['fusion_auc']:.3f}")
    lines.append(f"Base rate (random inspection):  {models['base_rate']:.3f}")
    lines.append(f"Lift vs random @ 10%:           {models['precision_at_10'] / models['base_rate']:.2f}x")
    lines.append(f"Lift vs random @ 20%:           {models['precision_at_20'] / models['base_rate']:.2f}x")
    lines.append("")

    lines.append("SUPPLIER RISK MODEL (XGBoost)")
    lines.append("-" * 60)
    lines.append(f"Accuracy: {models['supplier_acc']:.3f}")
    lines.append(f"AUC:      {models['supplier_auc']:.3f}")
    lines.append("")

    lines.append("LEARNED FUSION WEIGHTS")
    lines.append("-" * 60)
    feature_names = ["Document", "Supplier", "Cold-Chain", "Route", "Custody"]
    coefficients = log_model.coef_[0]
    for name, coef in sorted(zip(feature_names, coefficients), key=lambda x: -abs(x[1])):
        lines.append(f"  {name:<12} {coef:+.3f}")
    lines.append("")

    lines.append("WITS BENFORD VALIDATION")
    lines.append("-" * 60)
    real_values = load_real_uae_pharma_imports()
    result = benford_check_real(real_values) if real_values else None
    if result:
        lines.append(f"Source:            World Bank WITS (HS 3004, UAE imports, 2021)")
        lines.append(f"Records:           {result['n_records']}")
        lines.append(f"Benford deviation: {result['deviation']}")
        lines.append(f"Baseline used:     {BENFORD_NORMAL}")
    else:
        lines.append("WITS file not loaded.")
    lines.append("")

    lines.append("NOTES")
    lines.append("-" * 60)
    lines.append("All training data is synthetic. Metrics are a mechanism check,")
    lines.append("not a production performance claim. Real validation requires")
    lines.append("historical inspection outcomes from Customs.")

    return "\n".join(lines)
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
# SHIPMENT DATABASE
# ============================================================================

if "shipments" not in st.session_state:
    st.session_state.shipments = [
        {"id": "SH-1001", "note": "Clean shipment", "origin": "Mumbai, India",
         "destination": "Jebel Ali, UAE", "violations": 0, "years_active": 18, "volume": 400,
         "manufacturer": "Aldara Pharmaceuticals", "license": "LIC-1001",
         "values": generate_natural_values(), "temp_log": [4,5,4,5,6,5,4],
         "planned": 48, "actual": 47, "openings": 0},

        {"id": "SH-1002", "note": "Clean shipment", "origin": "Singapore",
         "destination": "Jebel Ali, UAE", "violations": 0, "years_active": 22, "volume": 600,
         "manufacturer": "Brightwell Biotech", "license": "LIC-1002",
         "values": generate_natural_values(), "temp_log": [4,4,5,5,4,4,5],
         "planned": 48, "actual": 46, "openings": 0},

        {"id": "SH-1003", "note": "Minor route delay", "origin": "Istanbul, Turkey",
         "destination": "Jebel Ali, UAE", "violations": 0, "years_active": 15, "volume": 350,
         "manufacturer": "Cedarline Medical", "license": "LIC-1003",
         "values": generate_natural_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 56, "openings": 0},

        {"id": "SH-1004", "note": "Expired manufacturer license (numbers look natural)", "origin": "Mumbai, India",
         "destination": "Sharjah, UAE", "violations": 1, "years_active": 8, "volume": 500,
         "manufacturer": "Dunmore Healthcare", "license": "LIC-1004",
         "values": generate_natural_values(), "temp_log": [5,6,5,6,5,6,5],
         "planned": 48, "actual": 49, "openings": 0},

        {"id": "SH-1005", "note": "Mild cold-chain excursion", "origin": "Singapore",
         "destination": "Abu Dhabi, UAE", "violations": 0, "years_active": 12, "volume": 450,
         "manufacturer": "Brightwell Biotech", "license": "LIC-1002",
         "values": generate_natural_values(), "temp_log": [4,5,6,9,10,7,5],
         "planned": 48, "actual": 50, "openings": 0},

        {"id": "SH-1006", "note": "Slight document anomaly", "origin": "Shanghai, China",
         "destination": "Jebel Ali, UAE", "violations": 1, "years_active": 10, "volume": 700,
         "manufacturer": "Evergreen Generics", "license": "LIC-1005",
         "values": generate_mildly_off_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 48, "openings": 0},

        {"id": "SH-1007", "note": "Container opened once", "origin": "Hong Kong",
         "destination": "Dubai Airport, UAE", "violations": 0, "years_active": 14, "volume": 300,
         "manufacturer": "Fairhaven Labs", "license": "LIC-1006",
         "values": generate_natural_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 47, "openings": 1},

        {"id": "SH-1008", "note": "Cold-chain breach", "origin": "Cairo, Egypt",
         "destination": "Jebel Ali, UAE", "violations": 1, "years_active": 9, "volume": 550,
         "manufacturer": "Glenfield Pharma", "license": "LIC-1007",
         "values": generate_natural_values(), "temp_log": [4,5,6,15,18,16,5],
         "planned": 48, "actual": 48, "openings": 0},

        {"id": "SH-1009", "note": "Fabricated paperwork", "origin": "Shanghai, China",
         "destination": "Jebel Ali, UAE", "violations": 2, "years_active": 6, "volume": 800,
         "manufacturer": "Zenith Biopharm International", "license": "LIC-9931",
         "values": generate_suspicious_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 49, "openings": 0},

        {"id": "SH-1010", "note": "Major route deviation", "origin": "Istanbul, Turkey",
         "destination": "Jebel Ali, UAE", "violations": 1, "years_active": 11, "volume": 500,
         "manufacturer": "Cedarline Medical", "license": "LIC-1003",
         "values": generate_natural_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 84, "openings": 0},

        {"id": "SH-1011", "note": "Two mild issues", "origin": "Mumbai, India",
         "destination": "Jebel Ali, UAE", "violations": 1, "years_active": 9, "volume": 500,
         "manufacturer": "Aldara Pharmaceuticals", "license": "LIC-1001",
         "values": generate_mildly_off_values(), "temp_log": [5,6,5,6,5,6,5],
         "planned": 48, "actual": 52, "openings": 0},

        {"id": "SH-1012", "note": "Moderate cold + mild route", "origin": "Singapore",
         "destination": "Abu Dhabi, UAE", "violations": 1, "years_active": 10, "volume": 600,
         "manufacturer": "Brightwell Biotech", "license": "LIC-1002",
         "values": generate_natural_values(), "temp_log": [4,6,8,11,12,9,5],
         "planned": 48, "actual": 58, "openings": 0},

        {"id": "SH-1013", "note": "Mild custody + document issues (wrong license)", "origin": "Hong Kong",
         "destination": "Dubai Airport, UAE", "violations": 2, "years_active": 7, "volume": 700,
         "manufacturer": "Fairhaven Labs", "license": "LIC-1007",
         "values": generate_mildly_off_values(), "temp_log": [4,5,4,5,4,5,4],
         "planned": 48, "actual": 50, "openings": 1},

        {"id": "SH-1014", "note": "Multiple red flags", "origin": "Cairo, Egypt",
         "destination": "Jebel Ali, UAE", "violations": 4, "years_active": 3, "volume": 900,
         "manufacturer": "Nilecrest Pharma", "license": "LIC-0000",
         "values": generate_suspicious_values(), "temp_log": [4,20,22,19,18,17,5],
         "planned": 48, "actual": 90, "openings": 2},
    ]

# ============================================================================
# HEADER + SIDEBAR
# ============================================================================

st.title("PharmaCity: Pharmaceutical Supply Chain Integrity")
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))
st.markdown("*Intelligent risk scoring for customs inspection prioritization*")
st.divider()

st.sidebar.header("Controls")

st.sidebar.subheader("Admin")
st.sidebar.download_button(
    "Download model report",
    data=build_model_report(),
    file_name="pharmacity_model_report.txt",
    mime="text/plain",
)
st.sidebar.divider()
st.sidebar.subheader("Inspection Capacity")
capacity_pct = st.sidebar.slider("Inspect the top % of shipments", 5, 100, 30, 5)

st.sidebar.divider()
st.sidebar.subheader("Add New Shipment")

with st.sidebar.form("add_shipment"):
    new_id = st.text_input("Shipment ID", "SH-1015")
    new_note = st.text_input("Scenario note", "Custom entry")
    new_origin = st.selectbox("Origin", list(PORTS.keys()), index=0)
    new_dest = st.selectbox("Destination", list(PORTS.keys()), index=6)
    new_violations = st.number_input("Supplier violations", 0, 10, 0)
    new_years = st.number_input("Supplier years active", 1, 50, 15)
    new_volume = st.number_input("Shipment volume", 10, 2000, 400)
    new_mfr = st.text_input("Manufacturer (as written on manifest)", REGISTRY[0]["manufacturer"])
    new_license = st.text_input("License number (as written on manifest)", REGISTRY[0]["license_id"])
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
            "manufacturer": new_mfr, "license": new_license,
            "values": vals, "temp_log": temps,
            "planned": new_planned, "actual": new_actual, "openings": new_openings,
        })
        st.session_state["flash"] = f"Added {new_id}"
        st.rerun()

# ============================================================================
# RUN PIPELINE
# ============================================================================

doc_notes = {}


def run_pipeline(s):
    doc_benford = document_integrity_check(s["values"])
    doc_registry, registry_note = registry_integrity_check(s["manufacturer"], s["license"])
    doc = max(doc_benford, doc_registry)
    doc_notes[s["id"]] = f"Document check: declared values (Benford) {doc_benford:.2f}; registry {doc_registry:.2f} ({registry_note})."
    supp = supplier_risk_score(s["violations"], s["years_active"], s["volume"])
    cold = cold_chain_integrity_check(s["temp_log"])
    route = route_integrity_check(s["planned"], s["actual"])
    cust = custody_integrity_check(s["openings"])
    fused = fuse_risk_scores(doc, supp, cold, route, cust)
    reasons = get_flag_reasons(doc, supp, cold, route, cust)
    if doc > 0.5:
        parts = (["declared values"] if doc_benford > 0.5 else []) + (["registry"] if doc_registry > 0.5 else [])
        reasons = [r.replace("Document anomaly", f"Document anomaly ({' + '.join(parts)})") for r in reasons]
    return {
        "Shipment": s["id"],
        "Scenario": s["note"],
        "Origin": s["origin"],
        "Destination": s["destination"],
        "Risk Score": round(fused, 3),
        "Reasons": ", ".join(reasons),
        "Document": round(doc, 2),
        "Supplier": round(supp, 2),
        "Cold-Chain": round(cold, 2),
        "Route": round(route, 2),
        "Custody": round(cust, 2),
    }


results = [run_pipeline(s) for s in st.session_state.shipments]
df = pd.DataFrame(results).sort_values("Risk Score", ascending=False).reset_index(drop=True)

n_flag = max(1, round(len(df) * capacity_pct / 100))
df.insert(5, "Flagged", ["YES" if i < n_flag else "No" for i in range(len(df))])
cutoff = df["Risk Score"].iloc[n_flag - 1]

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
# MAP — aggregated by route (one arc per origin→destination pair)
# ============================================================================

st.subheader("Global Shipment Routes")

route_groups = {}
for _, row in df.iterrows():
    key = (row["Origin"], row["Destination"])
    if key not in route_groups:
        route_groups[key] = {"count": 0, "max_risk": 0.0, "any_flagged": False}
    route_groups[key]["count"] += 1
    route_groups[key]["max_risk"] = max(route_groups[key]["max_risk"], row["Risk Score"])
    if row["Flagged"] == "YES":
        route_groups[key]["any_flagged"] = True

arc_data = []
for (origin, dest), info in route_groups.items():
    origin_coords = PORTS.get(origin)
    dest_coords = PORTS.get(dest)
    if origin_coords and dest_coords:
        color = [220, 50, 50] if info["any_flagged"] else [50, 180, 90]
        arc_data.append({
            "origin": origin_coords,
            "destination": dest_coords,
            "color": color,
            "width": 1 + info["count"],
        })

arc_df = pd.DataFrame(arc_data)

# Always show the map regardless of hovering
if not arc_df.empty:
    arc_layer = pdk.Layer(
        "ArcLayer", data=arc_df,
        get_source_position="origin", get_target_position="destination",
        get_source_color="color", get_target_color="color",
        get_width="width", get_height=0.3,
    )
    view_state = pdk.ViewState(latitude=20, longitude=70, zoom=2, pitch=0)
    st.pydeck_chart(pdk.Deck(
        layers=[arc_layer],
        initial_view_state=view_state,
    ))

# ============================================================================
# RANKED TABLE
# ============================================================================

st.subheader("Shipment Risk Ranking")

def color_risk(val):
    if val >= cutoff:
        return "background-color: #ffcccc; color: #000000"
    return "background-color: #ccffcc; color: #000000"
    
styled = df.style.map(color_risk, subset=["Risk Score"])
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
    st.caption(doc_notes[selected])

with col_b:
    factor_df = pd.DataFrame({
        "Check": ["Document", "Supplier", "Cold-Chain", "Route", "Custody"],
        "Risk": [row["Document"], row["Supplier"], row["Cold-Chain"], row["Route"], row["Custody"]],
    })
    st.bar_chart(factor_df.set_index("Check"))
 
