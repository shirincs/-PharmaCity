"""
Pharmaceutical Supply Chain Integrity - Demo Script
=====================================================

This is a small proof-of-concept demonstrating the core mechanism of PharmaCity: 
five signal checks (document, supplier, cold-chain, route, custody) get combined into one
fused risk score per shipment, so inspectors know which shipments to prioritize instead of picking randomly.

"""

import math
import random

random.seed(1)


# STEP 1: The five signal checks
# Each function takes raw shipment data and returns a risk score from 0 to 1 (0 = no concern, 1 = maximum concern).
# -------------------------------------------------------------------------------------------------------------------


def document_integrity_check(declared_values):
    """
    Benford's Law - Genuine numerical data values tend to start with a '1' far more often
    than a '9', following a specific known distribution. Fabricated numbers (like faked invoice values) 
    usually don't follow this pattern, because people tend to pick numbers that feel evenly random.
    """
    expected_distribution = {d: math.log10(1 + 1 / d) for d in range(1, 10)} # Benford's Law formula 

    # Get the leading (first) digit of each declared value
    leading_digits = []
    for value in declared_values:
        digits_only = str(int(abs(value))) 
        leading_digits.append(int(digits_only[0]))

    n = len(leading_digits)
    observed_distribution = {
        d: leading_digits.count(d) / n for d in range(1, 10) # Observed distribution of leading digits
    }

    # Total deviation between observed and expected (Benford) distributions
    deviation = sum(
        abs(observed_distribution[d] - expected_distribution[d])
        for d in range(1, 10)
    )

    # Normalize deviation into a 0-1 risk score
    risk_score = min(deviation / 1.0, 1.0) # Replace 1.0 for real max deviation observed in training data
    return risk_score


def supplier_risk_score(past_violation_count): 
    """
    Use gradient-boosted tree model (XGBoost / LightGBM) trained on real supplier violation
    and seizure records. Here, we just scale violation count directly,
    since we don't have a real training dataset for this demo.
    """
    # 5+ past violations is treated as maximum risk for this demo
    return min(past_violation_count / 5, 1.0)


def cold_chain_integrity_check(temperature_log_celsius, safe_threshold_c=8.0):
    """
    Mean Kinetic Temperature (MKT) is the actual pharmaceutical industry-standard (FDA/USP) way of
    turning a full temperature log into ONE number representing the cumulative thermal stress a 
    product experienced (rather than just looking at the peak temperature).
    """
    activation_energy = 83144  # J/mol (standard)
    R = 8.314462  # universal gas constant, J/(mol*K)

    kelvin_temps = [t + 273.15 for t in temperature_log_celsius]

    sum_of_exponentials = sum(
        math.exp(-activation_energy / (R * t)) for t in kelvin_temps # MKT formula 
    )
    average_exponential = sum_of_exponentials / len(kelvin_temps)
    mkt_kelvin = -activation_energy / (R * math.log(average_exponential))
    mkt_celsius = mkt_kelvin - 273.15

    if mkt_celsius <= safe_threshold_c:
        return 0.0

    excess_over_threshold = mkt_celsius - safe_threshold_c
    # 5 degrees over the safe threshold is treated as maximum risk
    risk_score = min(excess_over_threshold / 5.0, 1.0)
    return risk_score


def route_integrity_check(planned_hours, actual_hours):
    """
    Compares actual transit time against the planned schedule. Any large unexplained delay is a signal 
    the shipment may have gone somewhere it wasn't supposed to.
    """
    deviation_fraction = abs(actual_hours - planned_hours) / planned_hours
    # 50%+ deviation from the planned schedule is treated as maximum risk
    risk_score = min(deviation_fraction / 0.5, 1.0)
    return risk_score


def custody_integrity_check(unauthorized_opening_count):
    """
    E-seal check: any unexpected container opening during transit is a strong sign of tampering. 
    Even one unauthorized opening is a serious concern.
    """
    risk_score = min(unauthorized_opening_count * 0.6, 1.0)
    return risk_score


# STEP 2: Fusion - combine all five signals into one risk score
# ---------------------------------------------------------------------------

def fuse_risk_scores(document_risk, supplier_risk, cold_chain_risk,
                      route_risk, custody_risk):
    """
    Use logistic regression trained on real past inspection outcomes (i.e. which flagged shipments actually
    turned out to be problems). Here, a simple weighted average is used instead, due to data constraints.
    """
    weights = {
        "document": 0.20,
        "supplier": 0.20,
        "cold_chain": 0.25,   
        "route": 0.15,
        "custody": 0.20,
    }

    fused_score = (
        document_risk * weights["document"]
        + supplier_risk * weights["supplier"]
        + cold_chain_risk * weights["cold_chain"]
        + route_risk * weights["route"]
        + custody_risk * weights["custody"]
    )
    return fused_score


def get_flag_reasons(document_risk, supplier_risk, cold_chain_risk,
                      route_risk, custody_risk, threshold=0.5): 
    """
    Explains why specific checks triggered a concern.
    """
    reasons = []
    if document_risk > threshold:
        reasons.append("document anomaly")
    if supplier_risk > threshold:
        reasons.append("supplier history")
    if cold_chain_risk > threshold:
        reasons.append("cold-chain breach")
    if route_risk > threshold:
        reasons.append("route deviation")
    if custody_risk > threshold:
        reasons.append("custody breach")
    return reasons if reasons else ["none"]


# STEP 3: Designing test shipments
# ---------------------------------------------------------------------------

def generate_natural_values(n=100):
    """
    Generate values that replicate realistic data (Benford-compliant)
    """
    values = []
    for _ in range(n):
        magnitude = random.choice([10, 100, 1000, 10000])
        leading_coefficient = 10 ** random.uniform(0, 1)  # log-uniform, 1 to 10
        values.append(round(leading_coefficient * magnitude, 2))
    return values


def generate_suspicious_values(n=100):
    return [round(random.uniform(8000, 9999), 2) for _ in range(n)]


shipments = [
    {
        "id": "SH-1001",
        "note": "Clean shipment - no red flags expected",
        "supplier_violations": 0,
        "declared_values": generate_natural_values(),
        "temperature_log": [4, 5, 4, 5, 6, 5, 4],
        "planned_hours": 48, "actual_hours": 47,
        "unauthorized_openings": 0,
    },
    {
        "id": "SH-1002",
        "note": "Fabricated paperwork + risky supplier",
        "supplier_violations": 3,
        "declared_values": generate_suspicious_values(),
        "temperature_log": [4, 5, 4, 5, 4, 5, 4],
        "planned_hours": 48, "actual_hours": 49,
        "unauthorized_openings": 0,
    },
    {
        "id": "SH-1003",
        "note": "Cold-chain breach mid-transit",
        "supplier_violations": 0,
        "declared_values": generate_natural_values(),
        "temperature_log": [4, 5, 6, 15, 18, 16, 5],
        "planned_hours": 48, "actual_hours": 48,
        "unauthorized_openings": 0,
    },
    {
        "id": "SH-1004",
        "note": "Major unexplained route delay",
        "supplier_violations": 0,
        "declared_values": generate_natural_values(),
        "temperature_log": [4, 5, 4, 5, 4, 5, 4],
        "planned_hours": 48, "actual_hours": 84,
        "unauthorized_openings": 0,
    },
    {
        "id": "SH-1005",
        "note": "Container opened twice, unexplained",
        "supplier_violations": 0,
        "declared_values": generate_natural_values(),
        "temperature_log": [4, 5, 4, 5, 4, 5, 4],
        "planned_hours": 48, "actual_hours": 47,
        "unauthorized_openings": 2,
    },
    {
        "id": "SH-1006",
        "note": "Worst case - multiple red flags at once",
        "supplier_violations": 4,
        "declared_values": generate_suspicious_values(),
        "temperature_log": [4, 20, 22, 19, 18, 17, 5],
        "planned_hours": 48, "actual_hours": 90,
        "unauthorized_openings": 1,
    },
    {
        "id": "SH-1007",
        "note": "Mild - low risk overall",
        "supplier_violations": 1,
        "declared_values": generate_natural_values(),
        "temperature_log": [5, 6, 5, 6, 5, 6, 5],
        "planned_hours": 48, "actual_hours": 50,
        "unauthorized_openings": 0,
    },
    {
        "id": "SH-1008",
        "note": "Clean shipment",
        "supplier_violations": 0,
        "declared_values": generate_natural_values(),
        "temperature_log": [4, 4, 5, 5, 4, 4, 5],
        "planned_hours": 48, "actual_hours": 46,
        "unauthorized_openings": 0,
    },
]


# STEP 4: Run the full pipeline on every shipment
# ---------------------------------------------------------------------------

def run_pipeline(shipment):
    """Runs all five checks + fusion on a single shipment, returns results."""
    doc_risk = document_integrity_check(shipment["declared_values"])
    supp_risk = supplier_risk_score(shipment["supplier_violations"])
    cold_risk = cold_chain_integrity_check(shipment["temperature_log"])
    route_risk = route_integrity_check(
        shipment["planned_hours"], shipment["actual_hours"]
    )
    custody_risk = custody_integrity_check(shipment["unauthorized_openings"])

    fused = fuse_risk_scores(doc_risk, supp_risk, cold_risk, route_risk, custody_risk)
    reasons = get_flag_reasons(doc_risk, supp_risk, cold_risk, route_risk, custody_risk)

    return {
        "id": shipment["id"],
        "note": shipment["note"],
        "risk_score": fused,
        "reasons": reasons,
    }


if __name__ == "__main__":
    results = [run_pipeline(s) for s in shipments]

    # Rank shipments from highest to lowest risk 
    results.sort(key=lambda r: r["risk_score"], reverse=True)

    print("=" * 90)
    print("SHIPMENT RISK RANKING  (highest risk first)")
    print("=" * 90)
    print(f"{'Shipment':<10} {'Risk Score':<12} {'Flagged For':<35} {'Scenario'}")
    print("-" * 90)
    for r in results:
        reasons_str = ", ".join(r["reasons"])
        print(f"{r['id']:<10} {r['risk_score']:<12.2f} {reasons_str:<35} {r['note']}")
    print("=" * 90)
