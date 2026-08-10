"""
Recommendation Engine Module for ECOBITE
Generates actionable inventory guidance, reorder decisions, and human-readable natural language explanations.
"""

def generate_inventory_recommendation(
    product_name: str,
    waste_probability: float,
    risk_level: str,
    current_stock: float,
    predicted_demand: float,
    excess_stock: float,
    days_to_expiry: float,
    recommended_discount_pct: float,
    season: str
) -> dict:
    """
    Produces complete decision-support package containing:
    - Primary Inventory Action
    - Reorder Status
    - Natural Language Explanation
    """
    # 1. Primary Inventory Action
    if risk_level == "LOW":
        action = "No Action (Continue Normal Sales)"
        action_detail = "Inventory levels align well with demand forecast before expiry."
    elif risk_level == "MEDIUM":
        action = "Monitor Sales Velocity & Minor Promo"
        action_detail = f"Consider a small {recommended_discount_pct:.0f}% promotional discount if velocity drops."
    elif risk_level == "HIGH":
        action = "Apply Recommended Discount & Promote"
        action_detail = f"Immediately apply a {recommended_discount_pct:.0f}% discount and features in end-cap promotions."
    else: # CRITICAL
        action = "Urgent Clearance Discount & Halt Replenishment"
        action_detail = f"Apply urgent {recommended_discount_pct:.0f}% discount, bundle with fast-movers, and freeze supplier purchase orders."

    # 2. Purchase / Reorder Decision
    lead_time_buffer = 7 # default buffer
    coverage = current_stock - predicted_demand
    
    if excess_stock > 0 and days_to_expiry <= 30:
        reorder_status = "Stop Replenishment Temporarily"
        reorder_reason = "Excess stock exists close to product expiry date."
    elif current_stock > predicted_demand:
        reorder_status = "Do Not Reorder"
        reorder_reason = "Current stock is sufficient to meet forecasted demand."
    elif current_stock < (predicted_demand * 0.4) and days_to_expiry > 20:
        reorder_status = "Reorder Immediately"
        reorder_reason = "Stock levels are low relative to forecasted demand with sufficient shelf life."
    else:
        reorder_status = "Reorder Small Quantity"
        reorder_reason = "Stock coverage is approaching threshold; order conservative batch."

    # 3. Manager-Friendly Natural Language Explanation
    reasons = []
    if excess_stock > 0:
        reasons.append(f"current stock ({int(current_stock)} units) exceeds predicted demand ({int(predicted_demand)} units) before expiry by {int(excess_stock)} units")
    else:
        reasons.append(f"current stock ({int(current_stock)} units) matches expected demand ({int(predicted_demand)} units)")

    if days_to_expiry <= 15:
        reasons.append(f"only {int(days_to_expiry)} days remain before product expiry")
    elif days_to_expiry <= 30:
        reasons.append(f"product shelf life is moderate ({int(days_to_expiry)} days remaining)")

    reasons_str = " and ".join(reasons)

    if recommended_discount_pct > 0:
        explanation = (
            f"High risk of inventory waste detected for **{product_name}** because {reasons_str}. "
            f"A **{recommended_discount_pct:.0f}% discount** is recommended to increase daily sales velocity, "
            f"clear {int(excess_stock)} excess units, and avoid inventory write-off losses."
        )
    else:
        explanation = (
            f"**{product_name}** has a low waste risk ({waste_probability*100:.1f}%) because {reasons_str}. "
            f"Standard pricing should be maintained."
        )

    return {
        "action": action,
        "action_detail": action_detail,
        "reorder_status": reorder_status,
        "reorder_reason": reorder_reason,
        "explanation": explanation
    }
