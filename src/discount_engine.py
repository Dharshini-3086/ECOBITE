"""
Dynamic Discount Recommendation Engine for ECOBITE
Calculates revenue-aware product discounts using a hybrid ML prediction + business rules approach.
"""

import numpy as np

def calculate_dynamic_discount(
    waste_probability: float,
    current_stock: float,
    predicted_demand: float,
    days_to_expiry: float,
    daily_sales: float,
    selling_price: float,
    previous_discount: float = 0.0,
    sales_during_discount: float = 0.0
) -> dict:
    """
    Computes optimal recommended discount percentage and revenue impact metrics.
    """
    excess_stock = max(0.0, current_stock - predicted_demand)
    excess_ratio = excess_stock / max(current_stock, 1.0)
    
    # Expiry urgency factor (1.0 = very urgent/near expiry, 0.1 = long shelf life)
    urgency_factor = min(1.0, 15.0 / max(days_to_expiry, 1.0))

    # Base discount based on Waste Risk Tier
    if waste_probability < 0.30:
        risk_level = "LOW"
        base_discount = 0.0
    elif waste_probability < 0.60:
        risk_level = "MEDIUM"
        base_discount = 7.5
    elif waste_probability < 0.85:
        risk_level = "HIGH"
        base_discount = 15.0
    else:
        risk_level = "CRITICAL"
        base_discount = 25.0

    # Adjust discount based on excess ratio & expiry urgency
    excess_adjustment = excess_ratio * 15.0 # up to +15% for heavy excess
    urgency_adjustment = urgency_factor * 10.0 # up to +10% for near-expiry

    # Historical elasticity adjustment
    elasticity_mult = 1.0
    if previous_discount > 0 and sales_during_discount > daily_sales:
        # Sales response per 1% discount
        uplift_percent = (sales_during_discount - daily_sales) / max(daily_sales, 0.1)
        if uplift_percent > 0.3:
            # Product is highly price-sensitive, discount works well!
            elasticity_mult = 1.1
        elif uplift_percent < 0.05:
            # Low price sensitivity
            elasticity_mult = 0.95

    raw_recommended_discount = (base_discount + excess_adjustment + urgency_adjustment) * elasticity_mult

    # Cap recommended discount bounds based on risk tier
    if risk_level == "LOW":
        final_discount = min(5.0, raw_recommended_discount)
    elif risk_level == "MEDIUM":
        final_discount = max(5.0, min(12.0, raw_recommended_discount))
    elif risk_level == "HIGH":
        final_discount = max(10.0, min(22.0, raw_recommended_discount))
    else: # CRITICAL
        final_discount = max(20.0, min(40.0, raw_recommended_discount))

    # Round to clean integer or half-percent
    final_discount = round(final_discount, 0)
    discounted_price = round(selling_price * (1.0 - final_discount / 100.0), 2)

    # Revenue Impact Analysis
    # Projected sales boost from discount
    boost_factor = 1.0 + (final_discount / 100.0) * 1.4
    expected_sales_with_discount = min(current_stock, predicted_demand * boost_factor)
    
    expected_original_revenue = min(current_stock, predicted_demand) * selling_price
    expected_discounted_revenue = expected_sales_with_discount * discounted_price
    
    potential_waste_units = max(0.0, current_stock - expected_sales_with_discount)
    potential_waste_loss = potential_waste_units * selling_price
    waste_saved_value = excess_stock * selling_price - potential_waste_loss

    return {
        "risk_level": risk_level,
        "waste_probability": round(waste_probability, 4),
        "recommended_discount_pct": float(final_discount),
        "original_price": float(selling_price),
        "discounted_price": float(discounted_price),
        "excess_stock": float(excess_stock),
        "excess_ratio": round(excess_ratio * 100.0, 1),
        "expected_sales_with_discount": float(round(expected_sales_with_discount, 1)),
        "expected_discounted_revenue": float(round(expected_discounted_revenue, 2)),
        "potential_waste_loss": float(round(potential_waste_loss, 2)),
        "waste_saved_value": float(round(max(0.0, waste_saved_value), 2))
    }
