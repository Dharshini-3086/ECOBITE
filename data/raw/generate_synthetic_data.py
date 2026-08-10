"""
Realistic Synthetic Data Generator for ECOBITE
Generates domain-accurate supermarket inventory data with seasonal, expiry, and sales velocity relationships.
"""

import os
import numpy as np
import pandas as pd

def generate_ecobite_dataset(num_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    product_templates = [
        # Product_Name, Category, Base_Price, Base_Daily_Sales, Peak_Season, Shelf_Life_Days
        ("Sunscreen Lotion SPF 50", "Personal Care", 450.0, 4.0, "Summer", 45),
        ("Fresh Whole Milk 1L", "Dairy", 60.0, 25.0, "All-Season", 7),
        ("Whole Wheat Bread", "Bakery", 45.0, 20.0, "All-Season", 5),
        ("Fresh Orange Juice 1L", "Beverage", 120.0, 12.0, "Summer", 10),
        ("Butter Biscuits 200g", "Pantry", 30.0, 15.0, "All-Season", 180),
        ("Basmati Rice 5kg", "Pantry", 650.0, 5.0, "All-Season", 365),
        ("Anti-Dandruff Shampoo", "Personal Care", 280.0, 6.0, "All-Season", 300),
        ("Moisturizing Face Cream", "Personal Care", 350.0, 5.0, "Winter", 180),
        ("Greek Yogurt Strawberries", "Dairy", 85.0, 14.0, "Summer", 12),
        ("Refined Cooking Oil 1L", "Pantry", 160.0, 10.0, "All-Season", 180),
        ("Fresh Watermelon", "Produce", 150.0, 18.0, "Summer", 8),
        ("Fresh Strawberries 250g", "Produce", 180.0, 10.0, "Winter", 6),
        ("Vanilla Ice Cream 500ml", "Frozen", 220.0, 15.0, "Summer", 60),
        ("Organic Salad Mix", "Produce", 95.0, 8.0, "All-Season", 6),
        ("Cheddar Cheese Block", "Dairy", 250.0, 7.0, "Winter", 40),
        ("Potato Chips Salted", "Pantry", 40.0, 22.0, "All-Season", 90),
        ("Cold Carbonated Drink 2L", "Beverage", 95.0, 16.0, "Summer", 120),
        ("Laundry Detergent 1kg", "Household", 190.0, 8.0, "All-Season", 365),
        ("Salted Butter 500g", "Dairy", 270.0, 12.0, "All-Season", 45),
        ("Fresh Red Apples 1kg", "Produce", 160.0, 11.0, "Winter", 14),
    ]

    seasons = ["Summer", "Winter", "Monsoon", "Spring"]
    months_by_season = {
        "Summer": [4, 5, 6],
        "Winter": [11, 12, 1],
        "Monsoon": [7, 8, 9],
        "Spring": [2, 3, 10]
    }

    records = []

    for i in range(num_samples):
        # Pick product template
        template = product_templates[i % len(product_templates)]
        p_name, category, base_price, base_daily_sales, peak_season, shelf_life = template

        product_id = f"PRD-{1001 + (i % 200)}"
        season = np.random.choice(seasons)
        month = np.random.choice(months_by_season[season])

        # Seasonal multiplier
        if peak_season == "All-Season":
            season_mult = np.random.uniform(0.95, 1.15)
        elif season == peak_season:
            season_mult = np.random.uniform(1.3, 1.8)
        elif (peak_season == "Summer" and season == "Winter") or (peak_season == "Winter" and season == "Summer"):
            season_mult = np.random.uniform(0.45, 0.65)
        else:
            season_mult = np.random.uniform(0.8, 1.0)

        # Promotion
        promotion = np.random.choice([0, 1], p=[0.7, 0.3])
        promo_mult = 1.25 if promotion == 1 else 1.0

        # Price variation
        price = round(base_price * np.random.uniform(0.9, 1.1), 2)

        # Daily sales rate
        daily_sales = max(1.0, round(base_daily_sales * season_mult * promo_mult * np.random.uniform(0.85, 1.15), 1))

        # Days to expiry
        # Vary days to expiry relative to shelf life
        days_to_expiry = int(max(1, np.random.exponential(scale=shelf_life * 0.4)))
        days_to_expiry = min(days_to_expiry, shelf_life)

        # Historical sales (30-day volume)
        historical_sales = int(round(daily_sales * 30 * np.random.uniform(0.8, 1.2)))

        # Previous discount & sales during discount
        prev_discount = round(float(np.random.choice([0, 5, 10, 15, 20, 25])), 1)
        discount_elasticity = 1.0 + (prev_discount / 100.0) * 1.5
        sales_during_discount = round(daily_sales * discount_elasticity, 1)

        # Supplier lead time
        lead_time = int(np.random.choice([2, 3, 5, 7, 10, 14]))

        # Expected demand before expiry
        expected_demand_before_expiry = daily_sales * days_to_expiry

        # Current stock generation with overstocking scenarios
        is_overstocked = np.random.choice([True, False], p=[0.35, 0.65])
        if is_overstocked:
            # Overstocked stock > expected demand
            current_stock = int(round(expected_demand_before_expiry * np.random.uniform(1.3, 2.5) + np.random.randint(10, 50)))
        else:
            # Normal or low stock
            current_stock = int(round(expected_demand_before_expiry * np.random.uniform(0.4, 0.95)))

        current_stock = max(5, current_stock)

        # Target label logic: Waste_Label = 1 if current stock exceeds expected demand before expiry by a margin
        # and days to expiry is relatively urgent or stock-to-demand ratio is high
        stock_excess_ratio = current_stock / (expected_demand_before_expiry + 1e-5)
        
        # Probabilistic labeling based on true economic features
        waste_score = 0.5 * (stock_excess_ratio - 1.0) + 0.3 * (1.0 / (days_to_expiry + 1)) * 10 - 0.2 * (season_mult - 1.0)
        waste_prob = 1.0 / (1.0 + np.exp(-waste_score * 2.0))
        
        waste_label = 1 if np.random.uniform(0, 1) < waste_prob and stock_excess_ratio > 1.05 else 0

        records.append({
            "Product_ID": product_id,
            "Product_Name": p_name,
            "Category": category,
            "Current_Stock": current_stock,
            "Historical_Sales": historical_sales,
            "Daily_Sales": daily_sales,
            "Selling_Price": price,
            "Previous_Discount": prev_discount,
            "Sales_During_Discount": sales_during_discount,
            "Season": season,
            "Month": month,
            "Days_to_Expiry": days_to_expiry,
            "Supplier_Lead_Time": lead_time,
            "Promotion": promotion,
            "Waste_Label": waste_label
        })

    df = pd.DataFrame(records)

    # Ensure Sunscreen exact scenario from prompt is present in row 0 for demonstration!
    df.loc[0, "Product_ID"] = "PRD-9999"
    df.loc[0, "Product_Name"] = "Sunscreen Lotion SPF 50"
    df.loc[0, "Category"] = "Personal Care"
    df.loc[0, "Current_Stock"] = 140
    df.loc[0, "Historical_Sales"] = 100
    df.loc[0, "Daily_Sales"] = 7.0
    df.loc[0, "Selling_Price"] = 500.0
    df.loc[0, "Previous_Discount"] = 10.0
    df.loc[0, "Sales_During_Discount"] = 8.5
    df.loc[0, "Season"] = "Summer"
    df.loc[0, "Month"] = 6
    df.loc[0, "Days_to_Expiry"] = 15
    df.loc[0, "Supplier_Lead_Time"] = 5
    df.loc[0, "Promotion"] = 1
    df.loc[0, "Waste_Label"] = 1

    return df

def main():
    raw_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(raw_dir, exist_ok=True)

    print("Generating synthetic supermarket dataset...")
    df_train = generate_ecobite_dataset(num_samples=1500, random_state=42)
    train_path = os.path.join(raw_dir, "supermarket_inventory.csv")
    df_train.to_csv(train_path, index=False)
    print(f"Saved training dataset with {len(df_train)} rows to {train_path}")

    # Generate sample upload test dataset
    df_test = generate_ecobite_dataset(num_samples=50, random_state=123)
    # Drop Waste_Label from sample upload test to simulate realistic manager upload (or keep optional)
    test_path = os.path.join(raw_dir, "sample_upload_test.csv")
    df_test.to_csv(test_path, index=False)
    print(f"Saved sample upload dataset with {len(df_test)} rows to {test_path}")

if __name__ == "__main__":
    main()
