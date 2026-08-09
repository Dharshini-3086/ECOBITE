import pandas as pd
import numpy as np

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

np.random.seed(42)

NUMBER_OF_ROWS = 2000


# ---------------------------------------------------------
# PRODUCT INFORMATION
# ---------------------------------------------------------

products = [
    ("Milk", "Dairy", "Refrigerated"),
    ("Yogurt", "Dairy", "Refrigerated"),
    ("Cheese", "Dairy", "Refrigerated"),
    ("Butter", "Dairy", "Refrigerated"),
    ("Bread", "Bakery", "Room Temperature"),
    ("Cake", "Bakery", "Refrigerated"),
    ("Apple", "Fruit", "Refrigerated"),
    ("Banana", "Fruit", "Room Temperature"),
    ("Orange", "Fruit", "Room Temperature"),
    ("Mango", "Fruit", "Room Temperature"),
    ("Tomato", "Vegetable", "Room Temperature"),
    ("Potato", "Vegetable", "Pantry"),
    ("Carrot", "Vegetable", "Refrigerated"),
    ("Spinach", "Vegetable", "Refrigerated"),
    ("Cucumber", "Vegetable", "Refrigerated"),
    ("Chicken", "Meat", "Refrigerated"),
    ("Fish", "Meat", "Refrigerated"),
    ("Eggs", "Dairy", "Refrigerated"),
    ("Rice", "Grains", "Pantry"),
    ("Pasta", "Grains", "Pantry")
]


# ---------------------------------------------------------
# CREATE DATA
# ---------------------------------------------------------

rows = []

for i in range(NUMBER_OF_ROWS):

    product, category, storage = products[
        np.random.randint(len(products))
    ]

    # Stock quantity
    stock = np.random.randint(5, 151)

    # Daily sales
    daily_sales = np.random.randint(1, 31)

    # Days remaining before expiry
    days_to_expiry = np.random.randint(1, 61)

    # Discount percentage
    discount = np.random.choice(
        [0, 5, 10, 15, 20, 25],
        p=[0.40, 0.15, 0.15, 0.10, 0.10, 0.10]
    )

    # Purchase frequency
    purchase_frequency = np.random.choice(
        [3, 5, 7, 14, 21, 30]
    )

    # Product price
    price = round(
        np.random.uniform(20, 500),
        2
    )

    # -----------------------------------------------------
    # ESTIMATE WHETHER PRODUCT WILL BE WASTED
    # -----------------------------------------------------

    expected_sales = daily_sales * days_to_expiry

    waste_score = 0

    # Too much stock compared to expected sales
    if stock > expected_sales:
        waste_score += 3

    # Very close to expiry
    if days_to_expiry <= 3:
        waste_score += 3

    elif days_to_expiry <= 7:
        waste_score += 2

    elif days_to_expiry <= 14:
        waste_score += 1

    # Low sales
    if daily_sales <= 3:
        waste_score += 2

    elif daily_sales <= 7:
        waste_score += 1

    # High purchase frequency can increase overstock
    if purchase_frequency <= 5:
        waste_score += 1

    # Discount can reduce waste
    if discount >= 15:
        waste_score -= 1

    # Add some randomness so the dataset isn't perfectly rule-based
    random_value = np.random.random()

    if waste_score >= 5:
        waste = 1

    elif waste_score >= 3:
        waste = 1 if random_value < 0.75 else 0

    elif waste_score == 2:
        waste = 1 if random_value < 0.40 else 0

    else:
        waste = 0

    rows.append([
        product,
        category,
        stock,
        daily_sales,
        days_to_expiry,
        discount,
        purchase_frequency,
        price,
        storage,
        waste
    ])


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------

columns = [
    "product",
    "category",
    "stock_quantity",
    "daily_sales",
    "days_to_expiry",
    "discount",
    "purchase_frequency",
    "price",
    "storage_method",
    "waste"
]

df = pd.DataFrame(
    rows,
    columns=columns
)


# ---------------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------------

df.to_csv(
    "supermarket_inventory.csv",
    index=False
)


# ---------------------------------------------------------
# DISPLAY INFORMATION
# ---------------------------------------------------------

print("=" * 60)
print("ECOBITE DATASET GENERATED SUCCESSFULLY")
print("=" * 60)

print(f"\nNumber of records: {len(df)}")

print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst 10 records:")
print(df.head(10))

print("\nWaste distribution:")
print(df["waste"].value_counts())

print("\nDataset saved as:")
print("supermarket_inventory.csv")