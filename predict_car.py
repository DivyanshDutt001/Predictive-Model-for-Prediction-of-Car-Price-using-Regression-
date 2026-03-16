

import pandas as pd
import numpy  as np
from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import LabelEncoder



DATASET_PATH     = "car_price_prediction_.csv"
TARGET_COL       = "Price"
DROP_COL         = "Car ID"
CATEGORICAL_COLS = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]


BRAND_MODEL_MAP = {
    "Audi"    : ["A3", "A4", "Q5", "Q7"],
    "BMW"     : ["3 Series", "5 Series", "X3", "X5"],
    "Ford"    : ["Explorer", "Fiesta", "Focus", "Mustang"],
    "Honda"   : ["Accord", "Civic", "CR-V", "Fit"],
    "Mercedes": ["C-Class", "E-Class", "GLA", "GLC"],
    "Tesla"   : ["Model 3", "Model S", "Model X", "Model Y"],
    "Toyota"  : ["Camry", "Corolla", "Prius", "RAV4"],
}

FUEL_TYPES    = ["Petrol", "Diesel", "Electric", "Hybrid"]
TRANSMISSIONS = ["Manual", "Automatic"]
CONDITIONS    = ["New", "Used", "Like New"]
YEAR_RANGE    = (2000, 2023)
ENGINE_RANGE  = (1.0, 6.0)
MILEAGE_RANGE = (0, 300000)



def show_options(label: str, options: list) -> None:
    """Print a numbered list of choices."""
    print(f"\n  {label}:")
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")


def pick_from_list(label: str, options: list) -> str:
    """Ask user to pick a number from a list; keep asking until valid."""
    show_options(label, options)
    while True:
        raw = input(f"  Enter number (1–{len(options)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            chosen = options[int(raw) - 1]
            print(f"  Selected : {chosen}")
            return chosen
        print(f"  ✗  Please enter a number between 1 and {len(options)}.")


def pick_integer(label: str, min_val: int, max_val: int) -> int:
    """Ask user for an integer within a range."""
    while True:
        raw = input(f"  {label} ({min_val}–{max_val}): ").strip()
        if raw.isdigit() and min_val <= int(raw) <= max_val:
            return int(raw)
        print(f"  ✗  Please enter a whole number between {min_val} and {max_val}.")


def pick_float(label: str, min_val: float, max_val: float) -> float:
    """Ask user for a float within a range."""
    while True:
        raw = input(f"  {label} ({min_val}–{max_val}L): ").strip()
        try:
            val = round(float(raw), 1)
            if min_val <= val <= max_val:
                return val
        except ValueError:
            pass
        print(f"  ✗  Please enter a number between {min_val} and {max_val}.")



def train_model():
    """
    Load the full dataset, encode it, and fit a Linear Regression model.
    Returns the trained model and a dict of fitted LabelEncoders (one per
    categorical column) so the same encoding is applied at prediction time.
    """
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=[DROP_COL])

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le                    # save encoder for reuse later

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    model = LinearRegression()
    model.fit(X, y)

    return model, encoders



def get_car_details(encoders: dict) -> pd.DataFrame:
    """
    Interactively ask the user for each feature.
    Returns a single-row DataFrame ready for model.predict().
    """
    print("\n" + "─" * 50)
    print("  Enter the car details below")
    print("─" * 50)

    # Brand
    brands = sorted(BRAND_MODEL_MAP.keys())
    brand  = pick_from_list("Brand", brands)

    # Model (filtered by brand)
    models = BRAND_MODEL_MAP[brand]
    model_name = pick_from_list("Model", models)

    # Year
    year = pick_integer("Year", *YEAR_RANGE)

    # Engine Size
    engine_size = pick_float("Engine Size", *ENGINE_RANGE)

    # Fuel Type
    fuel_type = pick_from_list("Fuel Type", FUEL_TYPES)

    # Transmission
    transmission = pick_from_list("Transmission", TRANSMISSIONS)

    # Mileage
    mileage = pick_integer("Mileage (km)", *MILEAGE_RANGE)

    # Condition
    condition = pick_from_list("Condition", CONDITIONS)

   
    raw = {
        "Brand"       : brand,
        "Year"        : year,
        "Engine Size" : engine_size,
        "Fuel Type"   : fuel_type,
        "Transmission": transmission,
        "Mileage"     : mileage,
        "Condition"   : condition,
        "Model"       : model_name,
    }

    encoded = raw.copy()
    for col in CATEGORICAL_COLS:
        encoded[col] = encoders[col].transform([raw[col]])[0]

    return pd.DataFrame([encoded]), raw




def main():

    print("\n" + "=" * 50)
    print("   Car Price Predictor  —  Linear Regression")
    print("=" * 50)

    
    print("\n  Loading dataset and training model...")
    model, encoders = train_model()
    print("  Model ready ✓")

    while True:

        input_df, raw = get_car_details(encoders)

        # Predict
        predicted_price = model.predict(input_df)[0]
        predicted_price = max(predicted_price, 0)       # price can't be negative

        # Display result
        print("\n" + "═" * 50)
        print("  PREDICTION RESULT")
        print("═" * 50)
        print(f"  Brand        : {raw['Brand']}")
        print(f"  Model        : {raw['Model']}")
        print(f"  Year         : {raw['Year']}")
        print(f"  Engine Size  : {raw['Engine Size']} L")
        print(f"  Fuel Type    : {raw['Fuel Type']}")
        print(f"  Transmission : {raw['Transmission']}")
        print(f"  Mileage      : {raw['Mileage']:,} km")
        print(f"  Condition    : {raw['Condition']}")
        print("─" * 50)
        print(f"  Predicted Price  :  $ {predicted_price:>10,.2f}")
        print("═" * 50)

        # Ask to continue
        again = input("\n  Predict another car? (yes / no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n  Thank you! Goodbye.\n")
            break



if __name__ == "__main__":
    main()
