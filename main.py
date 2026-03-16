
import pandas as pd

from data_loader  import load_data, inspect_data
from preprocessor import (
    drop_id_column,
    encode_categoricals,
    split_features_target,
    split_train_test,
)
from model import (
    train,
    evaluate,
    plot_actual_vs_predicted,
    plot_coefficients,
)



DATASET_PATH = "car_price_prediction_.csv"   
TEST_SIZE    = 0.20                           
RANDOM_STATE = 42                             


def main():

    print("=" * 55)
    print("   Car Price Prediction  —  Linear Regression")
    print("=" * 55)

   
    df = load_data(DATASET_PATH)

    inspect_data(df)

    print("\n--- Preprocessing ---")
    df = drop_id_column(df)
    df = encode_categoricals(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
    )

    print("\n--- Training ---")
    model = train(X_train, y_train)

    results = evaluate(model, X_test, y_test)

    print("\n--- Plots ---")
    plot_actual_vs_predicted(y_test, results["y_pred"])
    plot_coefficients(model, list(X_train.columns))

    
    print("\n--- Single Car Prediction Demo ---")

    .
    sample = pd.DataFrame([{
        "Brand"       : X_train["Brand"].iloc[0],
        "Year"        : 2018,
        "Engine Size" : 2.3,
        "Fuel Type"   : X_train["Fuel Type"].iloc[0],
        "Transmission": X_train["Transmission"].iloc[0],
        "Mileage"     : 114832,
        "Condition"   : X_train["Condition"].iloc[0],
        "Model"       : X_train["Model"].iloc[0],
    }])

    predicted_price = model.predict(sample)[0]
    print(f"  Predicted Price : $ {predicted_price:,.2f}")

    print("\nDone.")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
