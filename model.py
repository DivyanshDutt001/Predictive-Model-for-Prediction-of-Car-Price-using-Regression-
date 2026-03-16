

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics      import mean_absolute_error, mean_squared_error, r2_score


def train(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit a Linear Regression model on the training data."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained   :  LinearRegression  ✓")
    return model


def evaluate(
    model:   LinearRegression,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
) -> dict:
    
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"  MAE   : $ {mae  :>10,.2f}   (avg error per car)")
    print(f"  RMSE  : $ {rmse :>10,.2f}   (root mean squared error)")
    print(f"  R²    :   {r2   :>10.4f}   (1.0 = perfect fit)")

    return {"mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}


def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray) -> None:
   
    plt.figure(figsize=(8, 5))

    plt.scatter(
        y_test, y_pred,
        alpha      = 0.5,
        color      = "steelblue",
        edgecolors = "white",
        linewidths = 0.4,
        label      = "Predictions",
    )

    # Perfect-fit reference line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color     = "tomato",
        linewidth = 1.5,
        linestyle = "--",
        label     = "Perfect prediction",
    )

    plt.xlabel("Actual Price (USD)")
    plt.ylabel("Predicted Price (USD)")
    plt.title("Actual vs Predicted Car Prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.show()
    print("Saved plot      :  actual_vs_predicted.png")


def plot_coefficients(model: LinearRegression, feature_names: list) -> None:
    
    coef_df = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": model.coef_,
    }).sort_values("Coefficient")

    colors = ["tomato" if c < 0 else "steelblue" for c in coef_df["Coefficient"]]

    plt.figure(figsize=(8, 5))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    plt.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("Coefficient Value")
    plt.title("Linear Regression — Feature Coefficients")
    plt.tight_layout()
    plt.savefig("coefficients.png", dpi=150)
    plt.show()
    print("Saved plot      :  coefficients.png")
