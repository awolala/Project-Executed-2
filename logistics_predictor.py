import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Project: Predictive Healthcare Logistics System 
## Overview
#This project applies machine learning techniques to predict hospital resource demand, specifically bed occupancy, using simulated healthcare data.

#It demonstrates how AI can support healthcare planning and operational efficiency.

def run_logistics_model():
    np.random.seed(42)

    # ✅ Generate dataset
    data = {
        'staff_on_shift': np.random.randint(10, 30, 200),
        'emergency_arrivals': np.random.randint(5, 50, 200),
        'prev_day_occupancy': np.random.randint(50, 100, 200),
    }

    df = pd.DataFrame(data)

    df['beds_needed'] = (
        0.5 * df['prev_day_occupancy'] +
        0.3 * df['emergency_arrivals'] +
        0.2 * df['staff_on_shift'] +
        np.random.normal(0, 5, 200)
    ).astype(int)

    # ✅ Save dataset
    df.to_csv("healthcare_data.csv", index=False)

    # Split
    X = df.drop('beds_needed', axis=1)
    y = df['beds_needed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)

    print("\n--- Model Results ---")
    print(f"MAE: {mae:.2f}")

    # ✅ Save predictions
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': predictions
    })
    results_df.to_csv("prediction_results.csv", index=False)

    # ✅ Feature importance plot
    plot_feature_importance(model, X.columns)

    # Sample prediction
    sample = X_test.iloc[0]
    pred = model.predict([sample])[0]

    print("\nSample Input:", sample.to_dict())
    print(f"Predicted: {pred:.2f}, Actual: {y_test.iloc[0]}")


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_

    plt.figure()
    plt.bar(feature_names, importances)
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.savefig("feature_importance.png")
    plt.show()


if __name__ == "__main__":
    run_logistics_model()