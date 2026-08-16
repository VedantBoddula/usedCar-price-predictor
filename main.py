import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import os
import joblib
from sklearn.model_selection import train_test_split



MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attributes, cat_attributes):

    # for numerical attributes
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy = "median")),
        ("standarized_scaler", StandardScaler())
    ])

    # for categorical attributes
    cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
    ])


    # combine both pipelines/Construct a full pipeline
    full_pipeline = ColumnTransformer([
        ("num_pipeline", num_pipeline, num_attributes),
        ("cat_pipeline", cat_pipeline, cat_attributes)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    data = pd.read_csv("used_car_price_dataset_extended.csv")

    dataFeatures = data.drop("price_usd", axis=1)
    dataLabels = data["price_usd"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
    dataFeatures,
    dataLabels,
    test_size=0.2,
    random_state=42
)

    test_data = X_test.copy()
    test_data["price_usd"] = y_test
    test_data.to_csv("input.csv", index=False)

    num_attributes = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    cat_attributes = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    full_pipeline = build_pipeline(
        num_attributes,
        cat_attributes
    )

    data_prepared = full_pipeline.fit_transform(X_train)

    model = LinearRegression()
    model.fit(data_prepared, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(full_pipeline, PIPELINE_FILE)

    print("Model and pipeline have been trained and saved. Congratulations!")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")

    # Keep actual price separately
    actual_values = input_data["price_usd"].copy()

    # Remove the label before giving data to the model
    input_features = input_data.drop("price_usd", axis=1)

    # Apply the already-fitted pipeline
    transformed_input = pipeline.transform(input_features)

    # Make predictions
    predictions = model.predict(transformed_input)

    # Remove original price from output
    input_data = input_data.drop("price_usd", axis=1)

    # Add actual and predicted prices
    input_data["actual_price_usd"] = actual_values
    input_data["predicted_price_usd"] = predictions

    # Calculate test RMSE
    final_rmse = root_mean_squared_error(actual_values, predictions)
    print("Final Test RMSE:", final_rmse)

    # Save results
    input_data.to_csv("output.csv", index=False)

    print("Inference is complete, results are saved to output.csv")

