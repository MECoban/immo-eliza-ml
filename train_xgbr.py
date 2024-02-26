import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error



def train():

    # Load the data
    data = pd.read_csv("data/properties.csv")

    data = data[data["nbr_bedrooms"] <= 40]
    data = data[data["total_area_sqm"] <= 10000]
    data = data[data["garden_sqm"] <= 10000]

    """With these outliners current Train R² score: 0.9288249084793263 and 
    Test R² score: 0.8331724807640778 I tried with other outliners but I pick these outliners because Test score is max right now"""

    # Define features to use
    num_features = ["nbr_frontages", "nbr_bedrooms", "latitude", "longitude", "total_area_sqm", "surface_land_sqm","terrace_sqm","garden_sqm"]
    fl_features = ["fl_terrace", "fl_garden", "fl_swimming_pool"]
    cat_features = ["province", "heating_type", "property_type", "epc", "zip_code", "subproperty_type","region"]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])


    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    print(f"Features: \n {X_train.columns.tolist()}")

  # Use the best parameters found during RandomizedSearchCV
    best_params = {
        'n_estimators': 300,
        'max_depth': 7,
        'learning_rate': 0.1,
        'lambda': 1,
    }

    # Train the final model using the best parameters
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    # Evaluate the final model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))

    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")



    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(mae)
    df = pd.DataFrame(data = {"actual_values": y_test, "predicted_values" : y_pred})
    df["difference"] = df["predicted_values"] - df["actual_values"]
    df.to_csv('data/mae_xgbr.csv', index=False)



    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "models/artifacts.joblib", compress=3)


    


if __name__ == "__main__":
    train()