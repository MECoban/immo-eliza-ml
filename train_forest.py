import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

def train():
    # Load the data
    data = pd.read_csv("data/properties.csv")

    num_features = ["total_area_sqm", "surface_land_sqm", "latitude", "longitude", "nbr_frontages", "construction_year"]
    cat_features = ["zip_code", "property_type", "locality", "epc", "nbr_bedrooms", "heating_type", "equipped_kitchen"] #Categorical features


    # Split the data into features and target
    X = data[num_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer for numerical features
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical column 'zip_code' with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train_2 = pd.concat(
        [
            pd.DataFrame(X_train[num_features], index=X_train.index),
            pd.DataFrame(X_train_cat, index=X_train.index, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )
    X_test_2 = pd.concat(
        [
            pd.DataFrame(X_test[num_features], index=X_test.index),
            pd.DataFrame(X_test_cat, index=X_test.index, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    print(f"Features: \n {X_train.columns.tolist()}")

    # Train the model using Random Forest
    model = RandomForestRegressor(n_estimators=10, random_state=505)
    model.fit(X_train_2, y_train)

    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train_2))
    test_score = r2_score(y_test, model.predict(X_test_2))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    y_pred = model.predict(X_test_2)
    mae = mean_absolute_error(y_test, y_pred)
    df = pd.DataFrame(data = {"actual_values": y_test, "predicted_values" : y_pred})
    df["difference"] = df["predicted_values"] - df["actual_values"]
    df.to_csv('data/mae_forest.csv', index=False)

    # Save the model and other artifacts
    artifacts = {
        "features": {
            "num_features": num_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "models/artifacts_forest.joblib", compress=3)

if __name__ == "__main__":
    train()