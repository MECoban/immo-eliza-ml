import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.metrics import mean_absolute_error


def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = ["latitude", "longitude", "total_area_sqm", "surface_land_sqm", "nbr_frontages","construction_year"]
    fl_features = ["fl_terrace","fl_furnished", "fl_garden", "fl_swimming_pool"]
    cat_features = ["zip_code", "region", "property_type", "province", "locality", "epc", "heating_type", "equipped_kitchen", "state_building", "nbr_bedrooms"] #Categorical features


    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean") # missing values = mean, median, most_frequent or constant
    imputer.fit(X_train[num_features]) # Fit the imputer to the data
    X_train[num_features] = imputer.transform(X_train[num_features]) # Transform the data by replacing NaN values with the imputed values
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

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    #Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(mae)
    df = pd.DataFrame(data = {"actual_values": y_test, "predicted_values" : y_pred})
    df["difference"] = df["predicted_values"] - df["actual_values"]
    df.to_csv('data/mae_linear.csv', index=False)


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
    joblib.dump(artifacts, "models/artifacts_linear.joblib" , compress=3)




if __name__ == "__main__":
    train()