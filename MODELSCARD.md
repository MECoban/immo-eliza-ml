
# Models Card for immo-eliza-ml

## Data

The models are trained on a proprietary dataset `properties.csv`. The dataset consists of 75,000 properties in Belgium with the following features:

* id: Property ID 
* price: Property price 
* property_type: Property type
* subproperty_type: Subproperty type
* region: Region
* province: Province 
* locality: Locality
* zip_code: Zip code
* latitude: Latitude
* longitude: Longitude
* construction_year: Construction year
* total_area_sqm: Total area in square meters
* surface_land_sqm: Surface land in square meters
* nbr_frontages: Number of frontages
* nbr_bedrooms: Number of bedrooms
* equipped_kitchen: Equipped kitchen
* terrace_sqm: Terrace area in square meters
* garden_sqm: Garden area in square meters
* state_building: State of the building
* primary_energy_consumption_sqm: Primary energy consumption per square meter
* epc: Energy performance certificate
* heating_type: Heating type
* cadastral_income: Cadastral income
* fl_furnished: Furnished
* fl_open_fire: Open fire
* fl_terrace: Terrace
* fl_garden: Garden
* fl_swimming_pool: Swimming pool
* fl_floodzone: Flood zone
* fl_double_glazing: Double glazing

## Model Details

This project includes three different models trained to predict house prices:

- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

Each model is trained on a dataset of features to predict the house prices.


## Evaluation

The models were evaluated using the R-squared metric and Mean Absolute Error (MAE) to quantify the prediction accuracy.

- Linear Regression:
  - Train R² score: [0.45172552923213694]
  - Test R² score: [0.4592505102669192]
  - Mean Absolute Error (MAE): [145585.84]
- Random Forest:
  - Train R² score: [0.9630410317568125]
  - Test R² score: [0.7234206769130711]
  - Mean Absolute Error (MAE): [77713.55]
- XGBoost:
  - Train R² score: [0.9288249084793263]
  - Test R² score: [0.8331724807640778]
  - Mean Absolute Error (MAE): [80651.43]

## Ethical Considerations

Any decision to use these models in real-world applications should be carefully considered, taking into account potential biases in the data and the impact of incorrect predictions.

## Caveats and Recommendations

The performance of the models may vary based on the quality and characteristics of the input data. It is recommended to further tune the models and validate them on a diverse dataset before any practical application.
