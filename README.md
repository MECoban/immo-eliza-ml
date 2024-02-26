# immo-eliza-ml

## Overview

This project aims to predict house prices based on various features using different machine learning models, including Linear Regression, Random Forest, and XGBoost. It includes data preprocessing, model training, evaluation, and visualization of the results.

## Installation

To set up your environment to run these notebooks, you will need Python and the necessary libraries:

1- Clone the repository:
```
git clone https://github.com/MECoban/immo-eliza-ml.git
```

2- Go to the immo-eliza-ml directory
```
cd immo-eliza-ml
```
### Install dependencies
 Before running the code install the requirements file:

```
pip install -r requirements.txt
```

## Project Structure

* train_linear.py: Trains a Linear Regression model.
* train_forest.py: Trains a Random Forest model.
* train_xgbr.py: Trains an XGBRegressor model.
* predict_linear.py, predict_forest.py, predict_xgbr.py: Make predictions using the trained models.
* plot.py: Visualizes the actual vs. predicted prices.

## Usage

### To train and evaluate models:

1- For the Linear Regression model, run the train_linear.py script using the command:

```
python train_linear.py
```
2- For the Random Forest model run the train_forest.py script using the command:

```
python train_forest.py
```

3- For the XGBRegressor model run the train_xgbr.py script using the command:

```
python train_xgbr.py
```

### To make predictions:

1- For the Linear Regression model, run the predict_linear.py script using the command:
```
python predict_linear.py -i "data/properties.csv" -o "output/predictions_linear.csv"
```

2- For the Random Forest model run the predict_forest.py script using the command:
```
python predict_forest.py -i "data/properties.csv" -o "output/predictions_forest.csv"
```

3- For the XGBRegressor model run the predict_xgbr.py script using the command:
```
python predict_xgbr.py -i "data/properties.csv" -o "output/predictions_xgbr.csv"
```

## Visualization

* Actual Price - Predicted Price with model :

<img width="636" alt="Screenshot 2024-02-26 at 01 41 45" src="https://github.com/theomarcode/immo-eliza-ml/assets/156511598/6d611a95-a6ca-4b07-b142-db083da67652">

* Actual Price - Predicted Price with model :
  
<img width="642" alt="Screenshot 2024-02-26 at 01 42 18" src="https://github.com/theomarcode/immo-eliza-ml/assets/156511598/cd381f4a-817b-47ba-ac69-80e6aaf17e45">

* Actual Price - Predicted Price with model :

<img width="638" alt="Screenshot 2024-02-26 at 01 42 03" src="https://github.com/theomarcode/immo-eliza-ml/assets/156511598/c0807e17-8f89-4ccb-bf66-a1e0d3d25bf6">





