import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/properties.csv")
n_data = data["price"]

pre = pd.read_csv("output/predictions.csv")

assert len(data) == len(pre)
sns.scatterplot(x=data["price"], y=pre["predictions"], alpha=0.5)
plt.title("Actual vs. Predicted Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()