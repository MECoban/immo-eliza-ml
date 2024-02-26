import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("data/properties.csv")
n_data = data["price"]

pre = pd.read_csv("output/predictions_linear.csv")

assert len(data) == len(pre)
sns.scatterplot(x=data["price"], y=pre["predictions_linear"], alpha=0.5)
plt.title("Actual vs. Predicted Prices with LinearRegression")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()


pre2 = pd.read_csv("output/predictions_forest.csv")
assert len(data) == len(pre2)
sns.scatterplot(x=data["price"], y=pre2["predictions_forest"], alpha=0.5)
plt.title("Actual vs. Predicted Prices with forest")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()


pre3 = pd.read_csv("output/predictions.csv")
assert len(data) == len(pre3)
sns.scatterplot(x=data["price"], y=pre3["predictions"], alpha=0.5)
plt.title("Actual vs. Predicted Prices with XGB")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()
