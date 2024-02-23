import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("data/properties.csv")
n_data = data["price"]

pre = pd.read_csv("output/predictions.csv")

assert len(data) == len(pre)
sns.scatterplot(x=data["price"], y=pre["predictions"], alpha=0.5)
plt.title("Actual vs. Predicted Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()


data2 = pd.read_csv("data/mae.csv")
n_data2 = data2["actual_values"]
n_data3 = data2["predicted_values"]
n_data4 = data2["difference"]

sns.scatterplot(x=n_data2, y=len(n_data2), alpha=0.5)
plt.show()

sns.scatterplot(x=n_data2, y=n_data3, alpha=0.5)
plt.show()

sns.scatterplot(x=n_data4, y=len(n_data3), alpha=0.5)
plt.show()



