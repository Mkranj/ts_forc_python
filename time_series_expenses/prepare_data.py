import pandas as pd

full_data = pd.read_csv("data/my_expenses.csv", delimiter = ";", decimal = ",")

full_data_amounts = full_data.drop(["original", "amount_kn", "Note", "Currency"], axis=1)
full_data_amounts.head()

daily = full_data_amounts.groupby("Date")[["Amount"]].sum()

daily.index = pd.to_datetime(daily.index)
daily.index

# Get monthly amounts
monthly = daily.resample("1MS").sum()

# Remove 2021-09 because it's inaccurately low
monthly.drop("2021-09-01", axis = 0, inplace = True)

# The dataframe should have specific column names
monthly.columns = ["y"]
monthly["ds"] = monthly.index

monthly.reset_index(drop = True)

# EXPORTS
# daily
# monthly