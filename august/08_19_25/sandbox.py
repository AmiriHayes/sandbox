import pandas as pd

data = {
    'State': ['NewYork', 'Texas', 'California', 'NewYork', 'Texas'],
    'Sales': [250, 180, 300, 120, 400],
    'Category': ['Furniture', 'Office Supplies', 'Technology', 'Furniture', 'Technology'],
    'Quantity': [3, 5, 2, 4, 1],
    'Date': pd.to_datetime(['2024-01-05', '2024-02-10', '2024-03-15', '2024-04-20', '2024-05-25'])
}

df = pd.DataFrame(data)

# ----------------- Exploration -----------------
# print(df.head(2))
# print(df.tail(2))
# print(df.info())
# print(df.describe)

# ----------------- Selection -----------------
# print(df.loc[0, "Sales"])
print(df.iloc[2,1])

# ----------------- Aggregation -----------------
