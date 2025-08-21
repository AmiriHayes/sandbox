import pandas as pd

data = {
    'State': ['NewYork', 'Texas', 'California', 'NewYork', 'Texas'],
    'Sales': [250, 180, 300, 120, 400],
    'Category': ['Furniture', 'Office Supplies', 'Technology', 'Furniture', 'Technology'],
    'Quantity': [3, 5, 2, 4, 1],
    'Date': pd.to_datetime(['2024-01-05', '2024-02-10', '2024-03-15', '2024-04-20', '2024-05-25'])
}

df = pd.DataFrame(data)

# code also available in adjacent ipynb file w/ results

df.head(2)
df.tail(2)
df.info()
df.describe()

df.loc[0, "Sales"]
df.iloc[2,1]

df.groupby("State")["Sales"].sum()
df.groupby("Category")["Sales"].transform("mean")

df.dropna()
df.fillna(0)
df.where(df["Sales"] > 200)

df["Sales"].apply(lambda x: x*1.1)
df["State"].map(lambda x: x.upper())

df["State"].value_counts()
df.nlargest(2, "Sales")

pd.melt(df, id_vars="State", value_vars=["Sales", "Quantity"])
df.pivot_table(index="State", values="Sales", aggfunc="mean")
df_ex = df.copy()
df_ex["CategoryList"] = [["A", "B"], ["X"], ["C", "D", "E"], ["F"], ["Y", "Z"]]
print(df_ex.explode("CategoryList"))

df.query("Sales > 200 and Quantity >=3")
df.assign(Discount=df["Sales"]*0.1)
pd.cut(df["Sales"], bins=3, labels=["Low", "Medium", "High"])

df.sort_values(by="Sales", ascending=False)
df.rename(columns={"Sales": "Revenue"})
df.duplicated()
df.drop_duplicates("State")
df.sample(2)
df.corr(numeric_only=True)