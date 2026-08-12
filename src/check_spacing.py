from src.eda import load_data

df = load_data()
print(df["timestamp"].diff().value_counts().head(10))