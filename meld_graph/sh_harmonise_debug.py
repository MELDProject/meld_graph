import pandas as pd

df = pd.read_csv("/Users/cll/meld_graph/meld_data/demographics_file.csv")
print(df['ID'].tolist())

print(df.columns.tolist())
