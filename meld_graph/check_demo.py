import pandas as pd

df = pd.read_csv("/Users/cll/meld_graph/meld_data/demographics_file_H1.csv")
print([col for col in df.columns])