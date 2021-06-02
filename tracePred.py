import torch as tc
import pandas as pd

data = pd.read_table("nc180vali/Out/trace.out",header = 0, delim_whitespace = True)
df = data[['UXB']]
df['UXB-1'] = df['UXB'].shift(1)
df['UXB-2'] = df['UXB'].shift(2)
df['UXB-3'] = df['UXB'].shift(3)
print(df)
print(df.drop(index=df.index[[0,1,2]]))
