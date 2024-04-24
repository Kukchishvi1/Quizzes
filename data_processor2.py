import ipaddress
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

df = pd.read_csv("Darknet.csv")
df = df.drop(["Flow ID", "Timestamp", "Label2", "Src IP", "Dst IP"], axis=1)
df = df.dropna()

label_encoder1 = LabelEncoder()
df['Label1'] = label_encoder1.fit_transform(df['Label1'])

df.to_csv("processed.csv")

# scaler = StandardScaler()
# dfs = scaler.fit_transform(df)
#
# dfs.to_csv("processed.csv")


