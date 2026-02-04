# Thomas Mackey, Sean Takahashi
# SoftDev pd 5
# P03

import pandas as pd
import numpy as np


csv_file = "air_quality.csv"
df = pd.read_csv(csv_file)




#print(df["value"].std())

df = df.dropna()


#print(df.city.unique())

df["datetime"] = pd.to_datetime(df["date.utc"])

#print("First date: ", df["datetime"].min())
#print("Last date: ", df["datetime"].max())
#print("Duration: ", df["datetime"].max() - df["datetime"].min() )

#print(df.head(10))

location_dow = df.groupby([df["datetime"].dt.weekday, "location"])["value"].mean()
#print(location_dow)

avg_dow = df.groupby([df["datetime"].dt.weekday])["value"].mean()
#print(avg_dow)

avg_hour = df.groupby([df["datetime"].dt.hour])["value"].mean(0)
#print(avg_hour)


### JSON

json_file = "file.json"
df_json = pd.read_json(json_file)
print("Before normalizing:\n", df_json.dtypes)
df_json_norm = pd.json_normalize(df_json)

print("After normalizing:\n", df_json_norm.dtypes)
#print(df_json_norm["price"])
