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

nested_columns = [col for col in df_json.columns if isinstance(df_json[col].iloc[0], dict)]
if nested_columns:
    normalized_dfs = []
    for col in nested_columns:
        normalized_df = pd.json_normalize(df_json[col])
        normalized_df.columns = [f"{col}.{sub_col}" for sub_col in normalized_df.columns]
        normalized_dfs.append(normalized_df)
    for col in nested_columns:
        if col in df.columns:
            combined_df = pd.concat([df.drop(columns = col)] + normalized_dfs, axis = 1)
        else:
            combined_df = pd.concat(normalized_dfs, axis = 1)
else:
    combined_df = df_json

print("After normalizing:\n", combined_df.dtypes)
print(df_json["price"])

#print(combined_df["reviews.rating"])
