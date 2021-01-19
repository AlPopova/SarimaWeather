import pandas as pd
weather_data = pd.read_csv(r'C:\Users\Александр\Downloads\Telegram Desktop\my_data.csv', sep=';', decimal=',')
import datetime
date=weather_data.apply(lambda x: datetime.date(int(x['year']), int(x['month']),int(x['day'])),axis=1)
date = pd.to_datetime(date)
weather_data = weather_data.drop(columns=['year', 'month', 'day'])
weather_data.insert(0, 'Date', date)
temp_df = weather_data[['Date','T']]
temp_df["T"] = temp_df["T"].astype(str).astype(float)
temp_df.head(10)
temp_df.shape
mask = (temp_df['Date'] >= '2005-01-01') & (temp_df['Date'] <= '2010-12-31')
temp_df = temp_df.loc[mask]
temp_df.set_index("Date", inplace=True)

predicted_df = temp_df["T"].to_frame().shift(1).rename(columns = {"T": "T_pred" })
actual_df = temp_df["T"].to_frame().rename(columns = {"T": "T_actual" })

# Concatenate the actual and predicted temperature
one_step_df = pd.concat([actual_df,predicted_df],axis=1)

# Select from the second row, because there is no prediction for today due to shifting.
one_step_df = one_step_df[1:]
one_step_df.head(10)

import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(one_step_df.T_actual,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
pred = results.get_prediction(start=pd.to_datetime('2011-01-01'),end=pd.to_datetime('2011-02-01'), dynamic=False)



from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
l=[]
for i in range(30):
    l.append(pred.predicted_mean[i])

class Item(BaseModel):
    name: str

@app.get("/items/{item_id}")
def read_item(item_id: int, q: List[float] = l):
    return {"NumberOfDays": item_id, "Temperature": q[:item_id]}


@app.put("/items/{item_id}")
def update_item(item_id: int):
    item=Item(name="AY")
    return {"item_name": item.name, "item_id": item_id}