import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

print("Load the training, test and store data using pandas")
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}
train = pd.read_csv("../data/train.csv", parse_dates=[2], dtype=types)
test = pd.read_csv("../data/test.csv", parse_dates=[3], dtype=types)
store = pd.read_csv("../data/store.csv")

train.fillna(1, inplace=True)
test.fillna(1, inplace=True)
train = train[train["Open"] != 0]
train = train[train["Sales"] > 0]
train = pd.merge(train, store, on='Store')


def build_features_store(store, train):
    train = train.copy()
#     train['Sales'] = train['Sales'].apply(np.log1p)
    train['Year'] = train.Date.dt.year
    train['Month'] = train.Date.dt.month
    train['Day'] = train.Date.dt.day
    train['DayOfWeek'] = train.Date.dt.dayofweek
    train['WeekOfYear'] = train.Date.dt.weekofyear
    
    store_ids = store.Store.unique()
    temp = 0
    for store_id in store_ids:
        # for day in train.DayOfWeek.unique():
        for day in range(0, 6): # this removes sunday, which is useless
            keyword = '_SalesMeanDoW'
            name_median = keyword + str(day)
            store.loc[store.Store == store_id, name_median] = train[(train.Store == store_id) & (train.DayOfWeek == day)].Sales.median()

        # for day in train.DayOfWeek.unique():
        for day in range(0, 6): # this removes sunday, which is useless
            for promo in train.Promo.unique():
                keyword = '_SalesMeanDoWPromo'
                name_median = keyword + str(day) + "_" + str(promo)
                store.loc[store.Store == store_id, name_median] = train[(train.Store == store_id) & (train.DayOfWeek == day) & (train.Promo == promo)].Sales.median()
        
        print str(temp) + " / " + str(len(store_ids))
        temp = temp + 1

    store.fillna(0, inplace=True)

print "starting!"
build_features_store(store, train)
print store.head()
store.to_csv("../data/store_features_new.pd", index=False)

