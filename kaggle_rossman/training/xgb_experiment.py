import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt
import cPickle as pickle

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])

    # Label encode some features
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    # Calculate time competition open time in months
    features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
        (data.Month - data.CompetitionOpenSinceMonth)
    # Promo open time in months
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Indicate that sales on that day are in promo interval
    features.append('IsPromoMonth')
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    return data


## Start of main script
print("Load the training, test and store data using pandas")
features = []
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}
train = pd.read_csv("../data/train.csv", parse_dates=[2], dtype=types)
test = pd.read_csv("../data/test.csv", parse_dates=[3], dtype=types)
store = pd.read_csv("../data/store_features.pd")
for feature in store.columns:
    if '_' in feature:
        features += [feature]

print("Assume store open, if not provided")
train.fillna(1, inplace=True)
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]
print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
train = train[train["Sales"] > 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

print("augment features")
build_features(features, train)
build_features([], test)
print(features)

print('training data processed')

params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.35,
          "max_depth": 11,
          "subsample": 1.0,
          "colsample_bytree": 0.40,
          "min_child_weight": 1.1,
          "silent": 1,
          "seed": 1337
          }
num_boost_round = 73


# [11]  train-rmspe:0.191163  eval-rmspe:0.134395
# [12]  train-rmspe:0.187263  eval-rmspe:0.128288
# [13]  train-rmspe:0.185512  eval-rmspe:0.124439
# [14]  train-rmspe:0.185153  eval-rmspe:0.122948
# [15]  train-rmspe:0.182950  eval-rmspe:0.120298
# [16]  train-rmspe:0.182970  eval-rmspe:0.119706
# [17]  train-rmspe:0.182012  eval-rmspe:0.119018
# [18]  train-rmspe:0.172508  eval-rmspe:0.117801
# [19]  train-rmspe:0.169209  eval-rmspe:0.116700
# [20]  train-rmspe:0.167377  eval-rmspe:0.115484
# [21]  train-rmspe:0.166650  eval-rmspe:0.114562
# [22]  train-rmspe:0.166252  eval-rmspe:0.114144
# [23]  train-rmspe:0.164865  eval-rmspe:0.112933
# [24]  train-rmspe:0.163453  eval-rmspe:0.111712
# [25]  train-rmspe:0.162684  eval-rmspe:0.111568
# [26]  train-rmspe:0.162153  eval-rmspe:0.111253
# [27]  train-rmspe:0.161210  eval-rmspe:0.110834
# [28]  train-rmspe:0.160644  eval-rmspe:0.110478
# [29]  train-rmspe:0.159614  eval-rmspe:0.108982
# [30]  train-rmspe:0.159369  eval-rmspe:0.108736
# [31]  train-rmspe:0.158308  eval-rmspe:0.107844
# [32]  train-rmspe:0.158073  eval-rmspe:0.107647
# [33]  train-rmspe:0.157293  eval-rmspe:0.106723
# [34]  train-rmspe:0.156633  eval-rmspe:0.105839
# [35]  train-rmspe:0.155605  eval-rmspe:0.104758
# [36]  train-rmspe:0.155298  eval-rmspe:0.104389
# [37]  train-rmspe:0.154982  eval-rmspe:0.104255
# [38]  train-rmspe:0.154349  eval-rmspe:0.103980
# [39]  train-rmspe:0.152395  eval-rmspe:0.103173
# [40]  train-rmspe:0.152337  eval-rmspe:0.103100
# [41]  train-rmspe:0.151826  eval-rmspe:0.102691
# [42]  train-rmspe:0.151300  eval-rmspe:0.102494
# [43]  train-rmspe:0.151044  eval-rmspe:0.102348
# [44]  train-rmspe:0.150598  eval-rmspe:0.102050
# [45]  train-rmspe:0.147569  eval-rmspe:0.101949
# [46]  train-rmspe:0.147358  eval-rmspe:0.101762
# [47]  train-rmspe:0.146927  eval-rmspe:0.101680
# [48]  train-rmspe:0.144809  eval-rmspe:0.101360
# [49]  train-rmspe:0.144559  eval-rmspe:0.101115
# [50]  train-rmspe:0.144387  eval-rmspe:0.101002
# [51]  train-rmspe:0.144092  eval-rmspe:0.100723
# [52]  train-rmspe:0.143926  eval-rmspe:0.100588
# [53]  train-rmspe:0.143647  eval-rmspe:0.101516
# [54]  train-rmspe:0.143039  eval-rmspe:0.101024
# [55]  train-rmspe:0.142826  eval-rmspe:0.100946
# [56]  train-rmspe:0.142721  eval-rmspe:0.100909
# [57]  train-rmspe:0.142658  eval-rmspe:0.100883
# [58]  train-rmspe:0.141500  eval-rmspe:0.100316
# [59]  train-rmspe:0.140833  eval-rmspe:0.100062
# [60]  train-rmspe:0.140814  eval-rmspe:0.099877
# [61]  train-rmspe:0.140294  eval-rmspe:0.099907
# [62]  train-rmspe:0.140015  eval-rmspe:0.099879
# [63]  train-rmspe:0.132897  eval-rmspe:0.099718
# [64]  train-rmspe:0.132769  eval-rmspe:0.099659
# [65]  train-rmspe:0.132616  eval-rmspe:0.099575
# [66]  train-rmspe:0.132391  eval-rmspe:0.099501
# [67]  train-rmspe:0.132280  eval-rmspe:0.099406
# [68]  train-rmspe:0.132133  eval-rmspe:0.099318
# [69]  train-rmspe:0.131836  eval-rmspe:0.099203
# [70]  train-rmspe:0.131571  eval-rmspe:0.099159
# [71]  train-rmspe:0.114509  eval-rmspe:0.099056
# [72]  train-rmspe:0.114237  eval-rmspe:0.098882


print("Train a XGBoost model")
X_train, X_valid = train_test_split(train, test_size=0.025, random_state=1337)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)

print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
pickle.dump(yhat, open('../data/xgb_valid', 'wb'))
error = rmspe(X_valid.Sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)
# Make Submission
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
result.to_csv("../data/xgboost_submission.csv", index=False)
