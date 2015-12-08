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
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])

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
# num_boost_round = 73
num_boost_round = 200

# original
# eta=0.35, depth=11, min_child=1.1
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

# added store
# [42]  train-rmspe:0.115287  eval-rmspe:0.100869
# [43]  train-rmspe:0.115182  eval-rmspe:0.100791
# [44]  train-rmspe:0.114926  eval-rmspe:0.100784
# [45]  train-rmspe:0.114618  eval-rmspe:0.100676
# [46]  train-rmspe:0.113982  eval-rmspe:0.100622
# [47]  train-rmspe:0.113034  eval-rmspe:0.099769
# [48]  train-rmspe:0.112561  eval-rmspe:0.099486
# [49]  train-rmspe:0.112286  eval-rmspe:0.099286
# [50]  train-rmspe:0.112274  eval-rmspe:0.099283
# [51]  train-rmspe:0.112123  eval-rmspe:0.099244
# [52]  train-rmspe:0.111837  eval-rmspe:0.099207
# [53]  train-rmspe:0.111097  eval-rmspe:0.098792
# [54]  train-rmspe:0.110567  eval-rmspe:0.098462
# [55]  train-rmspe:0.110503  eval-rmspe:0.098456
# [56]  train-rmspe:0.110309  eval-rmspe:0.098357
# [57]  train-rmspe:0.110198  eval-rmspe:0.098282
# [58]  train-rmspe:0.109666  eval-rmspe:0.097941
# [59]  train-rmspe:0.109390  eval-rmspe:0.097787
# [60]  train-rmspe:0.109278  eval-rmspe:0.097691
# [61]  train-rmspe:0.108907  eval-rmspe:0.097624
# [62]  train-rmspe:0.108604  eval-rmspe:0.097438
# [63]  train-rmspe:0.108465  eval-rmspe:0.097351
# [64]  train-rmspe:0.107837  eval-rmspe:0.096968
# [65]  train-rmspe:0.101680  eval-rmspe:0.097094
# [66]  train-rmspe:0.101215  eval-rmspe:0.096996
# [67]  train-rmspe:0.101105  eval-rmspe:0.096889
# [68]  train-rmspe:0.100764  eval-rmspe:0.096762
# [69]  train-rmspe:0.100489  eval-rmspe:0.096715
# [70]  train-rmspe:0.096814  eval-rmspe:0.096459
# [71]  train-rmspe:0.096586  eval-rmspe:0.096215
# [72]  train-rmspe:0.096309  eval-rmspe:0.096811
# [73]  train-rmspe:0.096064  eval-rmspe:0.096726
# [74]  train-rmspe:0.095996  eval-rmspe:0.096682
# [75]  train-rmspe:0.095911  eval-rmspe:0.096643
# [76]  train-rmspe:0.095663  eval-rmspe:0.096630
# [77]  train-rmspe:0.095581  eval-rmspe:0.096601
# [78]  train-rmspe:0.095458  eval-rmspe:0.096578
# [79]  train-rmspe:0.095223  eval-rmspe:0.096618
# [80]  train-rmspe:0.094980  eval-rmspe:0.096561
# [81]  train-rmspe:0.094566  eval-rmspe:0.096440
# [82]  train-rmspe:0.094207  eval-rmspe:0.096329
# [83]  train-rmspe:0.094176  eval-rmspe:0.096309
# [84]  train-rmspe:0.094170  eval-rmspe:0.096299
# [85]  train-rmspe:0.093980  eval-rmspe:0.096254
# [86]  train-rmspe:0.093864  eval-rmspe:0.096182
# [87]  train-rmspe:0.093807  eval-rmspe:0.096109
# [88]  train-rmspe:0.093799  eval-rmspe:0.096107
# [89]  train-rmspe:0.093499  eval-rmspe:0.096039
# [90]  train-rmspe:0.093243  eval-rmspe:0.095985
# [91]  train-rmspe:0.093184  eval-rmspe:0.095965
# [92]  train-rmspe:0.093014  eval-rmspe:0.095942
# [93]  train-rmspe:0.092984  eval-rmspe:0.095937
# [94]  train-rmspe:0.092575  eval-rmspe:0.095881
# [95]  train-rmspe:0.092372  eval-rmspe:0.095822
# [96]  train-rmspe:0.092319  eval-rmspe:0.095792
# [97]  train-rmspe:0.092262  eval-rmspe:0.095723
# [98]  train-rmspe:0.091823  eval-rmspe:0.095564
# [99]  train-rmspe:0.091725  eval-rmspe:0.095544
# [100] train-rmspe:0.091512  eval-rmspe:0.095499
# [101] train-rmspe:0.091491  eval-rmspe:0.095502
# [102] train-rmspe:0.091283  eval-rmspe:0.095450
# [103] train-rmspe:0.091203  eval-rmspe:0.095396
# [104] train-rmspe:0.091129  eval-rmspe:0.095366
# [105] train-rmspe:0.091038  eval-rmspe:0.095343
# [106] train-rmspe:0.090990  eval-rmspe:0.095356
# [107] train-rmspe:0.090967  eval-rmspe:0.095322
# [108] train-rmspe:0.090808  eval-rmspe:0.095345
# [109] train-rmspe:0.090783  eval-rmspe:0.095441
# [110] train-rmspe:0.090672  eval-rmspe:0.095428
# [111] train-rmspe:0.090610  eval-rmspe:0.095405
# [112] train-rmspe:0.090517  eval-rmspe:0.095364
# [113] train-rmspe:0.090373  eval-rmspe:0.095315
# [114] train-rmspe:0.090262  eval-rmspe:0.095277
# [115] train-rmspe:0.089834  eval-rmspe:0.095210
# [116] train-rmspe:0.089736  eval-rmspe:0.095205
# [117] train-rmspe:0.089659  eval-rmspe:0.095146
# [118] train-rmspe:0.089497  eval-rmspe:0.095188
# [119] train-rmspe:0.089368  eval-rmspe:0.095164
# [120] train-rmspe:0.089098  eval-rmspe:0.095005
# [121] train-rmspe:0.088948  eval-rmspe:0.094966
# [122] train-rmspe:0.088753  eval-rmspe:0.094934
# [123] train-rmspe:0.088690  eval-rmspe:0.094923
# [124] train-rmspe:0.088564  eval-rmspe:0.094904
# [125] train-rmspe:0.088388  eval-rmspe:0.094870
# [126] train-rmspe:0.088233  eval-rmspe:0.094817
# [127] train-rmspe:0.088047  eval-rmspe:0.094783
# [128] train-rmspe:0.087527  eval-rmspe:0.094716
# [129] train-rmspe:0.087450  eval-rmspe:0.094641
# [130] train-rmspe:0.086951  eval-rmspe:0.094645
# [131] train-rmspe:0.086947  eval-rmspe:0.094633


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
