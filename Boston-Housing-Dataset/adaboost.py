from sklearn.ensemble import AdaBoostRegressor
from preprocess import *

X, Y = getData()
X_test, Y_test = getTestData()

rf = AdaBoostRegressor(n_estimators=100, loss='linear', learning_rate=0.3)
rf.fit(X, Y)

def pred_ints(model, X, percentile=95):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X[x])[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up

err_down, err_up = pred_ints(rf, X_test, percentile=90)
 
truth = Y_test
correct = 0.
for i, val in enumerate(truth):
    if err_down[i] <= val <= err_up[i]:
        correct += 1
print correct/len(truth)
