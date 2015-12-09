import pandas as pd
import numpy as np

xgb_w = 0.91
rf_w = 0
et_w = 0.09

xgb_pred = pd.read_csv("../data/xgboost_submission.csv")
rf_pred = pd.read_csv("../data/rf_submission.csv")
et_pred = pd.read_csv("../data/et_submission.csv")

submission = xgb_w * xgb_pred.Sales + rf_w * rf_pred.Sales + et_w * et_pred.Sales
result = pd.DataFrame({"Id": xgb_pred["Id"], 'Sales': submission})
result.to_csv("../data/ensemble_submission.csv", index=False)
