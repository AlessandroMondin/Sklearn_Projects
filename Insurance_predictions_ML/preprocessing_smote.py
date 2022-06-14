import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from utils import (
            arrf_loader,
            clean_multicollinearity,
            col_with_nans,
            clean_nan,
            seed_everything,
            binary_pie_chart,
            shapiro_distribution,
            lilliefors_test,
            outliers_handler
)

seed_everything()
# collecting data used to create a json that will be used in the lambda function in Sagemaker
#lambda_fx = {}

# load the data
year1 = arrf_loader("data/1year.arff")
year2 = arrf_loader("data/2year.arff")
year3 = arrf_loader("data/3year.arff")


year1["class"] = np.where(year1["class"] == b'0', 0, 1)
year2["class"] = np.where(year2["class"] == b'0', 0, 1)
year3["class"] = np.where(year3["class"] == b'0', 0, 1)

# STEPS:
# 1. Missing values
# 2. Removing outliers
# 3. Down-sampling/Up-sampling

"""
#class_1 = len(year1[year1["class"] == 1])
#class_0 = len(year1[year1["class"] == 0])
#binary_pie_chart(class_1, class_0, x_label="bankrupt", y_label="survived")
"""
# since the dataset it's extremely imbalanced (class "1" represent the ~3.8 % of the entire dataset)

df = year1
train, test = train_test_split(df, test_size=0.25, stratify=df.iloc[:, -1])

# even if usually it should be better to avoid to fill values in the
# test set and to keep it untouched, due to very imbalanced classes we are obliged
# to fill the values instead of dropping them. If we deleted each observation of class_1
# containing at least 1 nan, the class_1 observation would drop from 61 to 8.
# notice that we have to apply for fillna() in the test-set the train median
train = clean_nan(train, fill_value=train.median(), train=True)
test = clean_nan(test, fill_value=train.median(), train=True)

#lambda_fx["train_median"] = train.median().tolist()

assert col_with_nans(train) == [], "NaN values are present"
assert col_with_nans(test) == [], "NaN values are present"


# I HAVE TO MODIFY THE FUNCTION TO APPLY THE TRANSFORMATION TO TEST SET TOO
#print(np.percentile(train["Attr1"], 0), np.percentile(train["Attr1"], 1))
not_norm_cols = shapiro_distribution(train)
#lambda_fx["not_norm_cols"] = not_norm_cols
# We are clipping the values of for each column: each value below and above the 1th and 99th percentiles
# are clipped to those values. As regard the ones of the test set, they are clipped to the w.r.t. the
# value of the training set
#train["Attr1"].plot.hist(bins=10)
#plt.show()
q1 = train[not_norm_cols].quantile(0.01)
q2 = train[not_norm_cols].quantile(0.99)
#lambda_fx["q1_q2"] = [q1.values.tolist(), q2.values.tolist()]


train, test = outliers_handler(train, test, floor_percentile=0.01, top_percentile=0.99,
                                columns=not_norm_cols, method="clip")

#train["Attr1"].plot.hist(bins=10)
#plt.show()
#print(np.percentile(train["Attr1"], 0), np.percentile(train["Attr1"], 1))


# check if any feature is exponentially distributed to apply log-transformation/binning
"""
#exp_cols = lilliefors_test(train)
#print(exp_cols)
"""
# after removing extreme outliers though, the no column has an exponential distribution


# len(year1.columns) = 65
# after applying clean_multicollinearity()
# the number of columns is reduced to 33
# alternatives to this approach would be:
# 1) PCA
# 2) Grouping features in highly correlated clusters
#    and then with a demain expert select the feature to be kept

corr_cols = clean_multicollinearity(train.iloc[:, :-1], 0.9)
#lambda_fx["corr_cols"] = corr_cols
train = train.drop(columns=corr_cols, axis=1).reset_index().drop(["index"],axis=1)
test = test.drop(columns=corr_cols, axis=1).reset_index().drop(["index"],axis=1)

assert train.columns.equals(test.columns), "error in clean_multicollinearity()"

#with open("lambda_sagemaker.json", "w") as f:
#    json.dump(lambda_fx, f)
smt = SMOTETomek(sampling_strategy=0.25, random_state=42)
X_smotek, y_smotek = smt.fit_resample(train.iloc[:, :-1], train.iloc[:, -1])
smotek_train = pd.concat([X_smotek, y_smotek], axis=1)

smotek_train.to_csv("./data/1year_preprocessed/1year_train_smotek", index=False)
#train.to_csv("./data/3year_preprocessed/3year_train", index=False)
#test.to_csv("./data/3year_preprocessed/3year_test", index=False)


