import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from scipy.stats import shapiro
import os
import random
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import lilliefors


def arrf_loader(path):
    """
    Parameters:
        path (path): path to the interester ARRF file
    Returns:
        pd.DataFrame: the file loaded in a pd.DataFrame
    """
    raw_data = loadarff(path)
    df_data = pd.DataFrame(raw_data[0])
    return df_data


def clean_multicollinearity(df, threshold):
    """
    Parameters:
        df (pd.Dataframe): pandas dataframe with collinear features
        threshold (float): the threshold over which we discard a features due to
                            multi-collinearity with a feature considered previously
    Returns:
        list: list of the columns that are linearly correlated with other features
    """
    col_corr = []
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold and (corr_matrix.columns[i] not in col_corr):
                col_name = corr_matrix.columns[i]
                col_corr.append(col_name)
    return col_corr


def col_with_nans(df):
    """
    Parameters:
        df (pd.Dataframe)
    Returns:
        list: list of the columns that contain nans
    """
    col_nan = []
    for col in df.columns:
        if df[col].isnull().values.any():
            col_nan.append(col)
    return col_nan


# this function drops the columns of the df whose number of nans is above a set percentage
# drops the rows of the df whose number of nans is above a set percentage,
# fills with -1 the remaining nan values
def clean_nan(df, col_threshold=0.5, row_threshold=0.4, fill_value=-1, train=True):
    """
    Parameters:
        df (pd.Dataframe)
        col_threshold (float): min % of nan values in a col to drop it
        row_threshold (float): min % of nan values in a row to drop it
        fill_value: number(s) to fill nans with
        train (boolean): if the True we fill nan, otherwise we drop them
    Returns:
        df (pd.Dataframe)
    """
    df = df.dropna(axis=1, thresh=int(row_threshold*len(df)))
    df = df.dropna(axis=0, thresh=int(col_threshold*len(df.columns)))
    if train:
        # replacing with df.median() should be more robust w.r.t outliers
        df = df.fillna(fill_value)
    else:
        # We don't want to artificially modify the test set
        df = df.dropna(axis=0, how="any")
    return df


def binary_pie_chart(x, y, x_label="first category", y_label="second category"):
    """
    Parameters:
        x (int): number of observation of the x class
        y (int): number of observation of the y class
        x_label (str)
        y_label (str)
    """
    plt.figure(figsize=(20,15))
    labels = x_label, y_label
    sizes = [x, y]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.show()


# checks the distribution of each column based on a 5000 observation sample
def shapiro_distribution(df):
    """
    Parameters:
        df (pd.Dataframe)
    Returns:
        not_normal_cols (list): list of column distribution is not normal according to the
        Shapiro-Wilk Test
    """
    not_normal_cols = []
    for col in df.columns:
        _, p = shapiro(df[col].sample(5000)) if len(df) > 5000 else shapiro(df[col])
        if p < 0.05:
            not_normal_cols.append(col)
    return not_normal_cols


# discards observations that contain values that go beyond the percentiles range
def outliers_handler(train, test, floor_percentile, top_percentile, columns=False, method="drop"):
    """
    Parameters:
        df (pd.Dataframe)
        floor_percentile (float): the floor percentile under which we discard the observations
        top_percentile (float): the top percentile over which we discard the observations
        columns (list):list of columns that we want to consider for outlier deletion, if not
                        specified the whole dataset will be considered
        method (str) : indicates whether the values beyond floor and top will be clipped or dropped
    Returns:
        df (pd.Dataframe): dataframe cleaned from outliers
    """
    assert method in ("drop", "clip"), 'the available methods are "clip" and "drop"'
    assert floor_percentile < 1, "floor_percentile in expressed between 0 and 1"
    assert top_percentile <= 1, "top_percentile in expressed between 0 and 1"

    if columns:
        q1 = train[columns].quantile(floor_percentile)
        q3 = train[columns].quantile(top_percentile)
    else:
        q1 = train.quantile(floor_percentile)
        q3 = train.quantile(top_percentile)

    if method == "drop":
        iqr = q3 - q1
        if columns:
            train = train[~((train[columns] < (q1 - 1.5 * iqr)) | (train[columns] > (q3 + 1.5 * iqr))).any(axis=1)]
            test = test[~((test[columns] < (q1 - 1.5 * iqr)) | (test[columns] > (q3 + 1.5 * iqr))).any(axis=1)]

        else:
            train = train[~((train < (q1 - 1.5 * iqr)) | (train > (q3 + 1.5 * iqr))).any(axis=1)]
            test = test[~((test < (q1 - 1.5 * iqr)) | (test > (q3 + 1.5 * iqr))).any(axis=1)]

    else:
        if columns:
            train[columns] = train[columns].clip(q1, q3, axis=1)
            test[columns] = test[columns].clip(q1, q3, axis=1)

        else:
            train = train.clip(q1, q3, axis=1)
            test = test.clip(q1, q3, axis=1)

    return train, test


def lilliefors_test(df):
    """
    Parameters:
        df (pd.Dataframe)
    Returns:
        not_normal_cols (list): list of columns who have an exponential
                                distribution with Lilliefors’ test.
    """
    exp_cols = []
    for col in df.columns:
        _, p = lilliefors(df[col], dist="exp")
        if p > 0.05:
            exp_cols.append(col)
    return exp_cols


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False








