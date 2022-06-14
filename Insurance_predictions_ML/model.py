import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from pickle import dump

models_dict = {
    LogisticRegression(): {
        "solver": ["liblinear", "lbfgs"],
        "C": [1, 5]},
    KNeighborsClassifier(): {
        "n_neighbors": [4, 8],
        "p": [1, 2]
    },
    RandomForestClassifier(): {
        "n_estimators": [100, 300],
        "max_depth": [3, 6]
    },
    GradientBoostingClassifier(): {
        "n_estimators": [100, 300],
        "max_depth": [3, 6]
    },
    XGBClassifier(): {
        "n_estimators": [100, 300],
        "max_depth": [3, 6]
    }
}
train = pd.read_csv("./data/1year_preprocessed/1year_train_smotek")
test = pd.read_csv("./data/1year_preprocessed/1year_test")
X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
train_normalizer = Normalizer().fit(X_train)
X_train = train_normalizer.transform(X_train)
X_test = train_normalizer.transform(X_test)


def best_model(features, labels, model_dicts, cv=3, scoring="recall"):
    """
    Parameters:
        features (pd.DataFrame/np.ndarray): train features
        labels (pd.Series/np.array): train targets
        model_dicts (dict): a dictionary composed by key (model): values (param grid)
                            i.e. LogisticRegression(): {
                                                        "penalty": ["l1", "l2"],
                                                        "C": [1, 5]}
        cv (int): number of split for cross val
        scoring= "str" metric to be optimized
    Returns:
        list: ranking of the models sorted by recall-score composed by model name, best_params, recall_score
    """

    results = []
    for model, param_grid in model_dicts.items():
        gsv = GridSearchCV(model, [param_grid], cv=cv, scoring=scoring)
        gsv.fit(features, labels)
        results.append([model, gsv.best_params_, gsv.best_score_])
    results = sorted(results, key=lambda x: x[2], reverse=True)
    return results

#model_ranks = best_model(X_train, y_train, models_dict, cv=3, scoring="recall")
#print("The best model in {} with a recall of {:.2f}".format(model_ranks[0][0], model_ranks[0][2]))


param_grid_gbc = {
        "n_estimators": [300, 400],
        "learning_rate": [0.1, 1],
        "max_depth": [8, 16]
}

param_grid_xbc = {
    "eta": [0.3, 1],  # alias for learning rate
    "max_depth": [8, 16],
    "n_estimators": [300, 400],
}

if __name__ == "__main__":
    gbc_cv = GridSearchCV(GradientBoostingClassifier(), param_grid=param_grid_gbc,
                          cv=3, scoring="recall")
    gbc_cv.fit(X_train, y_train)
    #dump(train_normalizer, open('data/3year_preprocessed/3year_scaler.pkl', 'wb'))
    #dump(gbc_cv, open('data/3year_preprocessed/3year_model.pkl', 'wb'))
    print(gbc_cv.best_params_)
    y_pred = gbc_cv.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['survived', 'default'])
    ax.yaxis.set_ticklabels(['survived', 'default'])
    print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1']))
    plt.show()
