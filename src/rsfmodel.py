import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.base import BaseEstimator, TransformerMixin
from sksurv.metrics import concordance_index_censored
from sklearn.utils.class_weight import compute_sample_weight

rangerPkg = importr("ranger")
SEED = 42
_TIME_COL  = "TIME"
_EVENT_COL = "EVENTMCI"


def _to_r(df):
    """Convert pandas DataFrame to R DataFrame."""
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


class RandomSurvivalForest(BaseEstimator, TransformerMixin):

    def __init__(self, num_trees=500, min_node_size=5,
                 mtry=10, splitrule="C", importance="none", compute_weights=True,
                 time_col="TIME", event_col="EVENT_MCI"):
        self.num_trees     = num_trees
        self.min_node_size = min_node_size
        self.mtry          = mtry
        self.splitrule     = splitrule
        self.importance    = importance
        self.time_col      = time_col    # original name in dataset e.g. "EVENT_MCI"
        self.event_col     = event_col   # original name in dataset e.g. "TIME"
        self.compute_weights = compute_weights

    def fit(self, X, y):
        ro.r('library(survival)')
        ro.r('library(ranger)')

        self._original_columns_naming = X.columns.to_list()
        df = pd.DataFrame(X).assign(
            TIME     = y[self.time_col].astype(float),
            EVENTMCI = y[self.event_col].astype(int)
        )

        df = self.__clean_column_names(df)
        clean_names = [col for col in df.columns
                       if col not in [_TIME_COL, _EVENT_COL]]
        self._col_mapping   = dict(zip(clean_names, self._original_columns_naming))
        self._feature_names = clean_names

        n_features = X.shape[1]
        if self.mtry == "sqrt":
            mtry = max(1, int(np.sqrt(n_features)))
        elif self.mtry == "log2":
            mtry = max(1, int(np.log2(n_features)))
        else:
            mtry = max(1, int(self.mtry))
        
        if self.compute_weights:
            weights  = compute_sample_weight("balanced", y[self.event_col])
            case_weights_r = ro.FloatVector(weights)
        else:
            case_weights_r = ro.NULL

        # always use internal R-safe names in the formula
        self.model_ = rangerPkg.ranger(
            ro.Formula("Surv(TIME, EVENTMCI) ~ ."),
            data          = _to_r(df),
            num_trees     = int(self.num_trees),
            min_node_size = int(self.min_node_size),
            mtry          = mtry,
            splitrule     = self.splitrule,
            num_threads   = 0,
            seed          = SEED,
            importance    = self.importance,
            case_weights  = case_weights_r
        )
        return self

    def predict(self, X):
        X_clean = self.__clean_column_names(pd.DataFrame(X).copy())
        pred = rangerPkg.predict_ranger(self.model_, data=_to_r(X_clean))
        return self.__field_extractor(pred, "chf").sum(axis=1)

    def predict_survival_function(self, X):
        X_clean = self.__clean_column_names(pd.DataFrame(X).copy())
        pred  = rangerPkg.predict_ranger(self.model_, data=_to_r(X_clean))
        surv  = self.__field_extractor(pred, "survival")
        times = self.__field_extractor(self.model_, "unique.death.times")
        return surv, times

    def get_importance(self):
        importance_values = np.array(self.model_.rx2("variable.importance"))
        feature_names     = list(
            self.model_.rx2("forest").rx2("independent.variable.names")
        )
        # translate cleaned names back to original dataset column names
        original_names = self.__get_original_columns(feature_names)

        feature_importance = pd.DataFrame(
            {"importances_mean": importance_values},
            index=original_names
        )
        feature_importance["importances_mean_abs"] = np.abs(
            feature_importance["importances_mean"]
        )
        return feature_importance.sort_values(
            by="importances_mean_abs", ascending=False
        )

    def transform(self, X):
        return self.predict(X).reshape(-1, 1)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def score(self, X, y):
        # use original column names from dataset
        c, *_ = concordance_index_censored(
            y[self.event_col].astype(bool),
            y[self.time_col],
            self.predict(X)
        )
        return c

    # ---------- Private Methods ----------

    def __get_original_columns(self, clean_names):
        return [self._col_mapping.get(name, name) for name in clean_names]

    def __field_extractor(self, model, field_name):
        return np.array(model.rx2(field_name))

    def __clean_column_names(self, df):
        """
        R formula interface does not allow special characters in column names.
        Replace anything that is not a letter, number or underscore with underscore.
        Column names starting with a digit are prefixed with X.
        """
        df = df.copy()
        df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
        df.columns = ['X' + col if col[0].isdigit() else col
                      for col in df.columns]
        return df

    def __transform_weight_to_r_format(self, case_weights):
        if case_weights is None:
            return ro.NULL
        else:
            return ro.FloatVector(case_weights)
