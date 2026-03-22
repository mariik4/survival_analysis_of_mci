import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils.class_weight import compute_sample_weight

class CoxPH(CoxPHFitter):
    def __init__(self, penalizer=0.0, l1_ratio=0.0, compute_weights=True):
        super().__init__(penalizer=penalizer, l1_ratio=l1_ratio)
        self.compute_weights = compute_weights
        self._weights_col    = "event_weights"

    def fit(self, df, duration_col, event_col, **kwargs):
        if self.compute_weights:
            df = self._create_weights(df, event_col)
            kwargs["weights_col"] = self._weights_col
            kwargs["robust"] = True
        return super().fit(df, duration_col=duration_col, event_col=event_col, **kwargs)

    def get_feature_importance(self):
        return self.summary[["coef", "exp(coef)", "p"]]

    def _create_weights(self, df, event_col):
        weighted_df = df.copy()
        weights = compute_sample_weight("balanced", weighted_df[event_col])
        weighted_df[self._weights_col] = weights
        return weighted_df
