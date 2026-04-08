import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils.class_weight import compute_sample_weight
from lifelines.utils import concordance_index

class CoxPH(CoxPHFitter, BaseEstimator):
    def __init__(self, penalizer=0.0, l1_ratio=0.0, duration_col="TIME", event_col="EVENT_MCI", compute_weights=True):
        super().__init__(penalizer=penalizer, l1_ratio=l1_ratio)
        self.duration_col    = duration_col
        self.event_col       = event_col
        self.compute_weights = compute_weights
        self._weights_col    = "event_weights"

    def fit(self, X, y=None, **kwargs):
        print(f"\nFitting CoxPH with parameters: penalizer={self.penalizer}, l1_ratio={self.l1_ratio}, compute_weights={self.compute_weights}")
        df = self._to_frame(X, y)
        if self.compute_weights:
            df = self._create_weights(df, self.event_col)
            kwargs["weights_col"] = self._weights_col
            kwargs["robust"] = True
        super().fit(df, duration_col=self.duration_col, event_col=self.event_col, **kwargs)
        self.is_fitted_ = True
        return self

    def get_feature_importance(self):
        return self.summary[["coef", "exp(coef)", "p"]]

    def _to_frame(self, X, y):
        df = X.copy()
        df[self.duration_col] = np.asarray(y[self.duration_col])
        df[self.event_col]    = np.asarray(y[self.event_col])
        return df

    def _create_weights(self, df, event_col):
        weighted_df = df.copy()
        weights = compute_sample_weight("balanced", weighted_df[event_col])
        weighted_df[self._weights_col] = weights
        return weighted_df

    def score(self, X, y=None):
        # predict_partial_hazard ranking patients by risk
        # that mean the higher the risk => the lower probability to surviva
        # But concordance index expects scores, so we need to negate the result
        # in order to receive opposite order
        df = self._to_frame(X, y)
        hazard_prediction = self.predict_partial_hazard(df)
        predicted_scores = -hazard_prediction
        return concordance_index(df[self.duration_col], predicted_scores, df[self.event_col])
